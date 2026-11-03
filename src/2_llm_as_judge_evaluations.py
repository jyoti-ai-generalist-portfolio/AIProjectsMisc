import os
import csv
from pydantic import BaseModel,Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq 



# 1. Structured Output Schema for the Judge
class EvaluationSchema(BaseModel):
    rationale: str = Field(description="Step-by-step reasoning evaluating the student output against the rubric and context.")
    verdict: bool = Field(description="True if the student answer is fully faithful and supported by the context. False if it contains hallucinations or contradictions.")
    #score: int = Field(..., description="Score from 1 to 10")
    #feedback: str = Field(..., description="Detailed feedback on the evaluation")
    #suggestions: Optional[List[str]] = Field(None, description="Optional suggestions for improvement")

#LLM Judge Class
class RAGFaithfulnessJudge:
    def __init__(self, model_name: str = "openai/gpt-oss-120b", api_key: Optional[str] = None):
        self.model_name = model_name
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")  # Use GROQ_API_KEY for Groq API
        if not self.api_key:
            raise ValueError("API key must be provided either as an argument or via the OPENAI_API_KEY environment variable.")
        # Initialize the OpenAI client with the provided API key
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        
        self.system_prompt = (
            "You are an expert, unbiased AI Quality Assurance Auditor executing a critical RAG Faithfulness Evaluation.\n\n"
            "CRITERIA FOR FAITHFULNESS:\n"
            "- Every assertion, metric, or factual claim made in the Student Answer MUST be directly traceable to the provided Context.\n"
            "- Extrapolations or logic leaps that cannot be proven by the text must be treated as unfaithful.\n\n"
            "INSTRUCTIONS:\n"
            "1. Extract all factual claims made in the Student Answer.\n"
            "2. Audit each claim against the Context one by one.\n"
            "3. Document your audit trail in the 'rationale' field before determining the final verdict.\n"
            "4. Output 'verdict' as true if and only if all claims pass. Otherwise, output false."
        )
    def evaluate(self, context: str, student_answer: str) -> EvaluationSchema:
        content = "" 
        #print ("Evaluating with context: ", context)
        #print ("Self.system_prompt: ", self.system_prompt)

        try:
            if (not context.strip()) or (not student_answer.strip()):
                raise ValueError("Both context and student_answer must be non-empty strings.")
            user_prompt = (
                f"CONTEXT:\n{context}\n\n"
                f"STUDENT ANSWER:\n{student_answer}\n\n"
                "Please provide your evaluation in the following JSON format:\n"
                "{\n"
                '  "rationale": "...",\n'
                '  "verdict": true/false\n'
                "}"
            )

            
            response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "system", "content": self.system_prompt},
                                  {"role": "user", "content": user_prompt}],
                            temperature=0.0
                        )
            
            #print ("Raw Evaluation Response: ", response)
            # Extract the content from the response
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
            #print ("Raw Evaluation Response: ", content)

            # Parse the JSON response into the EvaluationSchema
            evaluation_data = EvaluationSchema.model_validate_json(content)
            #print ("Evaluation Data: ", evaluation_data)
            return evaluation_data
        except Exception as e:
            content = "" 
            #print ("Error during evaluation: ", e)
            raise ValueError(f"Failed to parse evaluation response: {e}\nResponse Content: {content}")

        
    # 3. Batch Processing Pipeline
def batch_evaluate_csv(input_csv_path: str, output_csv_path: str):
    """
        Reads a CSV file with 'context' and 'student_answer' columns, evaluates each row, and writes the results to a new CSV.
    """
    judge = RAGFaithfulnessJudge()

    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile: 
        reader = csv.DictReader(infile)
        required_headers = ['context', 'prompt', 'student_answer']
        for header in required_headers:
            if header not in reader.fieldnames:
                raise ValueError(f"Required column '{header}' is missing from the input CSV.")

        output_headers = reader.fieldnames + ['judge_rationale', 'judge_verdict', 'eval_status']
        
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_headers)
            writer.writeheader()
            row_count = 0
            for row in reader:
                row_count += 1
                print (f"Evaluating row {row_count}...")
                context = row.get('context', '')
                student_answer = row.get('student_answer', '')
                try:
                    evaluation = judge.evaluate(context, student_answer)
                    row['judge_rationale'] = evaluation.rationale
                    row['judge_verdict'] = evaluation.verdict
                    row['eval_status'] = "Success"
                except Exception as e:
                    row['judge_rationale'] = f"Error: {str(e)}"
                    row['judge_verdict'] = "Error"
                    row['eval_status'] = "Error"
                writer.writerow(row)
        print(f"Batch evaluation complete! Results saved to: {output_csv_path}")


# 4. Local Test Mock Setup
if __name__ == "__main__":
    # Mocking a typical CSV output generated by an LLM routing script
    input_file = "router_outputs.csv"
    output_file = "judge_evaluations.csv"
    load_dotenv()
    open_ai_api_key = os.getenv("OPENAI_API_KEY")
    print ("Got the key from .env file: ", open_ai_api_key)
    
    mock_data = [
        {
            "route_destination": "finance_agent",
            "prompt": "What is the cost of Project Orion?",
            "context": "Project Orion cost a total of $12.4 million.",
            "student_answer": "The project cost $12.4 million."
        },
        {
            "route_destination": "general_agent",
            "prompt": "When did Project Orion launch?",
            "context": "Project Orion launched on March 14, 2025.",
            "student_answer": "It launched in December of 2025."
        }
    ]
    
    # Write mock data to local directory for execution testing
    with open(input_file, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=mock_data[0].keys())
        writer.writeheader()
        writer.writerows(mock_data)
        
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"Created '{input_file}'. Please set your OPENAI_API_KEY to execute the evaluation.")
    else:
        # Run batch suite
        batch_evaluate_csv(input_file, output_file)


        
        