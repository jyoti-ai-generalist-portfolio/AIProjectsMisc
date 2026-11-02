from google import genai
from groq import Groq
import os
import tiktoken
from tiktoken import Encoding
from google.genai import types
from dotenv import load_dotenv
import datetime 
import time

import csv


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
#print ("GEMINI_API_KEY:", gemini_api_key)
#print ("GROQ_API_KEY:", groq_api_key)
gemini_client = genai.Client(api_key=gemini_api_key)
groq_client = Groq(api_key=groq_api_key)
#groq_client = OpenAI(
 #   base_url="https://api.groq.com/openai/v1",
  #  api_key=groq_api_key
#)

LOG_FILE = "llm_router_logs.csv"


def init_log_file():
    """Initializes the CSV log file with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Prompt_Preview", "Structured_Requested", "Token_Count", "Chosen_Provider", "Latency_Sec", "Response_Preview"])
            

def log_metrics(prompt, is_structured, token_count, provider, latency, response):
    """Appends execution metrics to the CSV file."""
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.fromtimestamp(time.time()).isoformat(),
            prompt[:50] + "...", 
            is_structured,
            token_count,
            provider,
            round(latency, 4),
            response[:50].replace("\n", " ") + "..."
        ])

def estimate_tokens(text: str) -> int:
    """Uses tiktoken to quickly estimate the token count of the prompt."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough word count if tiktoken fails
        return len(text.split())

"""Tiktoken is OpenAI's open-source, high-performance Byte Pair Encoding (BPE) tokenizer.
 It breaks text strings down into smaller units called tokens (which can be parts of words, whole words, 
 characters, or punctuation) and converts them into integer IDs that Large Language Models (LLMs) can process.

Key Features

• Speed: Written largely in Rust, tiktoken is significantly faster (around 3 to 6 times faster) than many traditional Python-based tokenizers.
• Model Accuracy: It uses the exact same tokenization rules and vocabularies as OpenAI's official models, letting developers match API billing counts and manage context windows locally.
• Reversibility: It offers lossless encoding and decoding, meaning you can convert text to tokens (.encode()) and turn those tokens right back into the original text (.decode()).

Common Encodings

Different OpenAI models use different underlying encodings:
• o200k_base: Used by newer models like gpt-4o and gpt-4o-mini.
• cl100k_base: Used by gpt-4, gpt-4-turbo, gpt-3.5-turbo, and text-embedding models.
• p50k_base: Used by older Codex models and davinci-002/davinci-003.
• r50k_base (or gpt2): Used by legacy GPT-3 models."""

def route_request(prompt: str, require_structured: bool = False):
    """Routes the request to the appropriate LLM provider based on the prompt and requirements."""
    init_log_file()  # Ensure log file is initialized

    token_count = estimate_tokens(prompt)
    start_time = time.time()
    response_text = ""
   
    
    # Routing decision logic
    if require_structured or token_count > 200:
        provider = "Gemini"
    else:
        provider = "Groq"
        
    print(f"🤖 Routing to {provider} (Tokens: {token_count}, Structured: {require_structured})...")
    
   
   

    try :
        if provider == "Groq":
            # Using Llama 3.3 70B on Groq for blazing fast simple tasks
            completion = groq_client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
             # FIX: Native Groq client syntax uses .choices.message.content (no index)
            response_text = completion.choices[0].message.content
            
        elif provider == "Gemini":
            # Using Gemini 3.5 Flash for complex/structured tasks
            if require_structured:
                # Force JSON schema output using the new Google GenAI SDK
                completion = gemini_client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
            else:
                completion = gemini_client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt
                )
            response_text = completion.text
        latency = time.time() - start_time
    
            # Log the results
        log_metrics(prompt, require_structured, token_count, provider, latency, response_text)
    
        return response_text, provider, latency

    except Exception as e:
        response_text = f"Error: {str(e)}"
        provider = "Error"
        return response_text, provider, 0.0 


# --- Test Suite Execution ---
if __name__ == "__main__":
    init_log_file()
    
    print("--- Test 1: Simple/Short Prompt ---")
    p1 = "What is the capital of France? Reply in one word."
    res1, prov1, lat1 = route_request(p1)
    print(f"Result: {res1.strip()} | Latency: {lat1:.2f}s\n")
    
    print("--- Test 2: Structured Request ---")
    p2 = "List 3 primary colors and their hex codes."
    res2, prov2, lat2 = route_request(p2, require_structured=True)
    print(f"Result: {res2.strip()} | Latency: {lat2:.2f}s\n")
    
    print("--- Test 3: Long Prompt ---")
    p3 = "Write a comprehensive, detailed essay explaining the history of the internet from the 1960s ARPANET to modern fiber optics. " * 15
    res3, prov3, lat3 = route_request(p3)
    print(f"Result Preview: {res3[:100]}... | Latency: {lat3:.2f}s\n")
    
    print(f"📈 All metrics successfully logged to {LOG_FILE}!")

    


