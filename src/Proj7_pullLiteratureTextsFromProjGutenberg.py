import os
import requests

os.makedirs("data/literature_docs", exist_ok=True)

# Dictionary of {Book Title: Project Gutenberg ID}
books = {
    "pride_and_prejudice": 1342,
    "frankenstein": 84,
    "alice_in_wonderland": 11,
    "moby_dick": 2701,
    "dracula": 345,
    "sherlockholmes": 1661,
    "taleoftwocities": 98,
    "shakespeare": 100,
}
#https://www.gutenberg.org/cache/epub/84/pg84.txt

for title, book_id in books.items():
    url = f"https://gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    
    if response.status_code == 200:
        text = response.text
        # Optional: Strip Project Gutenberg header and footer boilerplate
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        
        if start_marker in text and end_marker in text:
            text = text.split(start_marker, 1)[1].split("\n", 1)[1]
            text = text.split(end_marker, 1)[0]
       
            
        filepath = os.path.join("data/literature_docs", f"{title}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"Downloaded: {title}.txt")
    else:
        print(f"Failed to download {title} (ID: {book_id})")
