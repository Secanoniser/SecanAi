import os
import requests
from settings import get_settings

def scrape_wikipedia_articles():
    print("[*] Starting Automated Web Scraping for Training Corpus...")
    
    # Target educational topics on Wikipedia
    topics = [
        "Artificial intelligence",
        "Machine learning",
        "Transformer (deep learning architecture)",
        "Calculus",
        "Pythagorean theorem",
        "Python (programming language)"
    ]
    
    scraped_text = []
    
    for topic in topics:
        url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={topic}&format=json"
        headers = {"User-Agent": "LocalLLMScraper/1.0 (Educational Project)"}
        
        try:
            print(f"[*] Fetching article: '{topic}'...")
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_info in pages.items():
                    if "extract" in page_info:
                        extract = page_info["extract"]
                        scraped_text.append(f"--- Article: {topic} ---\n{extract}\n")
                        print(f"[+] Successfully scraped {len(extract)} characters for '{topic}'.")
            else:
                print(f"[!] Failed to fetch '{topic}' (Status: {response.status_code})")
        except Exception as e:
            print(f"[!] Error scraping '{topic}': {e}")
            
    if scraped_text:
        corpus_path = get_settings().raw_data_dir / "corpus.txt"
        combined_text = "\n\n".join(scraped_text)
        
        os.makedirs(corpus_path.parent, exist_ok=True)
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
            
        print(f"[+] Scraped corpus successfully saved to {corpus_path} ({len(combined_text)} characters)")
    else:
        print("[!] No text scraped. Check internet connection or API limits.")

if __name__ == "__main__":
    scrape_wikipedia_articles()
