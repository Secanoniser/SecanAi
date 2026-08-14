import requests
import json

def test_local_api():
    url = "http://127.0.0.1:8000/api/chat"
    prompts = [
        "Hi",
        "What is Python?",
        "Solve 2x + 3 = 7"
    ]

    print("[*] Testing local AI server at http://127.0.0.1:8000...")
    
    for prompt in prompts:
        try:
            response = requests.post(url, json={"prompt": prompt}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(f"\nPrompt: '{prompt}'")
                print(f"AI Response: '{data.get('response')}'")
                print("-" * 50)
            else:
                print(f"[!] Server error for '{prompt}': Status {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[!] Could not connect to API for '{prompt}': {e}")

if __name__ == "__main__":
    test_local_api()
