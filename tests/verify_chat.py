import requests
import json

def test_chat():
    url = "http://localhost:1234/api/v1/chat"
    headers = {
        "X-SERVICE-KEY": "Secret_key",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": "ما هي حقوق المستأجر في منطقة كسروان؟",
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "How can I help you regarding Lebanese Law?"}
        ],
        "user_context": {"language": "ar"}
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_chat()
