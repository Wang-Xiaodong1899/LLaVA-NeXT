import os
import openai
import requests

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.ai-gaochao.cn/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "sk-W8ed9MXndRIxpFZB3a527498E21849F2Ba77B2235597255a")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }

messages = [
        {
            "role": "system",
            "content": "You are a helpful and precise assistant for checking the quality of the answer.",
        },
        {"role": "user", "content": "hello"},
    ]

payload = {
    "model": "gpt-3.5-turbo",
    "messages": messages,
    "temperature": 0.2,
    "max_tokens": 256,
}

if API_TYPE == "azure":
    payload.pop("model")


response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
response.raise_for_status()
response_data = response.json()

content = response_data["choices"][0]["message"]["content"].strip()
if content != "":
    print(content, response_data["model"])