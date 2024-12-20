import os
from openai import OpenAI
client = OpenAI(
    base_url="https://api.ai-gaochao.cn/v1/",
    api_key=os.getenv("API_KEY"),
)

completion = client.chat.completions.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                    },
                    {
                        "role": "user",
                        "content":
                            "hello!"
                    }
                ]
            )

# import requests
# import json

# key = os.getenv("API_KEY")

# response = requests.post(
#   url="https://openrouter.ai/api/v1/chat/completions",
#   headers={
#     "Authorization": f"Bearer {key}",
#   },
#   data=json.dumps({
#     "model": "openai/gpt-3.5-turbo-0613", # Optional
#     "messages": [
#       {
#         "role": "user",
#         "content": "What is the meaning of life?"
#       }
#     ]
    
#   })
# )

# print(response)
# if response.status_code == 200:
#     content = response.json()
#     print(content)
# else:
#     print(f"Error: {response.status_code}, {response.text}")

# Convert response to a Python dictionary.
response_message = completion.choices[0].message.content
print(response_message)
print(completion.model)