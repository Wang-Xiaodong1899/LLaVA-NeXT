import os
from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY"),
)

completion = client.chat.completions.create(
                model="openai/gpt-3.5-turbo-0613",
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
            # Convert response to a Python dictionary.
response_message = completion.choices[0].message.content
print(response_message)
print(completion.model)