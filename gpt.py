from openai import OpenAI
client = OpenAI(
    base_url="https://api.ai-gaochao.cn/v1/",
    api_key='sk-UYqwq36Z0hmfyaWJ69F675A344D645D79c9dB863Ae870eAd'
)

completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
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