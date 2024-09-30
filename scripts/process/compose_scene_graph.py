from openai import OpenAI

instances = ['neck0', 'hair0', 'hair1', 'shirt0', 'hand0', 'nose0', 'ear0', 'mouth0', 'hand1', 'arm0'] 
relations = [
    ['hair1', 'on', 'neck0'],
    ['hand1', 'on', 'shirt0'],
    ['hand1', 'holding', 'hand0']
]

context = f"""
Given an image, there are instances and some relations.
Instances: {instances}
relations: {relations}
Please use this information to compose a description of the image, including the main instances and the main relations. 
Please describe it in natural language. Do not use the names of the instances.
"""

client = OpenAI(
    base_url="https://api.ai-gaochao.cn/v1/",
    api_key='sk-UYqwq36Z0hmfyaWJ69F675A344D645D79c9dB863Ae870eAd'
)

completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot. "
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ]
            )
            # Convert response to a Python dictionary.
response_message = completion.choices[0].message.content
print(context)
print(response_message)
print(completion.model)