import json
from openai import OpenAI

client = OpenAI(
    base_url="https://api.ai-gaochao.cn/v1/",
    api_key='sk-UYqwq36Z0hmfyaWJ69F675A344D645D79c9dB863Ae870eAd'
)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def generate(q=None, a=None):

    q = "How does the background fabric enhance the decorative appeal of the mirror cover?" if q is None else q
    a = "The background fabric on which the mirror cover is displayed is golden with a shiny, patterned texture, which enhances the overall decorative appeal of the cover." if a is None else a

    content = f"""
    Give you a question and a corresponding answer. Please ask 3 questions that have the same meaning as the original question, but the style and wording are quite different. 

    Question: {q}

    Answer: {a}

    Response:
    Q1: 
    Q2: 
    Q3: 

    """

    # print(content)

    completion = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot. "
                        },
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                )
                # Convert response to a Python dictionary.
    response_message = completion.choices[0].message.content
    # print(response_message)
    qs = response_message.split(":")[-3:]
    qs = [x.split("\n")[0].strip() for x in qs]
    print(qs)
    return response_message
    # print(completion.model)

data = load_jsonl()

generate()