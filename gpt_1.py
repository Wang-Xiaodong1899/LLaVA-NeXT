import openai

# openai.base_url = "https://api.ai-gaochao.cn/v1/"
# openai.api_key = "sk-UYqwq36Z0hmfyaWJ69F675A344D645D79c9dB863Ae870eAd"

# openai.base_url = "https://api.chatanywhere.tech/v1/"
# openai.api_key = "sk-88uyXsAdEyDN5ESbWVWTtG6Do6vj9y2biMqtMsIsf6pqDvvY"

openai.base_url = "https://openrouter.ai/api/v1"
openai.api_key = "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"

# HTTPS_PROXY=http://fvgroup:48423590@10.54.0.93:3128 

completion = openai.chat.completions.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {
        "role": "user",
        "content": "how are you, what's your name?"
        },
        ],
)

print(completion.choices[0].message.content)
print(completion.model)

# ok
# gpt-3.5-turbo-1106
# gpt-3.5-turbo -> gpt-3.5-turbo-0125


# all
# "gpt-3.5-turbo",
# "gpt-3.5-turbo-1106",
# "gpt-3.5-turbo-0301",
# "gpt-3.5-turbo-0613",
# "gpt-3.5-turbo-1106",
# "gpt-3.5-turbo-0125",
# "gpt-3.5-turbo-1106-0613",

