from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = KEY)
#client = OpenAI()

def call_openai_model(prompt, modell="gpt-4o"):

    messages = [
        #{"role": "assistant", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt}
        ]

    response = client.chat.completions.create(
        model = modell,
        messages = messages
        )
    return response.choices[0].message.content