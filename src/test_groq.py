from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Reply with exactly one word: pong"}],
)

print(response.choices[0].message.content)
