from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path


load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

#print("Loaded OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

api_key = os.getenv("OPENAI_API_KEY")
print("✅ DEBUG: OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))


if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check .env file for environment")



client = OpenAI(api_key=api_key)


def ask_openai(prompt, model="gpt-4o"):  # Use gpt-4o or gpt-3.5-turbo if you don’t have gpt-4 access
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OPENAI Error: {e}"
    



# models = client.models.list()
# for model in models.data:
#     print(model.id)

# def ask_openai(prompt, model="gpt-3.5-turbo"):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"OpenAI Error: {e}"