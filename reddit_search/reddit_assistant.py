import requests
import json
import re


# LLM_STUDIO_URL = "http://10.125.141.244:1234/v1/chat/completions"
LLM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

LLM_MODEL = "hermes-3-llama-3.1-8b"

def stream_gpt_answer(user_query):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", 
             "content": (
                    "You are a helpful assistant like ChatGPT. Answer user questions "
                    "in clear, conversational English. Provide explanations, lists, and examples where useful. "
                    "Do NOT return JSON or code unless specifically asked."
                )
            },
            {"role": "user", "content": user_query},
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": True
    }

    try:
        response = requests.post(
            LLM_STUDIO_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            stream=True
        )

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8").lstrip("data: "))
                    content = data['choices'][0]['delta'].get('content', '')
                    yield content
                except Exception:
                    continue
    except Exception as e:
        yield "Sorry, I couldn't generate a response at the moment.\n"











# def get_gpt_answer(user_query):
#     payload = {
#         "model": LLM_MODEL,
#         "messages":[
#             {"role":"system", "content":(
#                 "You are a helpful assistant who answers user queries like ChatGPT would. "
#                 "Give a direct and informative answer in natural language based on the question. "
#                 "Do NOT return JSON unless explicitly asked."
#             )},
#             {"role": "user", "content": user_query},
#         ],
#         "temperature": 0.7,
#         "max_tokens": 500
#     }

#     try:
#         response = requests.post(
#             LLM_STUDIO_URL,
#             headers = {"Content-Type": "application/json"},
#             data = json.dumps(payload)
#         )

#         result = response.json()
#         text = result['choices'][0]['message']['content'].strip()

#         # Try to extract JSON if present
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         recommendations = []
#         if match:
#             try:
#                 json_obj = json.loads(match.group())
#                 recommendations = sorted(json_obj.items(), key=lambda x: x[1], reverse=True)
#                 # Remove JSON from the answer so GPT part is clean
#                 text = text.replace(match.group(), "").strip()
#             except:
#                 pass

#         return {
#             "gpt_answer": text,
#             "recommendations": recommendations
#         }

#     except Exception as e:
#         print("LLM error:", e)
#         return {
#             "gpt_answer": "Sorry, I couldn't generate a response at the moment.",
#             "recommendations": []
#         }
# import requests
# import json
# import re

# LLM_STUDIO_URL = "http://10.125.141.244:1234/v1/chat/completions"
# LLM_MODEL = "hermes-3-llama-3.1-8b"

# def get_gpt_answer(user_query):
#     payload = {
#         "model": LLM_MODEL,
#         "messages":[
#             {"role":"system", "content":(
#                     "You are a helpful assistant who answers user queries like ChatGPT would. "
#                     "Give a direct and informative answer in natural language based on the question. "
#                     "Do NOT return JSON unless explicitly asked."
#                     )
#                     },
#             {"role": "user", "content": user_query},
#         ],
#         "temperature": 0.7,
#         "max_tokens": 500

#     }

#     try:
#         response = requests.post(
#             LLM_STUDIO_URL,
#             headers = {"Content-Type": "application/json"},
#             data = json.dumps(payload)
#         )

#         result = response.json()
#         text = result['choices'][0]['message']['content'].strip()
    
#     except Exception as e:
#         print("LLM error:", e)
#         return "Sorry I couldnt generate a response at the moment"
    
#     # match = re.search(r"\{.*\}", text, re.DOTALL)  #re.DOTALL tells Python to include newlines when matching .*

#     # if match:
#     #     json_obj = json.loads(match.group())
#     #     sorted_items = sorted(json_obj.items(), key=lambda x:x[1], reverse=True)
#     #     return sorted_items

