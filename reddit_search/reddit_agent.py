import json
from openai import OpenAI
from .reddit_sdk import RedditAssistantSDK

client = OpenAI()

def run_agentic_query(user_input, model="kimi-k2"):
    """
    Hybrid version:
    ✅ Step 1: Skip GPT rewriting and use raw query
    ✅ Step 2: Use full pipeline that internally calls GPT to extract entities and attach 3 Reddit comments
    """
    print("📝 Original user input:", user_input)

    # Step 1: Use user input as-is
    query = user_input.strip()

    # Step 2: Run the original pipeline (search, rerank, extract entities + 3 comments per entity)
    sdk = RedditAssistantSDK(model=model)
    result = sdk.run_full_pipeline(query)

    return result

















# client = OpenAI()

# def run_agentic_query(user_input, model="kimi-k2"):
#     """
#     Hybrid version:
#     - Uses manual search and reranking for Reddit posts.
#     - Uses Agentic AI (GPT) only to extract entities from collected comments.
#     """
#     print("📝 Original user input:", user_input)

#     # Step 1: Manual Reddit post search + rerank
#     sdk = RedditAssistantSDK(model=model)
#     print("🔍 Searching and reranking Reddit posts...")
#     result = sdk.run_full_pipeline(user_input)

#     # Step 2: Extract all comments from result
#     all_comments = result.get("all_comments", [])
#     if not all_comments:
#         print("⚠️ No comments found for GPT to process.")
#         return result  # return whatever is there

#     # Step 3: Let Agentic AI extract top recommended entities from comments
#     formatted_comments = "\n\n".join([c["text"] for c in all_comments[:50]])  # limit to 50
#     prompt = (
#     "You are an assistant that analyzes Reddit comments and extracts the most frequently mentioned named entities "
#     "(e.g., tools, products, places, services) from them.\n\n"
#     "Please return the result as a JSON list of dictionaries with two keys:\n"
#     "1. 'name': the name of the entity\n"
#     "2. 'mentions': number of times it was mentioned\n\n"
   
#     f"Reddit Comments:\n{formatted_comments}"
#     )


#     print("🤖 Sending comments to Agentic AI for entity extraction...")
#     response = client.chat.completions.create(
#         model="gpt-4-1106-preview",  # or kimi-k2 if using OpenRouter
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant that extracts top recommendations from Reddit."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     extracted_entities_raw = response.choices[0].message.content.strip()
#     if extracted_entities_raw.startswith("```json"):
#         extracted_entities_raw = extracted_entities_raw.lstrip("```json").rstrip("```").strip()
#     elif extracted_entities_raw.startswith("```"):
#         extracted_entities_raw = extracted_entities_raw.lstrip("```").rstrip("```").strip()
#     print("✅ Agentic AI extracted entities:\n", extracted_entities_raw)

#     # Attach to result
#     # result["agentic_entities"] = extracted_entities
#     try:
#         entities = json.loads(extracted_entities_raw)
#         # Optional filtering
#         entities = [
#             e for e in entities
#             if isinstance(e, dict) and e.get("name") and e.get("mentions", 0) >= 2
#         ]
#     except json.JSONDecodeError:
#         print("❌ Failed to parse Agentic response")
#         entities = []

#     result["recommended_entities"] = entities

#     return result

# client = OpenAI()


# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "run_reddit_pipeline",
#             "description": "Search reddit and extract top recommended entities for a given topic",
#             "parameters":{
#                 "type": "object",
#                 "properties":{
#                     "query":{
#                         "type": "string",
#                         "description": "The Search topic or question (e.g. 'best djnago course')"
#                     }
#                 },
#                 "required":["query"]
#             }
#         }

#     }
# ]



# def run_agentic_query(user_input, model="kimi-k2"):
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant that can search Reddit and recommend the most mentioned tools or courses, places, attorneys etc."},
#         {"role": "user", "content": user_input}
#     ]

#     response = client.chat.completions.create(
#         model = "gpt-4-1106-preview",
#         messages= messages,
#         tools = tools,
#         # tool_choice = "auto"
#         tool_choice={"type": "function", "function": {"name": "run_reddit_pipeline"}}


#     )

#     tool_call= response.choices[0].message.tool_calls
#     if tool_call:
#         tool_args = json.loads(tool_call[0].function.arguments)
#         query = tool_args["query"]

#         print("🧠 Agentic AI interpreted query:", query)
#         print("📝 Original user input:", user_input)


#         sdk = RedditAssistantSDK(model=model)
#         result = sdk.run_full_pipeline(query)
#         return result
#         # final_output= sdk.format_output(result)

#         # # return final_output
#         # return {
#         #     "recommended_entities": final_output.get("entities", []),
#         #     "results": final_output.get("top_posts", [])
#         # }

#     else:
#         return {"answer": response.choices[0].message.content}