import requests
import json
from django.shortcuts import render
from django.http import HttpResponse
from .reddit_api import search_reddit_posts
from django.http import HttpResponse
from . generate_response import generate_reddit_gpt_response
from .reddit_sdk import RedditAssistantSDK  # ✅ Use the actual app/module name
from .reddit_agent import run_agentic_query

# from .reddit_assistant import get_gpt_answer

def home(request):
    #return HttpResponse("Reddit Assistant is Working")
    return render(request, 'home.html')

# Create your views here.
def search(request):
    query = request.GET.get('query')
    context = {}

    if query:
        results = search_reddit_posts(query)
        context = {
            'query': query,
            'recommendations': results.get('recommendations', []),
            "entity_comments": results['entity_comments', {}],
            'results': results.get('results', [])
        }

    return render(request, 'search.html', context)


# def reddit_gpt_view(request):
#     response_data = {}

#     if request.method == "GET" and "query" in request.GET:
#         query = request.GET.get("query")
#         response_data = generate_reddit_gpt_response(query)

#     return render(request, "reddit_assistant.html", {
#         "response_data": response_data
#     })


# from django.http import StreamingHttpResponse

# def reddit_gpt_stream_view(request):
#     LLM_STUDIO_URL = "http://10.125.141.244:1234/v1/chat/completions"
#     LLM_MODEL = "hermes-3-llama-3.1-8b"
#     query = request.GET.get("query")

#     def stream_response():
#         response = requests.post(
#             LLM_STUDIO_URL,
#             headers={"Content-Type": "application/json"},
#             data=json.dumps({
#                 "model": LLM_MODEL,
#                 "messages": [
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": query},
#                 ],
#                 "stream": True
#             }),
#             stream=True
#         )
#         for line in response.iter_lines():
#             if line:
#                 try:
#                     data = json.loads(line.decode("utf-8").lstrip("data: "))
#                     text = data['choices'][0]['delta'].get('content', '')
#                     yield text
#                 except Exception:
#                     continue

#     return StreamingHttpResponse(stream_response(), content_type='text/plain')

from django.http import StreamingHttpResponse

# views.py
from django.http import StreamingHttpResponse, JsonResponse
from .reddit_api import search_reddit_posts
from .reddit_assistant import stream_gpt_answer


def reddit_gpt_stream_view(request):
    query = request.GET.get("query")
    model = request.GET.get("model", "hermes")  # default is hermes

    if model == "gpt":
        def stream():
            yield ask_openai(query, model="gpt-4")
        return StreamingHttpResponse(stream(), content_type='text/plain')

    else:
        return StreamingHttpResponse(
            stream_gpt_answer(query),  # Hermes
            content_type='text/plain'
        )

    # return StreamingHttpResponse(
    #     stream_gpt_answer(query),  # ✅ not get_gpt_answer
    #     content_type='text/plain'
    # )

# def reddit_data_view(request):
#     query = request.GET.get("query")
#     reddit_data = search_reddit_posts(query)
#     return JsonResponse({
#         "recommended_entities": reddit_data.get("recommendations", []),
#         "reddit_posts": reddit_data.get("results", [])
#     })

def reddit_data_view(request):
    query = request.GET.get("query", "")
    model = request.GET.get("model", "hermes")  # default to hermes if not set

    # data = search_reddit_posts(query, model=model)

    # return JsonResponse({
    #     "reddit_posts": data["results"],
    #     "recommended_entities": data["recommended_entities"],
    # })

    sdk = RedditAssistantSDK(model=model)
    result = sdk.run_full_pipeline(query)
    formatted_output = sdk.format_output(result)
    print("🧠 Final SDK Output to frontend:", formatted_output)


    return JsonResponse(formatted_output)


    # return JsonResponse({
    #     "reddit_posts": data["results"],
    #     "recommended_entities": data["recommendations"],
    # })




from django.http import JsonResponse
from .openai_utils import ask_openai

def gpt_answer_view(request):
    user_query = request.GET.get("query", "")
    if not user_query:
        return JsonResponse({"error": "No query provided"}, status=400)
    
    answer = ask_openai(user_query, model="gpt-4")
    return JsonResponse({"answer": answer})


def reddit_agentic_view(request):
    query = request.GET.get("query")
    model = request.GET.get("model", "kimi-k2")

    if not query:
        return JsonResponse({"error": "No query provided"}, status=400)
    
    result = run_agentic_query(query, model=model)
    # return JsonResponse(result)
    return JsonResponse({
        "entities": result.get("recommended_entities", []),
        "top_posts": result.get("results", []),
        "debug": {
            "raw_result_keys": list(result.keys()),
            "first_entity": result.get("recommended_entities", [])[0] if result.get("recommended_entities") else {},
            "first_post": result.get("results", [])[0] if result.get("results") else {}
        }
    })
