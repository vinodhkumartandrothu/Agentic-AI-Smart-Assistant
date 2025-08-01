# from . reddit_api import search_reddit_posts
# from . reddit_assistant import get_gpt_answer

# def generate_reddit_gpt_response(user_query):

#     gpt_answer = get_gpt_answer(user_query)

#     reddit_data = search_reddit_posts(user_query)


#     return {
#         "gpt_answer":gpt_answer,
#         "recommended_entities":reddit_data['recommendations'],
#         "reddit_posts":reddit_data['results']
#     }

from .reddit_api import search_reddit_posts
from .reddit_assistant import stream_gpt_answer

def generate_reddit_gpt_response(user_query):
    gpt_data = stream_gpt_answer(user_query)
    reddit_data = search_reddit_posts(user_query)

    return {
        "gpt_answer": gpt_data,
        "recommended_entities": reddit_data['recommendations'],
        "reddit_posts": reddit_data['results']
    }
