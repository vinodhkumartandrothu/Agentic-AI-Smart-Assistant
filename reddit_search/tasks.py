# reddit_search/tasks.py
from celery import shared_task
from .reddit_agent import run_agentic_query

@shared_task
def run_agentic_query_task(query, model="gpt"):
    return run_agentic_query(query, model)
