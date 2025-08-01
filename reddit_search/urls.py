from django.urls import path
from . import views
# from .views import reddit_gpt_view

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search, name='search'),
    # path("reddit-assistant/", reddit_gpt_view, name="reddit_gpt"),
    # path("reddit-assistant-stream-all/", views.reddit_gpt_stream_all_view, name="reddit_gpt_stream_all"),

    path("reddit-assistant-stream/", views.reddit_gpt_stream_view, name="reddit_gpt_stream"),
    path("reddit-assistant-data/", views.reddit_data_view, name="reddit_data"),
    path("gpt-answer/", views.gpt_answer_view, name="gpt-answer"),
    path("reddit-agentic-data/", views.reddit_agentic_view, name="reddit_agentic_data"),
    path("get-task-result/", views.get_task_result, name="get_task_result"),

    



]


