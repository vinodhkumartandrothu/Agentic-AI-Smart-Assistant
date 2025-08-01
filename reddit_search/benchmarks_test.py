import os
import sys
import django

# ✅ Step 1: Add root directory to Python path
# Assumes your folder structure is like:
#   reddit_assistant_app/
#     ├── core/
#     ├── reddit_search/
#     └── manage.py

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# ✅ Step 2: Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

# ✅ Step 3: Now import your module
from reddit_search.reddit_api import search_reddit_posts


queries = [
    "Best Python course for beginners",
    "Top free AI tools",
    "Best Samsung phone under $1000",
    "Good laptops under $1000",
    "Best nighclubs in dallas",
    "best attorney for h1-b to b2",
    "best mexican restaraunts in dallas"
]

for model in [ "kimi-k2"]:
    print(f"\n--- Benchmarking Model: {model.upper()} ---")
    for q in queries:
        print(f"\n🧠 Query: {q}")
        data = search_reddit_posts(q, model=model)
        print(f"🔗 Reddit Posts: {len(data['results'])} | 🏷️ Entities: {len(data['recommendations'])}")
        print("Top Entities:", data['recommendations'][:5])

