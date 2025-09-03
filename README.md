# SmartPickr – Agentic AI Reddit Assistant

SmartPickr is a hybrid Agentic AI web application that helps users quickly find the most recommended courses, tools, and local places from Reddit discussions. Instead of reading through multiple 
Reddit threads, SmartPickr extracts, counts, and ranks real-world recommendations so that users can make decisions faster.

Instead of reading through dozens of Reddit threads or Google reviews SmartPickr summarizes the community’s wisdom into **ranked, transparent recommendations** — powered by **LLMs + NLP + semantic search**.


---

## Features
- Fetches and reranks Reddit posts using semantic search
- Detects and filters by location if mentioned in the query
- Extracts named entities (courses, tools, clubs, restaurants, etc.) from top Reddit comments using LLMs(GPT 4.1, Kimi-K2, DeepSeek, Hermes)
- Groups duplicates and shows how many times each entity was mentioned
- Displays top Reddit comment snippets and links back to the source
- Runs heavy tasks (LLM extraction, NLP processing) in the background with Celery + Redis
- Deployed to AWS Lightsail with Gunicorn, Nginx, and systemd

---

## Tech Stack
**Backend**
- Python, Django
- Celery, Redis
- PRAW (Reddit API)
- spaCy, SentenceTransformers
- OpenAI GPT-4, Hermes (local LLM), DeepSeek (via OpenRouter), Kimi-K2

**Frontend**
- HTML, CSS, JavaScript (AJAX for dynamic updates)

**Deployment**
- AWS Lightsail
- Gunicorn + Nginx + systemd

---

## How It Works
1. User enters a query, e.g. “best Django course”.
2. Application fetches Reddit posts via the PRAW API.
3. Posts are filtered and reranked using spaCy (for language parsing) and SentenceTransformers (for semantic embeddings).
4. LLMs extract recommended entities from the most relevant comments.
5. Entities are grouped, counted, and displayed with their top supporting comments and links to the original Reddit posts.

---

## Example Use Cases
- “Best Django course” → shows courses like CS50 Web, Django for Beginners, Dennis Ivy tutorials
- “Adult dance classes in Frisco” → finds local dance studios recommended by Reddit users
- “Best pizza places in NYC” → shows restaurant names mentioned most in Reddit threads



