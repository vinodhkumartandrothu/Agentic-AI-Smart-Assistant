import praw
import requests
import json
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from openai import OpenAI
from .openai_utils import client
# from . import reddit_config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz


# 🔁 Optional caching for NLP and embedding model
_embedding_model = None
_nlp = None

def get_embedding_model():
    """Loads the sentence-transformers model only once."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _embedding_model

def get_nlp_model():
    """Loads the spaCy model only once."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def normalize_entity_name(name: str) -> str:
    """Lowerase and remove trivial characters for grouping"""
    return re.sub(r"[^a-zA-Z0-9]", "", name.lower().strip())

def generate_entity_variants(entity_name: str):
    """
    Dynamically generate word combinations from entity name to improve matching.
    Example: 'Django for Beginners by Vincent' → ['django', 'for', 'beginners', 'vincent', 'django for', 'for beginners']
    """
    base = re.sub(r"[^a-zA-Z0-9 ]", "", entity_name.lower())
    words = base.split()
    variants = set()

    for i in range(len(words)):
        variants.add(words[i])
        if i < len(words) - 1:
            variants.add(f"{words[i]} {words[i+1]}")

    variants.add(base)  # Full normalized entity name
    return variants



# Prevent tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# # PRAW Reddit client
# reddit = praw.Reddit(
#     client_id=reddit_config.REDDIT_CLIENT_ID,
#     client_secret=reddit_config.REDDIT_CLIENT_SECRET,
#     user_agent=reddit_config.REDDIT_USER_AGENT
# )




# Try to load from environment
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

# If not found, fallback to reddit_config.py (local only)
if not client_id or not client_secret or not user_agent:
    try:
        from . import reddit_config
        client_id = reddit_config.REDDIT_CLIENT_ID
        client_secret = reddit_config.REDDIT_CLIENT_SECRET
        user_agent = reddit_config.REDDIT_USER_AGENT
    except ImportError:
        raise EnvironmentError("Missing Reddit credentials in both env and reddit_config.py")

# Finally, initialize
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)




# PRAW Reddit client
# reddit = praw.Reddit(
#     client_id=os.environ["REDDIT_CLIENT_ID"],
#     client_secret=os.environ["REDDIT_CLIENT_SECRET"],
#     user_agent=os.environ["REDDIT_USER_AGENT"]
# )


# Embedding model for reranking
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Hermes / local LLM endpoint
LLM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL = "hermes-3-llama-3.1-8b"



def fetch_reddit_posts(query: str, limit=200):
    """Search Reddit for posts matching the query.
       Pull Reddit’s “relevance” search
       Drop any post whose title doesn’t share at least one key noun/proper-noun from the query
    """
    subreddit = reddit.subreddit("all")
    posts = []
    for submission in subreddit.search(query, sort="relevance", time_filter="all", limit=limit):
        if submission.num_comments >= 3 and not submission.stickied:
            posts.append({
                'id': submission.id,
                'title': submission.title,
                'selftext': submission.selftext or "",
                'url': submission.url,
                'score': submission.score,
                'comments_count': submission.num_comments
            })
    return posts




def extract_phrases(query: str):
    """Extract 2- and 3-word phrases (no hardcoding, just chunking)."""
    words = query.lower().split()
    phrases = set()
    for size in [2, 3]:
        for i in range(len(words) - size + 1):
            phrases.add(" ".join(words[i:i+size]))
    return phrases

def filter_relevant_posts(posts: list, query: str):
    """Filter posts based on noun overlap and meaningful phrase matches."""
    nlp = get_nlp_model()
    doc = nlp(query)
    query_nouns = {tok.lemma_.lower() for tok in doc if tok.pos_ in ("NOUN", "PROPN")}
    query_phrases = extract_phrases(query)  # ['plant shops', 'in dfw', ...]

    filtered = []
    for post in posts:
        title = post["title"].lower()
        title_doc = nlp(title)
        title_nouns = {tok.lemma_.lower() for tok in title_doc if tok.pos_ in ("NOUN", "PROPN")}

        # ✅ Must match at least one key noun AND at least one key phrase
        has_noun_overlap = bool(query_nouns & title_nouns)
        has_phrase_match = any(phrase in title for phrase in query_phrases)

        if has_noun_overlap or has_phrase_match:
            filtered.append(post)

    return filtered



def rerank_posts(posts: list, query: str, top_k: int = 10):
    model = get_embedding_model()
    print(f"\n🔍 Query: {query}")
    query_emb = model.encode([query])[0]

    phrases = list(extract_phrases(query))
    print(f"🧩 Extracted Phrases: {phrases}")
    phrase_embs = model.encode(phrases) if phrases else []

    scored = []

    for idx, post in enumerate(posts):
        title = post['title']
        content = f"{title} {title} ||| {post['selftext'][:300]}"
        post_emb = model.encode([content])[0]

        # Similarity to full query
        sim_query = cosine_similarity([query_emb], [post_emb])[0][0]

        # Phrase-level similarities
        phrase_sims = []
        for pe, phrase in zip(phrase_embs, phrases):
            sim = cosine_similarity([pe], [post_emb])[0][0]
            phrase_sims.append((phrase, sim))

        if phrase_sims:
            best_phrase, best_sim = max(phrase_sims, key=lambda x: x[1])
        else:
            best_phrase, best_sim = ("", 0)

        # Final combined score
        final_score = 0.6 * sim_query + 0.4 * best_sim

        # print(f"\n📌 Post {idx+1}: {title}")
        # print(f"🔹 Query Similarity: {sim_query:.3f}")
        # print(f"🔹 Best Phrase: \"{best_phrase}\" (Score: {best_sim:.3f})")
        # print(f"⚖️ Final Weighted Score: {final_score:.3f}")

        scored.append((final_score, post))

    # scored.sort(key=lambda x: (x[0], x[1]['score']), reverse=True)
    # top_posts = [p for _, p in scored[:top_k]]

    # return top_posts
    # ⛔️ REMOVE: old scored.sort and top_k return

    # ✅ Add this instead
    MIN_SCORE_THRESHOLD = 0.35  # 🔧 Adjust based on your testing

    # Filter out posts below threshold
    scored = [(score, post) for score, post in scored if score >= MIN_SCORE_THRESHOLD]

    # Sort by score and return top_k
    scored.sort(key=lambda x: (x[0], x[1]['score']), reverse=True)
    top_posts = [p for _, p in scored[:top_k]]

    return top_posts








def fetch_top_comments(post_id: str, limit=10):
    """Pull the top `limit` upvoted comments from a submission."""
    submission = reddit.submission(id=post_id)
    submission.comment_sort = 'top'
    submission.comments.replace_more(limit=0)
    # return [c.body.strip() for c in submission.comments[:limit] if hasattr(c, 'body')]
    return [
    {
        "text": c.body.strip(),
        "url": f"https://www.reddit.com{c.permalink}",
        "post_id": submission.id,
        "score": c.score
    }
    for c in submission.comments[:limit]
    if hasattr(c, 'body')
    ]



def group_and_count_entities(entities: list) -> list:
    grouped = defaultdict(lambda: {"mentions": 0, "all_names": set()})

    for ent in entities:
        norm = normalize_entity_name(ent["name"])
        grouped[norm]["mentions"] += ent.get("mentions", 1)
        grouped[norm]["all_names"].add(ent["name"])

    final_output = []
    for norm_name, info in grouped.items():
        # Pick the most common display name
        display_name = max(info["all_names"], key=lambda x: list(info["all_names"]).count(x))
        
        # Generate rich variants from all names
        variants = set()
        for name in info["all_names"]:
            variants.update(generate_entity_variants(name))

        final_output.append({
            "name": display_name,
            "mentions": info["mentions"],
            "variants": list(variants),
            "norm_name": norm_name
        })

    return sorted(final_output, key=lambda x: x["mentions"], reverse=True)


def match_comments_to_entities(entities, comment_origin_map, post_url_map, max_comments=3):
    """
    Map each entity to the most relevant comments based on strong full-phrase matching.
    """
    from collections import defaultdict
    import re

    entity_comment_links = {}

    for entity in entities:
        norm_name = entity["norm_name"]
        mention_limit = min(entity["mentions"], max_comments)

        # Sort variants by length (longest first)
        sorted_variants = sorted(
            generate_entity_variants(entity["name"]),
            key=lambda x: -len(x)
        )

        matched_comments = []
        seen_texts = set()

        for comment_obj, post_id in comment_origin_map:
            text = comment_obj["text"].strip()
            text_lower = text.lower()

            # Match only if strong variant (2+ words or full name) appears
            matched = False
            for term in sorted_variants:
                if len(term.split()) >= 2:  # only use 2+ word phrases
                    pattern = r'\b' + re.escape(term.lower()) + r'\b'
                    if re.search(pattern, text_lower):
                        if text not in seen_texts:
                            seen_texts.add(text)
                            matched_comments.append({
                                "text": text,
                                "url": post_url_map[post_id],
                                "score": comment_obj.get("score", 0)
                            })
                        matched = True
                        break  # stop after first strong match

            # If no strong match, skip this comment entirely
            if matched:
                continue

        # Sort and limit
        top_comments = sorted(matched_comments, key=lambda x: x["score"], reverse=True)[:mention_limit]
        entity_comment_links[norm_name] = top_comments

    return entity_comment_links






def extract_recommendations(comments: list, model="hermes"):
    
    combined = "\n\n".join(comments)

    if model.lower().startswith("gpt"):
    
        prompt = f"""
            You are a smart assistant whose job is to extract the **most relevant real-world entities** from Reddit comments, based on the user's query and intent.

            Entities should be:
            - Specific and actionable (e.g., product names, books, companies, tools, law firms, individuals, courses)
            - Directly useful to someone asking the query
            - Mentioned positively or recommended by Reddit users

            Do not include generic topics or abstract concepts unless clearly recommended as a solution.

            1. Group together name variants (e.g. “Django for Beginners” and “Django for Beginners by William Vincent”).
            2. Count how many times each entity appears.
            3. Return *only the top 15* entities, sorted by mention count descending.
            4. Output exactly this JSON schema (no extra text):

            {{
            "entities": [
                {{ "name": "Entity A", "mentions": 12 }},
                {{ "name": "Entity B", "mentions": 8 }}
            ]
            }}

            Reddit comments:
            {combined}
            """


        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a JSON extractor for named entities."},
                    {"role": "user",   "content": prompt}
                ],
                functions=[{
                    "name": "extract_entities",
                    "description": "Extract and count each named entity and its occurrences.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "mentions": {"type": "integer"}
                                    },
                                    "required": ["name", "mentions"]
                                }
                            }
                        },
                        "required": ["entities"]
                    }
                }],
                function_call={"name": "extract_entities"},
                temperature=0.0,
            )

            choice = response.choices[0].message
            if getattr(choice, 'function_call', None):
                raw = choice.function_call.arguments
                parsed = json.loads(raw)
                entities = parsed.get("entities", [])
            else:
                print("⚠️ GPT did not return function_call. Falling back to content.")
                try:
                    parsed = json.loads(choice.content)
                    entities = parsed.get("entities", [])
                except Exception as e:
                    print("⚠️ Failed to parse GPT content:", e)
                    return []

            # if getattr(choice, 'function_call', None):
            #     raw = choice.function_call.arguments
            # else:
            #     raw = choice.content

            parsed = json.loads(raw)
            entities = parsed.get("entities", [])
            print("Raw GPT entities returned:", entities)
            entities = sorted(entities, key=lambda e: e["mentions"], reverse=True)[:15]

            # Optional spaCy filter (re-enable once you confirm GPT output)
            # filtered = [e for e in entities if is_meaningful(e['name'])]
            # return sorted(filtered, key=lambda x: x['mentions'], reverse=True)

            return sorted(entities, key=lambda x: x['mentions'], reverse=True)

        except Exception as e:
            print("OpenAI error in extract_recommendations:", e)
            return []
    

    elif model == "hermes":
    # Fallback to Hermes/local model
        payload = {
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant. Extract and rank the most "
                                "recommended named entities (courses, books, tools, clubs, etc.) "
                                "from the Reddit comments. Return a JSON mapping each entity "
                                "to its mention count."
                            )
                        },
                        {"role": "user", "content": combined}
                    ],
                    "temperature": 0.0
                }
        try:
            r = requests.post(LLM_STUDIO_URL, headers={"Content-Type": "application/json"},
                            data=json.dumps(payload))
            res = r.json()
            text = res['choices'][0]['message']['content']
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                print("Raw Hermes entities returned:", obj)  # ✅ ADD THIS

                return sorted(obj.items(), key=lambda x: x[1], reverse=True)
        except Exception as e:
            print("Hermes error in extract_recommendations:", e)
            return []
    
    else:
        # DeepSeek or other OpenRouter models
        MODEL_MAP = {
            "deepseek-r1": "deepseek/deepseek-r1-0528:free",
            "kimi-k2": "moonshotai/kimi-k2:free",
            
        }

        model_id = MODEL_MAP.get(model)
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }

        # Reuse the same prompt from earlier GPT block
        # prompt = f"""
        # You are a smart assistant whose only job is to read the Reddit comments below and pull out the handful of *most‑frequently mentioned* specific entities (course titles, book names, tool names, company names, locations, people, etc.).

        # 1. Group together any variants (e.g. “Django for Beginners” and “Django for Beginners by William Vincent”).
        # 2. Count how many times each entity appears.
        # 3. Return *only the top 15* entities, sorted by mention count descending.
        # 4. Output exactly this JSON schema (no extra text):

        # {{
        #     "entities": [
        #         {{ "name": "Entity A", "mentions": 12 }},
        #         {{ "name": "Entity B", "mentions": 8 }}
        #     ]
        # }}

        # Reddit comments:
        # {combined}
        # """
        prompt = f"""
                You are a smart assistant whose job is to extract the **most relevant real-world entities** from Reddit comments, based on the user's query and intent.

                Entities should be:
                - Specific and actionable (e.g., product names, books, companies, tools, law firms, individuals, courses)
                - Directly useful to someone asking the query
                - Mentioned positively or recommended by Reddit users

                Do not include generic topics or abstract concepts unless clearly recommended as a solution.

                1. Group together name variants (e.g. “Django for Beginners” and “Django for Beginners by William Vincent”).
                2. Count how many times each entity appears.
                 3. Return *only the top 15* entities, sorted by mention count descending.
                4. Output exactly this JSON schema (no extra text):

                {{
                "entities": [
                    {{ "name": "Entity A", "mentions": 12 }},
                    {{ "name": "Entity B", "mentions": 8 }}
                ]
                }}

                Reddit comments:
                {combined}
                """



        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a JSON extractor for named entities."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                return sorted(obj["entities"], key=lambda x: x["mentions"], reverse=True)[:15]
        except Exception as e:
            print(f"{model} error in extract_recommendations:", e)
            return []









def map_entities_to_comments(entities, comments):
    """
    Map each entity to comments that mention it.
    Number of comments must not exceed its mention count.
    Matching is done using dynamic word/phrase variants.
    """
    entity_comment_map = defaultdict(list)

    for entity in entities:
        display_name = entity["name"]
        mention_limit = entity["mentions"]
        variants = generate_entity_variants(display_name)

        matched = []

        for comment in comments:
            comment_text = comment.lower()

            # Match only if any full variant is present in the comment
            if any(variant in comment_text for variant in variants):
                matched.append(comment.strip())

        # Respect mention count: only up to that many comments
        entity_comment_map[display_name] = matched[:mention_limit]

    return entity_comment_map



def search_reddit_posts(query: str, model: str = "hermes"):
    posts = fetch_reddit_posts(query)
    filtered_posts = filter_relevant_posts(posts, query)
    top_posts = rerank_posts(posts, query, top_k=10)

    

    # all_comments = []
    entity_comment_sources = []
    comment_origin_map = []  # stores (comment_dict, post_id)

    query_keywords = query.lower().split()

    for post in top_posts:
        title = post["title"].lower()
        if any(word in title for word in query_keywords):
            comments = fetch_top_comments(post["id"], limit=15)
            entity_comment_sources.extend(comments)
            comment_origin_map.extend([(c, post["id"]) for c in comments])
        else:
            # Still collect comments for mapping later — but don't send them to GPT
            comments = fetch_top_comments(post["id"], limit=15)
            comment_origin_map.extend([(c, post["id"]) for c in comments])

    # for p in top_posts:
    #     comments = fetch_top_comments(p['id'], limit=15)
    #     all_comments.extend(comments)
    #     comment_origin_map.extend([(c, p['id']) for c in comments])

    # recommendations = extract_recommendations([c["text"] for c in all_comments], model=model)
    recommendations = extract_recommendations([c["text"] for c in entity_comment_sources], model=model)

    print(f"Model: {model} | Top Posts Found: {len(top_posts)}")
    print(f"Model: {model} | Entities extracted: {recommendations}")
    if not recommendations:
        print("⚠️ No recommendations after entity extraction step")
        return {
            "results": top_posts,
            "recommended_entities": []
        }
    
   

    recommendations = group_and_count_entities(recommendations)[:15]


    # NEW: Link entity to top 3 highest-voted comments mentioning it
    post_url_map = {p['id']: f"https://www.reddit.com/comments/{p['id']}" for p in top_posts}

    # print("\n🧪 DEBUG COMMENT MAPPING INPUTS")
    # print(f"Total extracted entities: {len(recommendations)}")
    # for e in recommendations:
    #     print(f" - ENTITY: {e['name']} (norm: {e['norm_name']}, variants: {e['variants']})")

    # print(f"\nTotal comments in comment_origin_map: {len(comment_origin_map)}")
    # for comment_obj, post_id in comment_origin_map[:5]:  # limit preview
    #     print(f" - COMMENT: {comment_obj.get('text', '')[:80]}... | post_id: {post_id}")

    
    
    entity_comment_links = match_comments_to_entities(
        recommendations,
        comment_origin_map,
        post_url_map,
        max_comments=3
    )



    # for e in recommendations:
    #     print(f"\n✅ ENTITY: {e['name']} | Mentions: {e['mentions']}")
    #     for c in entity_comment_links.get(e['norm_name'], []):
    #         print(f" - {c['text'][:80]}... [{c['score']}]")



    return {
            "results": top_posts,
            "recommended_entities": [
                {
                    "name": entity["name"],
                    "mentions": entity["mentions"],
                    # "comments": entity_comment_links.get(normalize_entity_name(entity["name"]), [])
                    "comments": entity_comment_links.get(entity["norm_name"], [])

                    # "comments": entity_comment_links.get(entity["name"].lower(), [])
                }
                for entity in recommendations
            ]
    }















# def group_and_count_entities(entities: list) -> list:
#     grouped = []

#     for ent in entities:
#         # Split entity name on delimiters to get meaningful parts
#         raw_parts = re.split(r"[\/,()]+", ent["name"])
#         clean_parts = [part.strip() for part in raw_parts if part.strip()]
#         total_mentions = ent.get("mentions", 1)
#         mention_share = total_mentions // max(1, len(clean_parts))

#         for part in clean_parts:
#             norm = normalize_entity_name(part)
#             matched = False

#             for group in grouped:
#                 similarity = fuzz.token_set_ratio(norm, group["norm_name"])
#                 if similarity >= 90:  # threshold without hardcoding any name
#                     group["mentions"] += mention_share
#                     group["variants"].append(part)
#                     matched = True
#                     break

#             if not matched:
#                 grouped.append({
#                     "norm_name": norm,
#                     "mentions": mention_share,
#                     "variants": [part]
#                 })

#     # Format final output
#     final_output = []
#     for group in grouped:
#         display_name = max(set(group["variants"]), key=group["variants"].count)
#         final_output.append({
#             "name": display_name,
#             "mentions": group["mentions"],
#             "variants": list(set(group["variants"])),
#             "norm_name": group["norm_name"]
#         })

#     return sorted(final_output, key=lambda x: x["mentions"], reverse=True)



    # prompt = f"""
        #         You are a smart assistant whose only job is to read the Reddit comments below and pull out the handful of *most‑frequently mentioned* specific entities (course titles, book names, tool names, company names, locations, people, etc.).  

        #         1. Group together any variants (e.g. “Django for Beginners” and “Django for Beginners by William Vincent”).  
        #         2. Count how many times each entity appears.  
        #         3. Return *only the top 15* entities, sorted by mention count descending.  
        #         4. Output exactly this JSON schema (no extra text):

        #         {{
        #         "entities": [
        #             {{ "name": "Entity A", "mentions": 12 }},
        #             {{ "name": "Entity B", "mentions": 8 }},
        #             … up to five entries …
        #         ]
        #         }}

        #         Reddit comments:
        #         {combined}
        #         """


    # return {
    #     "results": top_posts,
    #     "recommendations": recommendations,
    #     "entity_comments": entity_comment_links
    # }





# def group_and_count_entities(entities: list) -> list:
#     """Group entity variants and count mentions."""
#     grouped = defaultdict(list)
#     for ent in entities:
#         key = normalize_entity_name(ent['name'])
#         grouped[key].append(ent['name'])
#     result = []
#     for _, variants in grouped.items():
#         display = max(set(variants), key=variants.count)
#         result.append({"name": display, "mentions": len(variants)})
#     return sorted(result, key=lambda x: x['mentions'], reverse=True)



# def rerank_posts(posts: list, query: str, top_k=10):
#     q_emb = embedding_model.encode([query])[0]
#     scored = []

#     for post in posts:
#         content = f"{post['title']} ||| {post['selftext'][:300]}"
#         post_emb = embedding_model.encode([content])[0]
#         sim = cosine_similarity([q_emb], [post_emb])[0][0]
#         scored.append((sim, post))
    
#     sims = [s for s, _ in scored]
#     if sims:
#         print(f"📊 Similarity Debug → Avg: {sum(sims)/len(sims):.3f}, Min: {min(sims):.3f}, Max: {max(sims):.3f}")

#     # Sort all by similarity, regardless of threshold
#     scored.sort(key=lambda x: (x[0], x[1]['score']), reverse=True)
#     return [p for _, p in scored[:top_k]]


# def rerank_posts(posts: list, query: str, top_k=10, threshold=0.36):
#     """Embed and rerank posts by similarity to the query."""
#     q_emb = embedding_model.encode([query])[0]
#     scored = []
#     for post in posts:
#         # content = f"{post['title']} {post['selftext']}"
#         # Boost title weight + prioritize query keyword overlap
#         content = f"{post['title']} ||| {post['selftext'][:300]}"

#         post_emb = embedding_model.encode([content])[0]
#         sim = cosine_similarity([q_emb], [post_emb])[0][0]
#         if sim >= threshold:
#             scored.append((sim, post))
#     scored.sort(key=lambda x: (x[0], x[1]['score']), reverse=True)
#     return [p for _, p in scored[:top_k]]

# def rerank_posts(posts: list, query: str, top_k: int = 10):
    
#     model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
#     query_emb = model.encode([query])[0]
#     scored = []

#     for post in posts:
#         # Boost title by repeating it (gives more weight to relevance)
#         combined_text = f"{post['title']} {post['title']} ||| {post['selftext'][:300]}"
#         post_emb = model.encode([combined_text])[0]
#         sim = cosine_similarity([query_emb], [post_emb])[0][0]
#         scored.append((sim, post))

#     scored.sort(key=lambda x: (x[0], x[1]['score']), reverse=True)

#     # ⚠️ New filter: Only allow top posts if title has overlap with query
#     query_keywords = set(query.lower().split())
#     top_posts = []
#     for sim, post in scored:
#         title_words = set(post["title"].lower().split())
#         if query_keywords & title_words:
#             top_posts.append(post)
#         if len(top_posts) == top_k:
#             break

#     # Fallback: if fewer than 10, fill in with remaining high-similarity posts
#     if len(top_posts) < top_k:
#         for sim, post in scored:
#             if post not in top_posts:
#                 top_posts.append(post)
#             if len(top_posts) == top_k:
#                 break

#     return top_posts


# ✅ Clean entity variants like "Django docs" vs "Django documentation"
    

    # grouped_entities = defaultdict(lambda: {"mentions": 0, "variants": []})

    # for entity in recommendations:
    #     # variants = [v.strip() for v in entity["name"].split("/")]
    #     variants = re.split(r"[\/,()]+", entity["name"])
    #     variants = [v.strip() for v in variants if v.strip()]
    #     for variant in variants:
    #         norm = normalize_entity_name(variant)
    #         grouped_entities[norm]["mentions"] += entity["mentions"] // len(variants)
    #         grouped_entities[norm]["variants"].append(variant)

    # # ✅ Pick most frequent variant for display
    
    # # for norm_name, info in grouped_entities.items():
    # #     display_name = max(set(info["variants"]), key=info["variants"].count)
    # #     final_entities.append({
    # #         "name": display_name,
    # #         "mentions": info["mentions"],
    # #         "norm_name": norm_name  # 🔁 Add this for linking
    # #     })
    # final_entities = []
    # for norm_name, info in grouped_entities.items():
    #        display_name = max(set(info["variants"]), key=info["variants"].count)
    #        final_entities.append({
    #             "name":        display_name,
    #             "mentions":    info["mentions"],
    #             "norm_name":   norm_name,      # used as key
    #             "variants":    info["variants"]  # all of the original variants
    #         })

    # recommendations = sorted(final_entities, key=lambda x: x["mentions"], reverse=True)[:15]

# def map_entities_to_comments(entities, comments):
#     """Map each entity to up to 3 matching comment snippets."""
#     entity_comment_map = defaultdict(list)
#     for comment in comments:
#         for entity in entities:
#             name = entity["name"].lower() if isinstance(entity, dict) else entity[0].lower()

#             if name in comment.lower():
#                 if len(entity_comment_map[name]) < 3:
#                     entity_comment_map[name].append(comment.strip())
#     return entity_comment_map

# def search_reddit_posts(query: str, model: str = "hermes"):
#     posts = fetch_reddit_posts(query)
#     top_posts = rerank_posts(posts, query, top_k=10)

#     all_comments = []
#     comment_origin_map = []

#     for p in top_posts:
#         comments = fetch_top_comments(p['id'], limit=15)
#         all_comments.extend(comments)
#         comment_origin_map.extend([(c, p['id']) for c in comments])

#     comment_texts = [c['text'] for c in all_comments]
#     recommendations = extract_recommendations(comment_texts, model=model)

#     print(f"Model: {model} | Top Posts Found: {len(top_posts)}")
#     print(f"Model: {model} | Entities extracted: {recommendations}")

#     post_url_map = {p['id']: f"https://www.reddit.com/comments/{p['id']}" for p in top_posts}
#     entity_comment_links = {}

#     for entity in recommendations:
#         name = entity['name'].lower()
#         linked_comments = []
#         for comment_obj, post_id in comment_origin_map:
#             if name in comment_obj["text"].lower() and len(linked_comments) < 3:
#                 linked_comments.append({
#                     'text': comment_obj["text"].strip(),
#                     'url': comment_obj["url"]
#                 })
#         entity['comments'] = linked_comments  # <-- attach directly to each entity
#         entity_comment_links[name] = linked_comments

#     return {
#         'results': top_posts,
#         'recommendations': recommendations
#     }




# def search_reddit_posts(query: str, model: str = "hermes"):
#     posts = fetch_reddit_posts(query)
#     top_posts = rerank_posts(posts, query, top_k=10)

#     all_comments = []
#     for p in top_posts:
#         all_comments.extend(fetch_top_comments(p['id'], limit=15))
    

#     recommendations = extract_recommendations(all_comments, model=model)
#     print(f"Model: {model} | Top Posts Found: {len(top_posts)}")
#     print(f"Model: {model} | Entities extracted: {recommendations}")

#     return {
#         'results': top_posts,
#         'recommendations': recommendations
#     }




# def search_reddit_posts(query: str, model: str = "hermes"):
#     posts = fetch_reddit_posts(query)
#     top_posts = rerank_posts(posts, query)

#     all_comments = []
#     for p in top_posts:
#         all_comments.extend(fetch_top_comments(p['id']))
#     print(f"Total comments passed to model: {len(all_comments)}")

#     all_entities = extract_recommendations(all_comments, model=model) or []

#     standardized_entities = []
#     for ent in all_entities:
#         if isinstance(ent, dict):
#             standardized_entities.append(ent)
#         elif isinstance(ent, tuple) and len(ent) == 2:
#             standardized_entities.append({"name": ent[0], "mentions": ent[1]})

#     recommendations = group_and_count_entities(standardized_entities)

#     print(f"Model: {model} | Top Posts Found: {len(top_posts)}")
#     print(f"Model: {model} | Entities extracted: {recommendations}")

#     return {
#         'results': top_posts,
#         'recommendations': recommendations
#     }




# import praw
# import requests
# import json
# import re
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import Counter
# from . import reddit_config
# import openai
# import os
# from openai import OpenAI
# from .openai_utils import client
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # --- Setup ---

# # PRAW Reddit client
# reddit = praw.Reddit(
#     client_id=reddit_config.REDDIT_CLIENT_ID,
#     client_secret=reddit_config.REDDIT_CLIENT_SECRET,
#     user_agent=reddit_config.REDDIT_USER_AGENT
# )

# # Embedding model for reranking
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # LM Studio endpoint & model (unchanged)
# # LLM_STUDIO_URL = "http://10.125.141.244:1234/v1/chat/completions"
# LLM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

# LLM_MODEL = "hermes-3-llama-3.1-8b"

# # --- Helpers ---

# def fetch_reddit_posts(query: str, limit=200):  # Was previously too low
#     subreddit = reddit.subreddit("all")
#     posts = []

#     for submission in subreddit.search(query, sort="relevance", time_filter="all", limit=limit):
#         if submission.num_comments >= 3 and not submission.stickied:
#             posts.append({
#                 'id': submission.id,
#                 'title': submission.title,
#                 'selftext': submission.selftext or "",
#                 'url': submission.url,
#                 'score': submission.score,
#                 'comments_count': submission.num_comments
#             })

#     return posts


# def rerank_posts(posts: list, query: str, top_k=10, threshold=0.36):  # Lowered threshold
#     q_emb = embedding_model.encode([query])[0]
#     scored = []

#     for post in posts:
#         content = f"{post['title']} {post['selftext']}"
#         post_emb = embedding_model.encode([content])[0]
#         sim = cosine_similarity([q_emb], [post_emb])[0][0]
#         if sim >= threshold:
#             scored.append((sim, post))

#     scored.sort(key=lambda x: (x[0], x[1]['score']), reverse=True)
#     return [p for _, p in scored[:top_k]]


# def fetch_top_comments(post_id: str, limit=5):
#     """Pull the top `limit` upvoted comments from a submission."""
#     submission = reddit.submission(id=post_id)
#     submission.comment_sort = 'top'
#     submission.comments.replace_more(limit=0)
#     return [c.body.strip() for c in submission.comments[:limit] if hasattr(c, 'body')]



# # Load OpenAI key from environment (or configure manually)
# # openai.api_key = os.getenv("OPEN_API_KEY")
# client = OpenAI()

# def extract_recommendations(comments: list, model="hermes"):
#     combined = "\n\n".join(comments)

#     if model.lower().startswith("gpt"):
#         # GPT version using OpenAI API
#         prompt = f"""
#             Extract the exact names of products, tools, services, or proper nouns mentioned in the Reddit comments below.

#             - Only include clearly named things (e.g., course titles, book names, website names, companies, authors).
#             - Return them as a JSON array of raw strings, even if duplicates appear.
#             - Don't summarize, explain, or group. Just return the names mentioned as-is.

#             Reddit comments:
#             {combined}
#             """
#         # prompt = (
#         #     "From the following Reddit comments, extract and rank the most "
#         #     "recommended named entities (courses, books, tools, etc.). "
#         #     "Return the output as a valid JSON object mapping each entity "
#         #     "to its number of mentions.\n\n"
#         #     f"{combined}\n\n"
#         #     "Output format:\n{\n  \"Entity1\": count1,\n  \"Entity2\": count2\n}"
#         # )
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4",  # or gpt-4o, or gpt-3.5-turbo
#                 messages=[
#                     {"role": "system", 
#                      "content": ("You are a helpful assistant. Extract and rank the most "
#                         "recommended named entities (courses, books, tools, clubs, etc.) "
#                         "from the Reddit comments. "
#                     )},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.5,
#             )
#             content = response.choices[0].message.content
#             entity_list = json.loads(response.choices[0].message.content)



#             if isinstance(entity_list, list):
#                 counter = Counter([normalize_entity_name(e) for e in entity_list])
#                 result = [
#                     {"name":max([e for e in entity_list if normalize_entity_name(e) == k], key=entity_list.count), "mentions":v}
#                     for k, v in counter.items()
#                 ]
#                 return sorted(result, key=lambda x: x["mentions"], reverse=True)
#             else:
#                 print("Unexpected GPT Response format")
#                 return[]
            
            
#             # try:
#             #     entity_list = json.loads(content)

#             #     if isinstance(entity_list, dict):
#             #         # Convert old-style dict to list of dicts
#             #         entity_list = [{"name": k, "mentions": v} for k, v in entity_list.items()]

#             #     cleaned = group_and_count_entities(entity_list)
#             #     return cleaned
#             # except Exception as e:
#             #     print("Error parsing GPT JSON:", e)
#             #     print("Raw GPT output:", content)
#             #     return []

#             # match = re.search(r"\{.*?\}", content, re.DOTALL)
#             # if match:
#             #     entity_dict = json.loads(match.group())
#             #     return sorted(entity_dict.items(), key=lambda x: x[1], reverse=True)
#             # print("GPT raw content:", content)
#         except Exception as e:
#             print("OpenAI error:", e)
#             return []
            


#     else:
#         # Original Hermes version
#         payload = {
#             "model": LLM_MODEL,
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are a helpful assistant. Extract and rank the most "
#                         "recommended named entities (courses, books, tools, clubs, etc.) "
#                         "from the Reddit comments. Return a JSON mapping each entity "
#                         "to its mention count."
#                     )
#                 },
#                 {"role": "user", "content": combined}
#             ],
#             "temperature": 0.5
#         }
#         try:
#             r = requests.post(LLM_STUDIO_URL, headers={"Content-Type": "application/json"},
#                               data=json.dumps(payload))
#             res = r.json()
#             text = res['choices'][0]['message']['content']
#             m = re.search(r"\{.*\}", text, re.DOTALL)
#             if m:
#                 obj = json.loads(m.group())
#                 return sorted(obj.items(), key=lambda x: x[1], reverse=True)
#         except Exception as e:
#             print("Hermes error:", e)
#             return []

#     return []
# from collections import defaultdict

# def normalize_entity_name(name):
#     return name.strip().lower()

# def group_and_count_entities(entities):
#     """Group similar entity names regardless of case/spaces."""
#     grouped = defaultdict(list)
#     for ent in entities:
#         norm = normalize_entity_name(ent['name'])
#         grouped[norm].append(ent['name'])

#     result = []
#     for norm, items in grouped.items():
#         display = max(set(items), key=items.count)  # Most frequent variant
#         result.append({"name": display, "mentions": len(items)})

#     return sorted(result, key=lambda x: x["mentions"], reverse=True)


# def search_reddit_posts(query: str, model: str = "hermes"):
#     posts = fetch_reddit_posts(query)
#     top_posts = rerank_posts(posts, query)

#     all_comments = []
#     for p in top_posts:
#         all_comments.extend(fetch_top_comments(p['id']))

#     # 🛠 Pass model to extract_recommendations
#     recommendations = extract_recommendations(all_comments, model=model)

#     print(f"Model: {model} | Top Posts Found: {len(top_posts)}")
#     print(f"Model: {model} | Entities extracted: {recommendations}")

#     return {
#         'results': top_posts,
#         'recommendations': recommendations
#     }




# def extract_recommendations(comments: list):
#     """Your existing LLM-based entity extractor, unchanged."""
#     combined = "\n\n".join(comments)
#     payload = {
#         "model": LLM_MODEL,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a helpful assistant. Extract and rank the most "
#                     "recommended named entities (courses, books, tools, clubs, etc.) "
#                     "from the Reddit comments. Return a JSON mapping each entity "
#                     "to its mention count."
#                 )
#             },
#             {"role": "user", "content": combined}
#         ],
#         "temperature": 0.5
#     }
#     try:
#         r = requests.post(LLM_STUDIO_URL, headers={"Content-Type": "application/json"},
#                           data=json.dumps(payload))
#         res = r.json()
#         text = res['choices'][0]['message']['content']
#         m = re.search(r"\{.*\}", text, re.DOTALL)
#         if m:
#             obj = json.loads(m.group())
#             return sorted(obj.items(), key=lambda x: x[1], reverse=True)
#     except Exception as e:
#         print("LLM error:", e)
#     return []

# --- Main Function ---
# def search_reddit_posts(query: str):
#     posts = fetch_reddit_posts(query)
#     top_posts = rerank_posts(posts, query)

#     all_comments = []
#     for p in top_posts:
#         all_comments.extend(fetch_top_comments(p['id']))

#     recommendations = extract_recommendations(all_comments, model=model)
#     # recommendations = extract_recommendations(all_comments)

#     return {
#         'results': top_posts,
#         'recommendations': recommendations
#     }


# # reddit_api.py

# import praw
# import requests
# import json
# import re
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import Counter
# from . import reddit_config

# # Load embedding model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Set LM Studio endpoint and model
# LLM_STUDIO_URL = "http://10.125.141.244:1234/v1/chat/completions"
# LLM_MODEL = "hermes-3-llama-3.1-8b"

# # Setup Reddit API
# reddit = praw.Reddit(
#     client_id=reddit_config.REDDIT_CLIENT_ID,
#     client_secret=reddit_config.REDDIT_CLIENT_SECRET,
#     user_agent=reddit_config.REDDIT_USER_AGENT
# )


# def fetch_reddit_posts(query: str, limit=50):
#     """Fetch posts from Reddit 'all' using relevance search."""
#     posts = []
#     try:
#         for submission in reddit.subreddit("all").search(query, sort="relevance", time_filter="all", limit=limit):
#             if submission.num_comments < 2:
#                 continue
#             posts.append({
#                 'id': submission.id,
#                 'title': submission.title,
#                 'selftext': submission.selftext or "",
#                 'url': submission.url,
#                 'score': submission.score,
#                 'comments_count': submission.num_comments
#             })
#     except Exception as e:
#         print(f"Reddit fetch error: {e}")
#     return posts


# def rerank_posts(posts: list, query: str, top_k=10, threshold=0.3):
#     """Rerank posts using semantic similarity."""
#     query_embedding = embedding_model.encode([query])[0]
#     scored = []
#     for post in posts:
#         combined_text = f"{post['title']} {post['selftext']}"
#         post_embedding = embedding_model.encode([combined_text])[0]
#         similarity = cosine_similarity([query_embedding], [post_embedding])[0][0]
#         if similarity >= threshold:
#             scored.append((similarity, post))
#     scored.sort(key=lambda x: (x[0], x[1]['score']), reverse=True)
#     return [item[1] for item in scored[:top_k]]


# def fetch_top_comments(post_id: str, limit=5):
#     """Fetch top upvoted comments from a Reddit post."""
#     try:
#         submission = reddit.submission(id=post_id)
#         submission.comment_sort = 'top'
#         submission.comments.replace_more(limit=0)
#         return [c.body.strip() for c in submission.comments[:limit] if hasattr(c, 'body')]
#     except Exception as e:
#         print(f"Comment fetch error: {e}")
#         return []


# def extract_recommendations(comments: list):
#     """Extract top recommended entities from comments using LLM."""
#     combined = "\n\n".join(comments)
#     payload = {
#         "model": LLM_MODEL,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a helpful assistant. Extract and rank the most "
#                     "recommended named entities (like courses, tools, attorneys, restaurants, clubs, etc.) "
#                     "from the Reddit comments. Return a JSON mapping each entity to its number of mentions."
#                 )
#             },
#             {"role": "user", "content": combined}
#         ],
#         "temperature": 0.5
#     }
#     try:
#         response = requests.post(
#             LLM_STUDIO_URL,
#             headers={"Content-Type": "application/json"},
#             data=json.dumps(payload)
#         )
#         result = response.json()
#         content = result['choices'][0]['message']['content']
#         match = re.search(r"\{.*\}", content, re.DOTALL)
#         if match:
#             json_obj = json.loads(match.group())
#             return sorted(json_obj.items(), key=lambda x: x[1], reverse=True)
#     except Exception as e:
#         print("LLM extraction error:", e)
#     return []


# def search_reddit_posts(query: str):
#     """Main function to fetch posts, rerank, extract comments, and generate entity recommendations."""
#     raw_posts = fetch_reddit_posts(query)
#     reranked_posts = rerank_posts(raw_posts, query)
#     all_comments = []
#     for post in reranked_posts:
#         all_comments.extend(fetch_top_comments(post['id']))
#     recommendations = extract_recommendations(all_comments)

#     return {
#         'results': reranked_posts,
#         'recommendations': recommendations
#     }








# import praw
# import requests
# import json
# import re
# import spacy
# from collections import Counter
# from . import reddit_config

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Initialize Reddit API client
# reddit = praw.Reddit(
#     client_id=reddit_config.REDDIT_CLIENT_ID,
#     client_secret=reddit_config.REDDIT_CLIENT_SECRET,
#     user_agent=reddit_config.REDDIT_USER_AGENT
# )

# # Set LM Studio endpoint and model
# LM_STUDIO_URL = "http://10.125.141.244:1234/v1/chat/completions"
# LLM_MODEL = "hermes-3-llama-3.1-8b"


# def fetch_reddit_posts(query, limit=10):
#     subreddit = reddit.subreddit("all")
#     posts = subreddit.search(query, limit=limit, sort="relevance")

#     results = []
#     for post in posts:
#         results.append({
#             'title': post.title,
#             'url': post.url,
#             'score': post.score,
#             'comments_count': post.num_comments,
#             'id': post.id
#         })

#     return results


# def fetch_top_comments(post_id, limit=5):
#     submission = reddit.submission(id=post_id)
#     submission.comment_sort = 'top'
#     submission.comments.replace_more(limit=0)
#     top_comments = [comment.body for comment in submission.comments[:limit]]
#     return top_comments


# def extract_recommendations(comments):
#     combined_comments = "\n\n".join(comments)

#     payload = {
#         "model": LLM_MODEL,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant. Extract and rank the most recommended named entities or items (such as courses, books, tools, etc.) from the Reddit comments. Return a JSON object mapping each entity to its number of mentions."
#             },
#             {
#                 "role": "user",
#                 "content": combined_comments
#             }
#         ],
#         "temperature": 0.5
#     }

#     try:
#         response = requests.post(
#             LM_STUDIO_URL,
#             headers={"Content-Type": "application/json"},
#             data=json.dumps(payload)
#         )
#         result = response.json()

#         if 'choices' in result and len(result['choices']) > 0:
#             text = result['choices'][0]['message']['content']

#             match = re.search(r"\{.*\}", text, re.DOTALL)
#             if match:
#                 json_obj = json.loads(match.group())
#                 sorted_items = sorted(json_obj.items(), key=lambda x: x[1], reverse=True)
#                 return sorted_items

#     except Exception as e:
#         print("LLM error:", e)

#     return []


# def search_reddit_posts(query):
#     posts = fetch_reddit_posts(query)
#     all_comments = []

#     for post in posts:
#         top_comments = fetch_top_comments(post['id'])
#         all_comments.extend(top_comments)

#     recommendations = extract_recommendations(all_comments)

#     return {
#         'results': posts,
#         'recommendations': recommendations
#     }


# import praw
# import re
# import json
# import spacy
# import requests
# import difflib
# from . import reddit_config
# from collections import Counter

# # Load spaCy (optional but helps LLM generalize better)
# nlp = spacy.load("en_core_web_sm")

# # Reddit API credentials
# reddit = praw.Reddit(
#     client_id=reddit_config.REDDIT_CLIENT_ID,
#     client_secret=reddit_config.REDDIT_CLIENT_SECRET,
#     user_agent=reddit_config.REDDIT_USER_AGENT
# )

# # Talk to LM Studio
# def ask_llm(prompt, port=1234):
#     url = f"http://localhost:{port}/v1/completions"
#     headers = {"Content-Type": "application/json"}
#     payload = {
#         "prompt": prompt,
#         "max_tokens": 200,
#         "temperature": 0.7,
#         "stop": ["</s>"]
#     }

#     try:
#         response = requests.post(url, json=payload, headers=headers)
#         if response.ok:
#             return response.json()['choices'][0]['text'].strip()
#         else:
#             print("LLM error:", response.text)
#             return None
#     except Exception as e:
#         print("Connection error to LM Studio:", e)
#         return None

# # Main function to fetch Reddit posts
# def search_reddit_posts(query, limit=15):
#     recommendation_counter = Counter()
#     detailed_results = []

#     for submission in reddit.subreddit("all").search(query, sort="relevance", time_filter="all", limit=limit):
#         # 1. Semantic filtering using difflib
#         title_similarity = difflib.SequenceMatcher(None, query.lower(), submission.title.lower()).ratio()
#         if title_similarity < 0.35:
#             print("Skipping low similarity title:", submission.title)
#             continue

#         if submission.num_comments < 3:
#             continue

#         submission.comments.replace_more(limit=0)
#         comments = submission.comments.list()
#         all_comments = [comment.body for comment in comments]

#         # 2. Check if query is mentioned in at least one top-level comment
#         top_comments = " ".join([comment.body.lower() for comment in comments[:5]])
#         if not any(word in top_comments for word in query.lower().split()):
#             continue

#         # 3. Concatenate max 12,000 chars from comments for token-safe prompt
#         combined_comments = ""
#         max_chars = 12000
#         for comment in all_comments:
#             if len(combined_comments) + len(comment) < max_chars:
#                 combined_comments += comment + "\n"
#             else:
#                 break

#         # 4. Prompt the LLM
#         prompt = f"""
# You are a helpful assistant. Based only on the following Reddit comments, extract a dictionary of entities (e.g., names of people, products, services, tools, platforms, attorneys, etc.) that are positively recommended or endorsed related to: "{query}".

# Output should be a valid JSON dictionary like:
# {{"entity name": count}}

# If nothing relevant is found, return an empty dictionary.

# Comments:
# {combined_comments}

# Answer:
# """

#         llm_response = ask_llm(prompt)

#         # 5. Extract dictionary from LLM response safely
#         try:
#             if llm_response:
#                 match = re.search(r"\{.*\}", llm_response, re.DOTALL)
#                 if match:
#                     extracted = json.loads(match.group(0))
#                     for name, count in extracted.items():
#                         recommendation_counter[name.lower()] += count
#                 else:
#                     print("No JSON dictionary found in response.")
#         except Exception as e:
#             print("Failed to parse LLM output:", e)

#         # Store metadata
#         detailed_results.append({
#             'title': submission.title,
#             'url': submission.url,
#             'score': submission.score,
#             'comments_count': submission.num_comments,
#         })

#     return {
#         'recommendations': recommendation_counter.most_common(10),
#         'results': detailed_results
#     }


# import praw
# import re
# import nltk
# import spacy
# from . import reddit_config
# from collections import defaultdict, Counter
# from nltk.corpus import stopwords


# nlp = spacy.load("en_core_web_sm")

# reddit = praw.Reddit(
#     client_id=reddit_config.REDDIT_CLIENT_ID,
#     client_secret=reddit_config.REDDIT_CLIENT_SECRET,
#     user_agent=reddit_config.REDDIT_USER_AGENT


# )

# stop_words = set(stopwords.words('english'))
# def clean_and_tokenize(comment):
#     comment=comment.lower()
#     comment=re.sub(r'[^a-zA-Z0-9\s]', '', comment)
#     words=comment.split()
#     return [word for word in words if word not in stop_words]

# def search_reddit_posts(query, limit=5):
#     entity_counter = Counter()
#     detailed_results = []

#     for submission in reddit.subreddit("all").search(query, limit=limit):
#         submission.comments.replace_more(limit=0)
#         all_comments = [comment.body for comment in submission.comments.list()]
        
#         for comment in all_comments:
#             doc = nlp(comment)
#             for ent in doc.ents:
#                 if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]:  # Relevant types
#                     entity_counter[ent.text.lower()] += 1

#         detailed_results.append({
#             'title': submission.title,
#             'url': submission.url,
#             'score': submission.score,
#             'comments_count': submission.num_comments,
#         })

#     top_entities = entity_counter.most_common(10)

#     return {
#         'top_keywords': top_entities,
#         'results': detailed_results
#     }


# def search_reddit_posts(query, limit=5):
#     word_counter = Counter()
#     detailed_results = []

#     for submission in reddit.subreddit("all").search(query, limit=limit):
#         submission.comments.replace_more(limit=0)
#         all_comments = [comment.body for comment in submission.comments.list()]
        
#         # Tokenize and count
#         for comment in all_comments:
#             words = clean_and_tokenize(comment)
#             word_counter.update(words)

#         detailed_results.append({
#             'title': submission.title,
#             'url': submission.url,
#             'score': submission.score,
#             'comments_count': submission.num_comments,
#         })

#     # Get top 10 most common words
#     top_keywords = word_counter.most_common(10)

#     return {
#         'top_keywords': top_keywords,
#         'results': detailed_results
#     }