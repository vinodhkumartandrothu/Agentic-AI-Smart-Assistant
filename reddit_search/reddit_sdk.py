# reddit_sdk.py
import re
from fuzzywuzzy import fuzz
from collections import defaultdict


from .reddit_api import (
    fetch_reddit_posts,
    rerank_posts,
    fetch_top_comments,
    extract_recommendations,
    search_reddit_posts,
)


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

def match_entity_to_comment(entity_name: str, comment_text: str, threshold: int = 80) -> bool:
    norm_entity = normalize_entity_name(entity_name)
    norm_comment = normalize_entity_name(comment_text)
    variants = generate_entity_variants(entity_name)

    # if any(term in comment_text.lower() for term in variants):
    if any(re.search(r"\b" + re.escape(term) + r"\b", comment_text.lower()) for term in variants):

        return True

    from fuzzywuzzy import fuzz
    return fuzz.partial_ratio(norm_entity, norm_comment) >= threshold


class RedditAssistantSDK:
    def __init__(self, model: str = "hermes"):
        self.model = model

    def search_posts(self, query: str, limit: int = 200):
        """Fetch posts from Reddit"""
        return fetch_reddit_posts(query, limit=limit)

    def rerank_posts(self, posts: list, query: str, top_k: int = 10):
        """Rerank posts by semantic similarity"""
        return rerank_posts(posts, query, top_k=top_k)

    def get_top_comments(self, post_id: str, limit: int = 10):
        """Fetch top comments for a Reddit post"""
        return fetch_top_comments(post_id, limit=limit)

    # def extract_entities(self, comments: list):
    #     """Extract top recommended entities from Reddit comments"""
    #     return extract_recommendations(comments, model=self.model)

    def extract_entities(self, comments: list):
        """Extract top recommended entities from Reddit comments"""
        # FIX: Convert list of dicts → list of strings
        texts = [c["text"] for c in comments if isinstance(c, dict) and "text" in c]
        return extract_recommendations(texts, model=self.model)
    
    def run_full_pipeline(self, query: str):
            """
            Hybrid pipeline:
            ✅ 1. Raw Reddit search (no GPT rewriting)
            ✅ 2. Rerank posts using MiniLM
            ✅ 3. Fetch top comments from each post
            ✅ 4. Use LLM (GPT/Hermes) to extract top 15 entities
            ✅ 5. Map 3 relevant comments per entity
            """
            print(f"\n🔁 [SDK] Calling search_reddit_posts() with model: {self.model}")
            return search_reddit_posts(query, model=self.model)
 
    def format_output(self, pipeline_result: dict):
        """
        Format the output of `run_full_pipeline()` for UI or agent use.
        Directly uses comments attached during the entity extraction step.
        """

        if not pipeline_result:
            return {"error": "No results found."}

        # Step 1: Collect Reddit posts
        top_posts = [
            {
                "title": post["title"],
                "url": post["url"],
                "score": post["score"],
                "comments_count": post["comments_count"]
            }
            for post in pipeline_result.get("results", [])
        ]

        # Step 2: Use already cleaned entities from pipeline_result
        clean_entities = pipeline_result.get("recommended_entities", [])

        return {
            "top_posts": top_posts,
            "entities": clean_entities
        }



   









   


   
    # def run_full_pipeline(self, query: str):
    #     """
    #     Hybrid pipeline:
    #     1. Manually fetch Reddit posts.
    #     2. Rerank posts.
    #     3. Fetch comments from top posts.
    #     4. Extract entities using LLM.
    #     5. Match each entity to 3 relevant comments (clean + accurate).
    #     """

    #     # Step 1: Manual Reddit search
    #     posts = self.search_posts(query)
    #     top_posts = self.rerank_posts(posts, query)

    #     # Step 2: Fetch top comments
    #     all_comments = []
    #     comment_origin_map = []
    #     for post in top_posts:
    #         post_id = post["id"]
    #         comments = self.get_top_comments(post_id, limit=15)
    #         all_comments.extend(comments)
    #         comment_origin_map.extend([(c, post_id) for c in comments])

    #     # Step 3: Extract entities using LLM
    #     raw_entities = self.extract_entities(all_comments)
    #     if not raw_entities:
    #         return {
    #             "results": top_posts,
    #             "recommended_entities": []
    #         }

    #     # Step 4: Normalize + group variants
    #     grouped = defaultdict(lambda: {"mentions": 0, "variants": []})
    #     for entity in raw_entities:
    #         variants = re.split(r"[\/,()]+", entity["name"])
    #         variants = [v.strip() for v in variants if v.strip()]
    #         for v in variants:
    #             norm = normalize_entity_name(v)
    #             grouped[norm]["mentions"] += entity["mentions"] // len(variants)
    #             grouped[norm]["variants"].append(v)

    #     final_entities = []
    #     for norm_name, info in grouped.items():
    #         display = max(set(info["variants"]), key=info["variants"].count)
    #         final_entities.append({
    #             "name": display,
    #             "mentions": info["mentions"],
    #             "norm_name": norm_name,
    #             "variants": info["variants"]
    #         })

    #     recommendations = sorted(final_entities, key=lambda x: x["mentions"], reverse=True)[:15]

    #     # Step 5: Map top 3 matching comments for each entity
    #     post_url_map = {p['id']: f"https://www.reddit.com/comments/{p['id']}" for p in top_posts}
    #     entity_comment_links = {}

    #     for entity in recommendations:
    #         norm_name = normalize_entity_name(entity['name'])
    #         entity_terms = generate_entity_variants(entity['name'])

    #         matched_comments = []
    #         seen_texts = set()

    #         for comment_obj, post_id in comment_origin_map:
    #             text = comment_obj["text"]
    #             score = comment_obj.get("score", 0)
    #             url = post_url_map.get(post_id, "#")

    #             if any(re.search(r"\b" + re.escape(term) + r"\b", text.lower()) for term in entity_terms):
    #                 if text.strip() not in seen_texts:
    #                     seen_texts.add(text.strip())
    #                     matched_comments.append({
    #                         "text": text.strip(),
    #                         "url": url,
    #                         "score": score
    #                     })

    #         top_comments = sorted(matched_comments, key=lambda x: x["score"], reverse=True)[:3]
    #         entity_comment_links[norm_name] = top_comments

    #     # Final output
    #     return {
    #         "results": top_posts,
    #         "recommended_entities": [
    #             {
    #                 "name": entity["name"],
    #                 "mentions": entity["mentions"],
    #                 "comments": entity_comment_links.get(entity["norm_name"], [])
    #             }
    #             for entity in recommendations
    #         ]
    #     }



    # def run_full_pipeline(self, query: str):
    #     """Run the complete search → rerank → comment → extract flow"""
    #     return search_reddit_posts(query, model=self.model)

    # def run_full_pipeline(self, query: str):
    #     """
    #     Hybrid pipeline:
    #     1. Search Reddit posts manually using keyword query.
    #     2. Rerank posts using semantic similarity.
    #     3. Fetch top comments from those posts.
    #     4. Use Agentic AI model to extract recommended entities from comments.
    #     """
    #     # Step 1: Manual Reddit search
    #     posts = self.search_posts(query)

    #     # Step 2: Semantic reranking
    #     top_posts = self.rerank_posts(posts, query)

    #     # Step 3: Fetch top comments
    #     all_comments = []
    #     for post in top_posts:
    #         post_id = post["id"]
    #         comments = self.get_top_comments(post_id)
    #         all_comments.extend(comments)

    #     # Step 4: Agentic AI extracts entities from collected comments
    #     # recommended_entities = self.extract_entities(all_comments)
    #     # Step 4: Agentic AI extracts entities from collected comments
    #     raw_entities = self.extract_entities(all_comments)

    #     # Map post_id → URL
    #     post_url_map = {p['id']: f"https://www.reddit.com/comments/{p['id']}" for p in top_posts}
    #     comment_origin_map = [(c, c["post_id"]) for c in all_comments if "post_id" in c]
    #     entity_comment_links = {}

    #     # Group entity variants
    #     from .reddit_api import normalize_entity_name, generate_entity_variants
    #     for entity in raw_entities:
    #         name = normalize_entity_name(entity['name'])
    #         matched_comments = []
    #         seen_texts = set()  # ✅ FIXED: Now scoped to each entity

    #         for comment_obj, post_id in comment_origin_map:
    #             text = comment_obj["text"]
    #             score = comment_obj.get("score", 0)
    #             url = post_url_map.get(post_id, "#")

    #             if match_entity_to_comment(entity["name"], text):
    #                 if text.strip() not in seen_texts:
    #                     seen_texts.add(text.strip())
    #                     matched_comments.append({
    #                         "text": text.strip(),
    #                         "url": url,
    #                         "score": score
    #                     })



    #         # Sort by score descending
    #         top_comments = sorted(matched_comments, key=lambda x: x["score"], reverse=True)[:3]
    #         entity_comment_links[name] = top_comments


    #     return {
    #             "results": top_posts,
    #             "recommended_entities": [
    #                 {
    #                     "name": entity["name"],
    #                     "mentions": entity["mentions"],
    #                     "comments": entity_comment_links.get(normalize_entity_name(entity["name"]), [])

    #                     # "comments": entity_comment_links.get(entity["name"].lower(), [])
    #                 }
    #                 for entity in raw_entities
    #             ]
    #     }

        # final_entities = []

        # for entity in raw_entities:
        #     name = entity["name"]
        #     count = entity["mentions"]
        #     norm = normalize_entity_name(name)

        #     variants = generate_entity_variants(name)
        #     matched_comments = []
        #     seen_texts = set()

        #     for comment_obj, post_id in comment_origin_map:
        #         text = comment_obj["text"]
        #         score = comment_obj.get("score", 0)
        #         url = post_url_map.get(post_id, "#")

        #         if any(term in text.lower() for term in variants):
        #             if text.strip() not in seen_texts:
        #                 seen_texts.add(text.strip())
        #                 matched_comments.append({
        #                     "text": text.strip(),
        #                     "url": url,
        #                     "score": score
        #                 })

        #     top_comments = sorted(matched_comments, key=lambda x: x["score"], reverse=True)[:3]

        #     final_entities.append({
        #         "name": name,
        #         "mentions": count,
        #         "comments": top_comments
        #     })

        # return {
        #     "results": top_posts,
        #     "all_comments": all_comments,
        #     "recommended_entities": final_entities
        # }


                # return {
                #     "results": top_posts,
                #     "all_comments": all_comments,
                #     "recommended_entities": recommended_entities
                # }

    
    
    # def format_output(self, pipeline_result: dict):
    #     """
    #     Format the output of `run_full_pipeline()` for UI or agent use.
    #     Filters entities to ensure only relevant and clean comment snippets are shown.
    #     """

    #     if not pipeline_result:
    #         return {"error": "No results found."}

    #     # Step 1: Collect all Reddit posts (for metadata)
    #     top_posts = [
    #         {
    #             "title": post["title"],
    #             "url": post["url"],
    #             "score": post["score"],
    #             "comments_count": post["comments_count"]
    #         }
    #         for post in pipeline_result.get("results", [])
    #     ]

    #     # Step 2: Clean and structure entities
    #     raw_entities = pipeline_result.get("recommended_entities", [])
    #     all_comments = pipeline_result.get("all_comments", [])  # this must be present from search_reddit_posts()

    #     clean_entities = []

    #     for entity in raw_entities:
    #         name = entity["name"]
    #         count = entity["mentions"]

    #         base = re.sub(r"[^\w\s]", "", name.lower())
    #         patterns = [
    #             name.lower(),
    #             base,
    #             base.replace(" ", ""),
    #             base.replace(" ", "-"),
    #         ]

    #         matched = []
    #         for comment in all_comments:
    #             text = comment["text"].lower()

    #             if any(re.search(rf"\b{re.escape(p)}\b", text) for p in patterns):
    #                 matched.append(comment)

    #         # Optional: sort by comment score (if present), and get top 3
    #         sorted_comments = sorted(matched, key=lambda x: x.get("score", 0), reverse=True)[:3]

    #         if sorted_comments:
    #             clean_entities.append({
    #                 "name": name,
    #                 "mentions": count,
    #                 "comments": sorted_comments
    #             })


    #         # Collect only comments where the entity is actually mentioned (case-insensitive match)
    #         # relevant_comments = [
    #         #     comment for comment in all_comments
    #         #     if name.lower() in comment["text"].lower()][:3]  # limit to top 3

    #         # if relevant_comments:
    #         #     clean_entities.append({
    #         #         "name": name,
    #         #         "mentions": count,
    #         #         "comments": relevant_comments
    #         #     })

    #     return {
    #         "top_posts": top_posts,
    #         "entities": clean_entities
    #     }
