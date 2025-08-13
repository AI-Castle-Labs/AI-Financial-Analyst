from langchain.chat_models import init_chat_model
from prompt import system_planner_prompt
import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import json
from typing import List, Dict

from extra_tools import ask_sonar
from langchain.tools import StructuredTool

load_dotenv()

# Initialize OpenAI v1 client once (for embeddings)
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _strip_code_fences(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) > 1:
            block = parts[1].lstrip()
            if block.startswith("json"):
                block = block.split("\n", 1)[1] if "\n" in block else ""
            raw = block.strip()
    return raw


available_tools = [
    StructuredTool.from_function(
        name="Sonar_Search",
        func=ask_sonar,
        schema=None,  # ask_sonar takes a single string query
        description="Conducts online search for the given query"
    )
]


class DeepSearchTool:

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.api_key = api_key
        self.llm = init_chat_model(
            "gpt-4o-2024-08-06",
            temperature=0.0,
            model_provider="openai",
            api_key=api_key,
        )

    def run(self, prompt: str, levels: int = 1) -> List[Dict]:
        """Run planning + hybrid ranking once and return final_ranking list."""
        bfs = self.BFS(self.api_key)
        scenarios = bfs.planner_agent(prompt)
        llm_scores = self.Ranking.get_llm_score(prompt, scenarios)
        sorted_by_similarity = sorted(
            llm_scores, key=lambda x: x.get("similarity_score", 0.0), reverse=True
        )
        embedding_scores = self.Ranking.add_embedding_scores(prompt, scenarios)
        final_ranking = self.Ranking.hybrid_rerank_run(
            llm_scores, embedding_scores, sorted_by_similarity
        )
        return final_ranking

    class BFS:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.llm = init_chat_model(
                "gpt-4o-2024-08-06",
                temperature=0.0,
                model_provider="openai",
                api_key=api_key,
            )

        def planner_agent(self, prompt: str) -> List[Dict]:
            system_prompt = (
                "You are an AI Research Agent responsible for generating 5 different investment scenarios for the given prompt. "
                "Your response must be a valid JSON array of objects, each with the following fields:\n"
                "- idea: The name of the investment scenario (do not use 'idea_name', just the actual idea).\n"
                "- idea_description: A detailed description of the scenario.\n"
                "- confidence_score: A number between 0 and 1 representing your confidence in this scenario (0 = lowest, 1 = highest).\n"
                "Example:\n"
                "[\n"
                "  {\n"
                "    \"idea\": \"Long Brazil Economy\",\n"
                "    \"idea_description\": \"Invest in Brazilian equities due to strong growth outlook and favorable monetary policy.\",\n"
                "    \"confidence_score\": 0.85\n"
                "  },\n"
                "  ...\n"
                "]"
            )
            result = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Provide sector research analysis for {prompt}"},
            ])
            raw = _strip_code_fences(result.content)
            scenarios = json.loads(raw)
            return scenarios

        def question_agent(self, prompt: str) -> Dict:
            result = self.llm.invoke([
                {"role": "system", "content": system_planner_prompt},
                {"role": "user", "content": f"Provide the next research graph level for: {prompt}"},
            ])
            raw = _strip_code_fences(result.content)
            return json.loads(raw)

    class Ranking:
        @staticmethod
        def get_embedding(text: str):
            resp = _openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return resp.data[0].embedding

        @staticmethod
        def cosine_similarity(a, b) -> float:
            a = np.array(a, dtype=float)
            b = np.array(b, dtype=float)
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            return float(np.dot(a, b) / denom)

        @staticmethod
        def add_embedding_scores(query: str, ideas: List[Dict]) -> List[Dict]:
            query_embedding = DeepSearchTool.Ranking.get_embedding(query)
            results = []
            for item in ideas:
                idea_text = f"{item.get('idea','')}: {item.get('idea_description','')}"
                idea_embedding = DeepSearchTool.Ranking.get_embedding(idea_text)
                sim = DeepSearchTool.Ranking.cosine_similarity(query_embedding, idea_embedding)
                results.append({
                    "idea": item.get("idea", ""),
                    "embedding_similarity": sim,
                })
            return results

        @staticmethod
        def get_llm_score(query: str, idea_text: List[Dict]) -> List[Dict]:
            api_key = os.getenv("OPENAI_API_KEY")
            llm = init_chat_model(
                "gpt-4o-2024-08-06",
                temperature=0.0,
                model_provider="openai",
                api_key=api_key,
            )
            ideas_json = json.dumps(idea_text, ensure_ascii=False)
            prompt = (
                f"Compare the following ideas (JSON):\n{ideas_json}\n\n"
                f"against the query: \"{query}\" and return a JSON array. You are a part of AI Investment Analyst where you are responsible for "
                "conducting a similarity score focus on the context and how relevant it will be for an investment analyst. You should assign score based on relevance, but also credit ideas" \
                "which are relevant."
                "Each object should have:\n"
                "- idea: The scenario name\n"
                "- similarity_score: A float between 0 and 1 representing similarity\n"
                "Example:\n"
                "[\n"
                "  {\"idea\": \"Long Brazil Economy\", \"similarity_score\": 0.92},\n"
                "  ...\n"
                "]"
            )
            result = llm.invoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Compare the prompt and ideas and focus on how relevant it is to the required prompt"},
            ])
            raw = _strip_code_fences(result.content)
            return json.loads(raw)

        @staticmethod
        def hybrid_rerank_run(llm_scores: List[Dict], embedding_scores: List[Dict], sorted_by_similarity: List[Dict]) -> List[Dict]:
            api_key = os.getenv("OPENAI_API_KEY")
            llm = init_chat_model(
                "gpt-4o-2024-08-06",
                temperature=0.0,
                model_provider="openai",
                api_key=api_key,
            )
            payload = {
                "llm_scores": llm_scores,
                "embedding_scores": embedding_scores,
                "sorted_by_similarity": sorted_by_similarity,
            }
            system = (
                "You are an AI Hybrid Reranker. Combine multiple signals to produce a final ordered list of ideas from most important to least. For your reasoning be factual do not just use reasons like similarity score, use factual concepts with that as well\n"
                "Guidelines:\n"
                "- Prefer higher LLM similarity_score and higher embedding_similarity.\n"
                "- Remove the single lowest-ranked idea from the final output.\n"
                "- If signals disagree, provide a balanced final ranking and include a brief reason per item.\n"
                "Return strictly valid JSON: a list of objects with fields: idea (str), final_score (0..1), reason (str)."
            )
            content = json.dumps(payload, ensure_ascii=False)
            result = llm.invoke([
                {"role": "system", "content": system},
                {"role": "user", "content": f"Rerank using these inputs:\n{content}"},
            ])
            raw = _strip_code_fences(result.content)
            ranked = json.loads(raw)
            try:
                if isinstance(ranked, list):
                    ranked = sorted(ranked, key=lambda x: x.get("final_score"), reverse=True)
                    if len(ranked) > 0:
                        ranked = ranked[:-1]
            except Exception:
                pass
            return ranked




