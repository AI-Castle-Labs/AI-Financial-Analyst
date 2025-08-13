from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from prompt import system_source_prompt,system_macro_prompt,system_sector_research_analyst,system_central_bank_prompt,system_fx_research_prompt,system_agent_prompt,system_portfolio_manager_prompt
from schema import PlanningSchema,LLMScore,SonarInput
import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import json

load_dotenv()

# Initialize OpenAI client once
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def planner_agent(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)
    system_prompt = (
        """
        <role>
        You are an AI Research Agent responsible for generating 5 different investment scenarios for the given prompt.
        </role>
        <Format>
        Your response must be a valid JSON array of objects, each with the following fields:\n
        - idea: The name of the investment scenario (do not use 'idea_name', just the actual idea).\n
        - idea_description: A detailed description of the scenario.\n
        - confidence_score: A number between 0 and 1 representing your confidence in this scenario (0 = lowest, 1 = highest).\n
        </Format>
        <Example>
        Example:\n"
        "[\n"
        "  {\n"
        "    \"idea\": \"Long Brazil Economy\",\n"
        "    \"idea_description\": \"Invest in Brazilian equities due to strong growth outlook and favorable monetary policy.\",\n"
        "    \"confidence_score\": 0.85\n"
        "  },\n"
        "  ...\n"
        "]"
        </Example>
        """
    )
    result = llm.invoke([
          {'role':'system','content' : system_prompt},
          {'role':'user','content' : f"Provide sector research analysis for {prompt}"}
        
    ])
    # Ensure the model returned valid JSON for scenarios
    raw = result.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) > 1:
            block = parts[1]
            block = block.lstrip()
            if block.startswith("json"):
                block = block.split("\n", 1)[1] if "\n" in block else ""
            raw = block.strip()
    scenarios = json.loads(raw)

    # Get similarity scores for the scenarios and sort them by similarity_score (desc)
    scores = get_llm_score(prompt, scenarios)
    sorted_by_similarity = sorted(scores, key=lambda x: x.get('similarity_score', 0), reverse=True)
    print("Sorted by similarity", sorted_by_similarity)

    # Return all for downstream processing
    return scenarios, scores, sorted_by_similarity



def get_llm_score(query: str, idea_text: json) -> list[dict]:
    
    """
    A LLM Score focused on ranking the ideas with the query based on relevance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)

    # Pass the ideas as proper JSON to the model
    ideas_json = json.dumps(idea_text, ensure_ascii=False)

    prompt = (
        f"Compare the following ideas (JSON):\n{ideas_json}\n\n"
        f"against the query: \"{query}\" and return a JSON array. You are a part of AI Investment Analyst where you are responsible for "
        "conducting a similarity score focus on the context and how relevant it will be for an investment analyst. You should assign score based on relevance, but also credit ideas"
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
         {'role':'system', 'content': prompt},
         {'role':'user','content' : "Compare the prompt and ideas and focus on how relevant it is to the required prompt"}
    ])

    raw = result.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) > 1:
            block = parts[1]
            block = block.lstrip()
            if block.startswith("json"):
                block = block.split("\n", 1)[1] if "\n" in block else ""
            raw = block.strip()

    result = json.loads(raw)

    print("LLM SCORE", result)
    return result


# --- Embedding utilities (standalone) ---

def get_embedding(text: str):
    # Use OpenAI v1 client
    resp = _openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


def cosine_similarity(a, b) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def add_embedding_scores(query: str, ideas: list[dict]) -> list[dict]:
    """Compute embedding-based similarity per idea and return list[{idea, embedding_similarity}]."""
    query_embedding = get_embedding(query)
    results = []
    for item in ideas:
        idea_text = f"{item.get('idea','')}: {item.get('idea_description','')}"
        idea_embedding = get_embedding(idea_text)
        sim = cosine_similarity(query_embedding, idea_embedding)
        results.append({
            'idea': item.get('idea',''),
            'embedding_similarity': sim
        })
    return results


# --- Hybrid reranker ---

def hybrid_rerank_run(llm_scores: list[dict], embedding_scores: list[dict], sorted_by_similarity: list[dict]):
    api_key = os.getenv("OPENAI_API_KEY")
    llm = init_chat_model("gpt-4o-2024-08-06", temperature=0.0, model_provider="openai", api_key=api_key)

    payload = {
        'llm_scores': llm_scores,
        'embedding_scores': embedding_scores,
        'sorted_by_similarity': sorted_by_similarity
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
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': f"Rerank using these inputs:\n{content}"}
    ])

    raw = result.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) > 1:
            block = parts[1]
            block = block.lstrip()
            if block.startswith("json"):
                block = block.split("\n", 1)[1] if "\n" in block else ""
            raw = block.strip()

    ranked = json.loads(raw)

    # Enforce ordering (most-to-least) and drop the last (lowest-ranked) item
    try:
        if isinstance(ranked, list):
            ranked = sorted(
                ranked,
                key=lambda x: x.get('final_score'),
                reverse=True
            )
            if len(ranked) > 0:
                ranked = ranked[:-1]
    except Exception:
        pass

    return ranked


def run(prompt: str):
    scenarios, scores, sorted_scores = planner_agent(prompt)
    embedding_scores = add_embedding_scores(prompt, scenarios)
    final_ranking = hybrid_rerank_run(scores, embedding_scores, sorted_scores)
    print("Final ranking:", final_ranking)
    return final_ranking


# Entry point
run("Long Brazil Economy")