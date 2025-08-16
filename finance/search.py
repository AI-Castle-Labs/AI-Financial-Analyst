from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from prompt import system_planner_prompt,system_agent_prompt,system_portfolio_manager_prompt
from schema import PlanningSchema,LLMScore,SonarInput
import os
from dotenv import load_dotenv
from state import AgentState
import tweepy
from fpdf import FPDF
from extra_tools import chart_agent
from extra_tools import ask_sonar
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain, SimpleSequentialChain
from openai import OpenAI
import numpy as np
import json
from typing import Literal
from ai_finance_analyst import DeepResearchAgent

#from tavily import TavilyClient
from deepagents import create_deep_agent


load_dotenv()

# Initialize OpenAI v1 client once (for embeddings)
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Helper to strip code fences from LLM outputs

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

        name = "Sonar_Search", #Cannot have underscore because the LLM is particular on syntax

        func = ask_sonar,

        schema = SonarInput,

        description="Conducts online search for respective query"

    )


]




"""
Build a deep search method using BFS AND DFS
BFS
    -Planner Agent gives plan and agent role to 5 different nodes
    - Each node gives out their research and cite
    - 5 -> 4 Do deepth search, 4-> 3
DFS



"""





class DeepSearchTool:
    def __init__(self):
        self.llm = init_chat_model("gpt-4o-2024-08-06", temperature=0.0, model_provider="openai", api_key=os.getenv("OPENAI_API_KEY"))
        self.BFS = self.BFS(ideas={})  
        self.Ranking = self.Ranking(model=None, api_key=os.getenv("OPENAI_API_KEY"), query="")  # Instantiate Ranking
        self.DeepResearchAgent = DeepResearchAgent()

    def run(self,prompt,number) -> str:

        
        scenarios, scores, sorted_scores = self.BFS.planner_agent(prompt)
        embedding_scores = self.Ranking.add_embedding_scores(prompt, scenarios)
        final_ranking = self.Ranking.hybrid_rerank_run(scores, embedding_scores, sorted_scores)

        # Keep track of the last non-empty final_ranking as a safe fallback
        last_non_empty_final = final_ranking if final_ranking else None

        final = None
        for i in range(number):
            # If final_ranking is empty, break early and keep the last non-empty ranking
            if not final_ranking:
                # No further meaningful work can be done; break to avoid cascading empty LLM calls
                break

            nextlevel = self.BFS.question_agent(final_ranking)

            # Normalize nextlevel to a list of idea dicts if possible
            if isinstance(nextlevel, dict):
                # If it's an error dict or single object, try to extract reranked_ideas or ideas
                if nextlevel.get('error'):
                    # stop iteration on error
                    break
                elif 'reranked_ideas' in nextlevel:
                    normalized_next = nextlevel.get('reranked_ideas') or []
                else:
                    # wrap single dict into list
                    normalized_next = [nextlevel]
            elif isinstance(nextlevel, list):
                normalized_next = nextlevel
            else:
                normalized_next = []

            llm_score = self.Ranking.get_llm_score(prompt, normalized_next)
            embedding_scores = self.Ranking.add_embedding_scores(prompt, normalized_next)
            sorted_by_similarity = sorted(embedding_scores, key=lambda x: x.get('embedding_similarity', 0), reverse=True)
            final_ranking = self.Ranking.hybrid_rerank_run(llm_score, embedding_scores, sorted_by_similarity)

            if final_ranking:
                last_non_empty_final = final_ranking

            # Only keep the last value
            if i == number - 1:
                final = final_ranking

        # If final is empty, fallback to last_non_empty_final or initial scores
        if not final:
            if last_non_empty_final:
                final = last_non_empty_final
            else:
                # fallback to initial scores (if any)
                final = scores if scores else []

        print("This is final", final)
        research = self.DeepResearchAgent.run(prompt = final)
        return research
        




    class BFS:

        def __init__(self,ideas):

            self.ideas = {}
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key =  os.getenv("OPENAI_API_KEY"))
            self.Ranking = DeepSearchTool.Ranking(model=None, api_key=os.getenv("OPENAI_API_KEY"), query="")  

        def planner_agent(self,prompt):

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
            result = self.llm.invoke([
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

            scores = self.Ranking.get_llm_score(prompt, scenarios)
            sorted_by_similarity = sorted(scores, key=lambda x: x.get('similarity_score', 0), reverse=True)
            #print("Sorted by similarity", sorted_by_similarity)

            # Return all for downstream processing
            return scenarios, scores, sorted_by_similarity


        
        
        def question_agent(self, payload):
            """Agents responsible for questioning and validating every node. Accepts a payload (usually final_ranking list)."""
            # Ensure payload is normalized to a list
            if not payload:
                return []

            # If payload is a dict with an 'error' key, surface it so caller can decide
            if isinstance(payload, dict) and payload.get('error'):
                return payload

            api_key = os.getenv("OPENAI_API_KEY")
            llm = init_chat_model("gpt-4o-2024-08-06", temperature=0.0, model_provider="openai", api_key=api_key)

            system_prompt = system_planner_prompt
            try:
                user_prompt = json.dumps(payload, ensure_ascii=False)
            except TypeError:
                # fallback: coerce to string
                user_prompt = str(payload)

            result = llm.invoke([
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Rerank using these inputs:\n{user_prompt}"}
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
            try:
                output = json.loads(raw)
            except Exception:
                output = raw
            # Ensure a list is returned where possible
            if isinstance(output, dict) and 'reranked_ideas' in output:
                return output['reranked_ideas']
            if isinstance(output, dict) and not output:
                return []
            return output
        
    
        def main(self,prompt,level):

            if level == 1:
                scenarios, scores, sorted_scores = self.planner_agent(prompt)
            else:
                scenarios,scores,sorted_scores = self.question_agent(prompt)
            


    

    class Ranking:
        def __init__(self, model, api_key, query):
            
            self.query = query
            # (Embeddings use global OpenAI client)

        def get_embedding(self,text: str):
            """Turning words into embeddings"""
            # Use OpenAI v1 client
            resp = _openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return resp.data[0].embedding

        def compare_embeddings(self, query_embedding, idea_embedding):
            # Example: Euclidean distance (unchanged logic)
            return float(np.linalg.norm(np.array(query_embedding) - np.array(idea_embedding)))
        
        def cosine_similarity(self,a, b) -> float:
            """Cosine similarity between query and ideas/nodes"""
            a = np.array(a, dtype=float)
            b = np.array(b, dtype=float)
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            return float(np.dot(a, b) / denom)

        def add_embedding_scores(self,query: str, ideas) -> list[dict]:
            """Compute embedding-based similarity per idea and return list[{idea, embedding_similarity}]."""
            # Normalize ideas into a list of dicts
            normalized = []
            if not ideas:
                return []
            if isinstance(ideas, dict):
                # If wrapped in a structure with reranked_ideas
                if 'reranked_ideas' in ideas:
                    normalized = ideas.get('reranked_ideas') or []
                else:
                    normalized = [ideas]
            elif isinstance(ideas, list):
                normalized = ideas
            else:
                # Unknown type, try to coerce
                try:
                    normalized = list(ideas)
                except Exception:
                    return []

            query_embedding = self.get_embedding(query)
            results = []
            for item in normalized:
                if isinstance(item,dict):
                    idea_text = f"{item.get('idea','')}: {item.get('idea_description','')}"
                    # if idea_text empty, try other fields
                    if not idea_text.strip():
                        idea_text = json.dumps(item, ensure_ascii=False)
                    idea_embedding = self.get_embedding(idea_text)
                    sim = self.cosine_similarity(query_embedding, idea_embedding)
                    results.append({
                        'idea': item.get('idea','') or idea_text,
                        'embedding_similarity': sim
                    })
                else:
                    # fallback for non-dict entries
                    try:
                        idea_embedding = self.get_embedding(str(item))
                        sim = self.cosine_similarity(query_embedding, idea_embedding)
                        results.append({
                            'idea': str(item),
                            'embedding_similarity': sim
                        })
                    except Exception:
                        continue

            return results

        def hybrid_rerank_run(self,llm_scores: list, embedding_scores: list, sorted_by_similarity: list[dict]):
            """Re-rank of the ideas through a LLM"""
            # If all inputs are empty, return safe fallback
            if not llm_scores and not embedding_scores and not sorted_by_similarity:
                return []

            # Normalize llm_scores to list
            if isinstance(llm_scores, dict):
                llm_scores = [llm_scores]

            api_key = os.getenv("OPENAI_API_KEY")
            llm = init_chat_model("gpt-4o-2024-08-06", temperature=0.0, model_provider = "openai", api_key=api_key)

            payload = {
                'llm_scores': llm_scores,
                'embedding_scores': embedding_scores,
                'sorted_by_similarity': sorted_by_similarity
            }

            system = (
                "You are an AI Hybrid Reranker. Combine multiple signals to produce a final ordered list of ideas from most important to least.\n"
                "Reasoning requirements:\n"
                "- Do not justify with 'high similarity' signals alone; explain the true underlying meaning/causal or economic rationale behind the ranking.\n"
                "- Link drivers (macro regime, policy, flows, valuation, earnings, positioning) to each idea's merits/risks.\n"
                "- Use the signals as supporting evidence only.\n"
                "Guidelines:\n"
                "- Prefer higher LLM similarity_score and higher embedding_similarity when aligned with the substantive rationale.\n"
                "- Remove the single lowest-ranked idea from the final output.\n"
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

            try:
                ranked = json.loads(raw)
            except Exception:
                # If LLM did not return JSON, fallback to using llm_scores sorted by similarity_score
                ranked = llm_scores or embedding_scores or []

            # Enforce ordering (most-to-least) and drop the last (lowest-ranked) item
            try:
                if isinstance(ranked, list):
                    ranked = sorted(
                        ranked,
                        key=lambda x: x.get('final_score', x.get('similarity_score', 0) or x.get('embedding_similarity', 0)),
                        reverse=True
                    )
                    if len(ranked) > 0:
                        ranked = ranked[:-1]
            except Exception:
                pass

            return ranked



        def get_llm_score(self,query: str, idea_text) -> list[dict]:
            """
            A LLM Score focused on ranking the ideas with the query based on relevance
            """
            api_key = os.getenv("OPENAI_API_KEY")
            llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)

            # Normalize idea_text to JSON serializable structure
            try:
                ideas_json = json.dumps(idea_text, ensure_ascii=False)
            except Exception:
                # fallback for non-serializable input
                if isinstance(idea_text, dict):
                    ideas_json = json.dumps([idea_text], ensure_ascii=False)
                else:
                    ideas_json = json.dumps([], ensure_ascii=False)

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

            try:
                parsed = json.loads(raw)
            except Exception:
                # if the model returned a single object, wrap it
                try:
                    parsed_candidate = json.loads(f"[{raw}]")
                    parsed = parsed_candidate
                except Exception:
                    # final fallback: build a conservative score list from idea_text
                    if isinstance(idea_text, list):
                        parsed = []
                        for it in idea_text:
                            if isinstance(it, dict):
                                parsed.append({'idea': it.get('idea', str(it)), 'similarity_score': 0.0})
                    elif isinstance(idea_text, dict):
                        parsed = [{'idea': idea_text.get('idea', str(idea_text)), 'similarity_score': 0.0}]
                    else:
                        parsed = []

            # Ensure we return a list of dicts
            if isinstance(parsed, dict):
                parsed = [parsed]

            return parsed

        def reranking(self, ideas) -> dict:
            """
            Re Ranking of ideas based on confidence score
            """
            ideas = self.ideas

            sorted_ideas = dict(sorted(ideas.items(), key = lambda item : item[1]['confidence_score']))
            
            return sorted_ideas

        
        def get_vector_score(query : str, idea_text : dict) -> dict:
            return 0

        

    class DFS:
        def __init__(self,query,api_key, llm):
            self.query = query
            self.llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key =  os.getenv("OPENAI_API_KEY"))
            #self.tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        

        def node_search(self,query : str, idea_research : dict) -> dict:
            """
            Conducts a deep research on the node topic
            """
            prompt = f"""
            You are an AI Assistant responsible for conducting and summarizing research on {query}
            """
            agent = initialize_agent(
                tools = available_tools,
                llm = self.llm,
                agent = AgentType.OPENAI_FUNCTIONS,
                verbose = True,
                max_iteration = 2
            )
            result = agent.run("Conduct an indepth analysis of {idea_research} based on the tools provided")

            final_result = self.llm.invoke([
                {'role':'system', 'content': prompt},
                {'role':'user','content' : f"Summarize result"}
            ])
            return final_result
        
        
        
        def internet_search(self,
            query: str,
            max_results: int = 5,
            topic: Literal["general", "news", "finance"] = "general",
            include_raw_content: bool = False,
        ):
            """Run a web search"""
            return self.tavily_client.search(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic,
            )
        
        def run(self,node):
            research = self.node_search(node)




search = DeepSearchTool()
print(search.run(prompt = "Brazil Economy", number = 5))