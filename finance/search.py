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

    def run(self,prompt,number) -> str:

        final_number = number
        final_ideas = ""
        
        scenarios, scores, sorted_scores = self.BFS.planner_agent(prompt)
        embedding_scores = self.Ranking.add_embedding_scores(prompt, scenarios)
        final_ranking = self.Ranking.hybrid_rerank_run(scores, embedding_scores, sorted_scores)

        
        final = None
        for i in range(number):
            nextlevel = self.BFS.question_agent(final_ranking)
            llm_score = self.Ranking.get_llm_score(prompt, nextlevel)
            embedding_scores = self.Ranking.add_embedding_scores(prompt, nextlevel)
            sorted_by_similarity = sorted(embedding_scores, key=lambda x: x.get('similarity_score', 0), reverse=True)
            final_ranking = self.Ranking.hybrid_rerank_run(llm_score, embedding_scores, sorted_by_similarity)
            # Only keep the last value
            if i == 2:
                final = final_ranking

        print("This is final", final)
        
        




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


        def question_agent(payload):
            """Agents responsible for questioning and validating every node. Accepts a payload (usually final_ranking list)."""
            api_key = os.getenv("OPENAI_API_KEY")
            llm = init_chat_model("gpt-4o-2024-08-06", temperature=0.0, model_provider="openai", api_key=api_key)

            system_prompt = system_planner_prompt
            user_prompt = json.dumps(payload, ensure_ascii=False)

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
            #print("Output coming from question_agent:", output)

            return output
        
        def question_agent(self,payload):
            """Agents responsible for questioning and validating every node. Accepts a payload (usually final_ranking list)."""
            api_key = os.getenv("OPENAI_API_KEY")
            llm = init_chat_model("gpt-4o-2024-08-06", temperature=0.0, model_provider="openai", api_key=api_key)

            system_prompt = system_planner_prompt
            user_prompt = json.dumps(payload, ensure_ascii=False)

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
            #print("Output coming from question_agent:", output)

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

        def add_embedding_scores(self,query: str, ideas: list[dict]) -> list[dict]:
            """Compute embedding-based similarity per idea and return list[{idea, embedding_similarity}]."""
            query_embedding = self.get_embedding(query)
            results = []
            for item in ideas:
                if isinstance(item,dict):
                    idea_text = f"{item.get('idea','')}: {item.get('idea_description','')}"
                    idea_embedding = self.get_embedding(idea_text)
                    sim = self.cosine_similarity(query_embedding, idea_embedding)
                    results.append({
                        'idea': item.get('idea',''),
                        'embedding_similarity': sim
                    })
                else:
                    for action in ideas.get('reranked_ideas', []):
                        next_action = action.get('next_action')
                        idea_embedding = self.get_embedding(str(next_action))
                        sim = self.cosine_similarity(query_embedding, idea_embedding)
                        results.append({
                            'idea': str(next_action),
                            'embedding_similarity': sim
                        })

            return results

        def hybrid_rerank_run(self,llm_scores: list[dict], embedding_scores: list[dict], sorted_by_similarity: list[dict]):
            """Re-rank of the ideas through a LLM"""
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

            #print("LLM SCORE", result)
            return result

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
print(search.run(prompt = "Long Brazil", number = "5"))