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
from finance.extra_tools import chart_agent
from extra_tools import ask_sonar
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain, SimpleSequentialChain
from openai import OpenAI
import numpy as np
import json
from typing import Literal

from tavily import TavilyClient
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

    ),

    StructuredTool.from_function(
        name = "Tavily_Search",


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

    def __init__(self, model, api_key, llm):

        self.model = model
        self.api_key = api_key
        self.llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)
        self.graph = {}
    

    def run(self,model,api_key,prompt,number) -> str:

        final_number = number
        final_ideas = ""
        
        scenarios, scores, sorted_scores = self.BFS.planner_agent(prompt)
        embedding_scores = self.Ranking.add_embedding_scores(prompt, scenarios)
        final_ranking = self.Ranking.hybrid_rerank_run(scores, embedding_scores, sorted_scores)
        
        for i in range(number - 1):
            scenarios, scores, sorted_scores = self.BFS.question_agent(prompt)
            embedding_scores = self.Ranking.add_embedding_scores(prompt, scenarios)
            final_ranking = self.Ranking.hybrid_rerank_run(scores, embedding_scores, sorted_scores)

            prompt = final_ranking
        
        




    class BFS:

        def __init__(self,ideas):

            self.ideas = {}
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)


        def planner_agent(self,prompt,level):
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
                {'role':'system','content' : system_prompt},
                {'role':'user','content' : f"Provide sector research analysis for {prompt}"}
                
            ])
            raw = _strip_code_fences(result.content)
            scenarios = json.loads(raw)

            return scenarios

        def question_agent(self,prompt):
            system_prompt = system_planner_prompt
            result = self.llm.invoke([
                {'role':'system','content' : system_prompt},
                {'role':'user','content' : f"Provide analysis for {prompt}"}
                
            ])

        def main(self,prompt,level):

            if level == 1:
                scenarios, scores, sorted_scores = self.planner_agent(prompt)
            else:
                scenarios,scores,sorted_scores = self.question_agent(prompt)
            

    


    class Ranking:
        def __init__(self, model, api_key, query):
            
            self.query = query
            self.api_key = api_key
            # (Embeddings use global OpenAI client)

        def get_embedding(self, text):
            # OpenAI v1 embeddings API
            resp = _openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return resp.data[0].embedding

        def compare_embeddings(self, query_embedding, idea_embedding):
            # Example: Euclidean distance (unchanged logic)
            return float(np.linalg.norm(np.array(query_embedding) - np.array(idea_embedding)))

        def add_embedding_difference(self, ideas):
            query_embedding = self.get_embedding(self.query)
            for key, value in ideas.items():
                idea_text = value['idea_text']
                idea_embedding = self.get_embedding(idea_text)
                difference = self.compare_embeddings(query_embedding, idea_embedding)
                value['difference'] = difference
            return ideas
        


        def get_llm_score(query : str, idea_text : json) -> dict:
    
            """
            A LLM Score focused on ranking the ideas with the query based on relevance
            """
            api_key = os.getenv("OPENAI_API_KEY")
            llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)

            # Format and instruct model for strict JSON, mirroring test structure
            ideas_json = json.dumps(idea_text, ensure_ascii=False)
            prompt = (
            f"Compare the following ideas (JSON):\n{ideas_json}\n\n"
            f"against the query: \"{query}\" and return a JSON array. You are a part of AI Investment Analyst where you are responsible for " \
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
                {'role':'system', 'content': prompt},
                {'role':'user','content' : f"Compare the prompt and ideas and focus on how relevant it is to the required prompt"}
            ])
            raw = _strip_code_fences(result.content)
            result = json.loads(raw)

            print(result)

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

        

        def hybrid_rerank_run(self,reranking,get_llm_score,add_embedding_difference):
            prompt = """
            You are an AI Hyrbid Reranker responsible for reranking based on given input, take alook at the rankings of each of them
            and give a final ranking
            """
            result = self.llm.invoke([
                {'role':'system', 'content': prompt},
                {'role':'user','content' : f"Analyze results from {reranking}, {get_llm_score}, {add_embedding_difference}"}
            ])
            return result
    
    class DFS:
        def __init__(self,query,api_key, llm):
            self.query = query
            self.llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)
            self.tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        

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




