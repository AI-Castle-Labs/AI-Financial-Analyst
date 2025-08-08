from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from prompt import system_source_prompt,system_macro_prompt,system_sector_research_analyst,system_central_bank_prompt,system_fx_research_prompt,system_agent_prompt,system_portfolio_manager_prompt
from schema import PlanningSchema,LLMScore
import os
from dotenv import load_dotenv
from state import AgentState
import tweepy
from fpdf import FPDF
from finance.extra_tools import chart_agent





"""
Build a deep search method using BFS AND DFS
BFS
    -Planner Agent gives plan and agent role to 5 different nodes
    - Each node gives out their research and cite
    - 5 -> 4 Do deepth search, 4-> 3
DFS



"""

class DeepSearchTool:

    def __init__(self, model, api_key):

        self.model = model
        self.api_key = api_key


    class BFS:

        def __init__(self,ideas):

            self.ideas = {}


        def planner_agent(self,prompt):
            prompt = f"""
            You are an AI Planning Agent who is responsible for planning different roles and description for different sub-agents.Based on the given
            {prompt}
            """
            llm = self.llm.with_structured_output(PlanningSchema)
            result = llm.invoke([
                {'role':'system','content' : prompt},
                {'role':'user','content' : f"Provide sector research analysis for {prompt}"}
            
            ])
            for key,values in result.research_ideas.items():
                self.ideas[key] = values

            return result
    


    class Ranking:
        def __init__(self,model, api_key, query):
            
            self.query = query
            self.api_key = api_key


        def get_llm_score(self, query : str, idea_text : str) -> dict:
            """
            A LLM Score focused on ranking the ideas with the query
            """

            llm = llm.with_structured_output(LLMScore)

            prompt = "Conduct a similarity score between the {query} and {idea_text}"

            result = llm.invoke([
                {'role':'system', 'content': prompt},
                {'role':'user','content' : f"Compare the prompt and ideas and focus on how relevant it is to the required prompt"}
            ])

            return result



        def reranking(self, ideas) -> dict:
            """
            Re Ranking of ideas based on confidence score
            """
            ideas = self.ideas

            sorted_ideas = dict(sorted(ideas.items(), key = lambda item : item[1]['confidence_score']))

        
        def get_vector_score(query : str, idea_text : str) -> dict:
            return 0

        

        def hybrid_rerank_run():
            return 0
    
    class DFS:
        def __init(self,query):
            self.query = query
        

        def node_search(query : str, idea_research : dict) -> dict:
            """
            Conducts a research on the node topic
            """

