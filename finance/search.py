from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from prompt import system_source_prompt,system_macro_prompt,system_sector_research_analyst,system_central_bank_prompt,system_fx_research_prompt,system_agent_prompt,system_portfolio_manager_prompt
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
import openai
import numpy as np


load_dotenv()


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
    

    def main(self,model,api_key,prompt,number) -> str:

        final_number = number
        final_ideas = ""
        
        for i in range(number):
            response = self.BFS.planner_agent(final_ideas,prompt,final_number)
            final_ideas = self.Ranking.hybrid_rerank_run(response)
            final_number = final_number - 1
            



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
        
        def main(self,prompt):

            response = self.planner_agent(prompt)

    


    class Ranking:
        def __init__(self, model, api_key, query):
            
            self.query = query
            self.api_key = api_key
            openai.api_key = api_key

        def get_embedding(self, text):
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']

        def compare_embeddings(self, query_embedding, idea_embedding):
            # Example: Euclidean distance
            return float(np.linalg.norm(np.array(query_embedding) - np.array(idea_embedding)))

        def add_embedding_difference(self, ideas):
            query_embedding = self.get_embedding(self.query)
            for key, value in ideas.items():
                idea_text = value['idea_text']
                idea_embedding = self.get_embedding(idea_text)
                difference = self.compare_embeddings(query_embedding, idea_embedding)
                value['difference'] = difference
            return ideas
        



        def get_llm_score(self, query : str, idea_text : str) -> dict:
            """
            A LLM Score focused on ranking the ideas with the query based on relevance
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
        def __init(self,query,api_key, llm):
            self.query = query
            self.llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)
        

        def node_search(self,query : str, idea_research : dict) -> dict:
            """
            Conducts a research on the node topic
            """
            prompt = f"""
            You are an AI Assistant responsible for conducting summarizing research on {query}
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



