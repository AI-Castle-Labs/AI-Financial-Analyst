from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from prompt import system_source_prompt,system_macro_prompt,system_sector_research_analyst,system_central_bank_prompt,system_fx_research_prompt,system_agent_prompt,system_portfolio_manager_prompt
from schema import Classification_outputSchema,MacroAnalystSchema,SectorAnalystSchema
#from tools import FRED_Chart
import os
from dotenv import load_dotenv
from state import AgentState
import tweepy
from fpdf import FPDF

load_dotenv()



class DeepResearchAgent:


    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)

        self.llm = llm
        self.api_key = api_key
        self.report = {}

    def chart_agent(self,prompt = None, State = None):

        llm = self.llm.with_structured_output(Classification_outputSchema)
        system_prompt = system_source_prompt

        result = self.llm.invoke([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"You are a macro analyst and you are analyzing {prompt} for a firm provide 2 data points, be very specific"}
        ])
        print(result)
        output_structure = {'data_source':result.datasource, 'name_point':result.name_point, 'data_point': result.data_point}

        print(FRED_Chart(result.name_point,result.data_point))


    def macro_analyst_agent(self,prompt = None, State = None):
        if (State is not None and State.prompt != None):
            prompt = State.prompt
        else:
            prompt = prompt 

        
        llm = self.llm.with_structured_output(MacroAnalystSchema)
        
        result = llm.invoke([
            {'role':'system','content' : system_macro_prompt},
            {'role':'user', 'content': f"Provide an macro investment pitch for {prompt}"}
        ])
        
        self.report['Macro Analyst'] = result.pitch
        
        State.prompt = result.pitch
        
        State.agent_description  = result.created_agent_description

        return State


    def sector_analyst_agent(self,prompt = None ,State = None):
        if (State is not None and State.prompt != None):
            prompt = State.prompt
        
        else:
            prompt = prompt

        llm = self.llm.with_structured_output(SectorAnalystSchema)

        prompt = State.prompt

        result = llm.invoke([
            {'role':'system','content' : system_sector_research_analyst},
            {'role':'user','content' : f"Provide sector research analysis for {prompt}"}
        
        ])
        State.sector_analyst_agent = result

        return State


    def central_bank_agent(self,prompt = None, State = None):

        llm = self.llm.with_structured_output(SectorAnalystSchema)

        system_prompt = system_central_bank_prompt

        result = llm.invoke([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Provide analysis as a central bank analyst for {prompt}"}
        ])
        return result

    def fx_research_agent(self,prompt = None, State = None):
        if (State.prompt != None):
            prompt = State.prompt
        else:
            prompt = prompt


        system_prompt =  system_fx_research_prompt

        result = self.llm.invoke([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': "Provide FX research analysis on {prompt}"}
        ])

        State.fx_research_agent = result

        return State

    def agent(self,prompt = None, State = None):
        if (State.prompt):
            prompt = State.prompt
        else:
            prompt = prompt
        

        system_prompt = system_agent_prompt.format(
            role = State.agent_description
        )

        result = self.llm.invoke([
            {'role':'system', 'content': system_prompt},
            {'role':'user', 'content': 'Provide analysis based on {prompt}'}
        ])

        return State

    def summarizer_agent(self,prompt = None, State = None):
        
        if State is not None and hasattr(State, "prompt") and State.prompt is not None:
            prompt = State.prompt
        else:
            prompt = prompt
        
        system_prompt = system_portfolio_manager_prompt.format(

            sector_agent = State.sector_agent,
            fx_agent = State.fx_agent,
            central_bank_agent = State.central_bank_agent,
            agent = State.agent
        )
        result = self.llm.invoke([
            {'role':'sytem','content':system_prompt},
            {'role': 'user', 'content': 'Provide analysis based on {prompt}'}
        ])
        return State
        
    def create_pdf_report(self, results, filename = "investment_report"):
        pdf =  FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size = 12)

        pdf.cell(200, 10, txt = "Agentic AI Research Report")
        pdf.ln(10)

        for agent_name, result in result.items():
            pdf.set_font("Arial", style ='B', size = 12)


    
    def run(self,prompt):

        workflow = StateGraph(AgentState)
        workflow.add_node("macro_analyst", self.macro_analyst_agent)
        workflow.add_node("sector_analyst", self.sector_analyst_agent)
        workflow.add_node("central_bank_analyst", self.central_bank_agent)
        workflow.add_node("fx_research_analyst", self.fx_research_agent)

        # Add edges
        workflow.add_edge(START, "macro_analyst")
        workflow.add_edge("macro_analyst", "sector_analyst")
        workflow.add_edge("macro_analyst", "central_bank_analyst")
        workflow.add_edge("macro_analyst", "fx_research_analyst")



        workflow.add_edge("sector_analyst", "summarizer_agent")
        workflow.add_edge("central_bank_analyst", "summarizer_agent")
        workflow.add_edge("fx_research_analyst", "summarizer_agent")
        workflow.add_edge("summarizer_agent", "summarizer_agent")


        # Compile and run the workflow
        app = workflow.compile()
        result = app.invoke({"input":prompt})
        print(result)




deepsearch = DeepResearchAgent()
deepsearch.run(prompt = "Conduct research on equity valuations across the US")