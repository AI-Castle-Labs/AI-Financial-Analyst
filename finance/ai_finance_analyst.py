from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from prompt import system_source_prompt,system_macro_prompt,system_sector_research_analyst,system_central_bank_prompt,system_fx_research_prompt,system_agent_prompt,system_portfolio_manager_prompt
from schema import SonarInput,MacroAnalystSchema,SectorAnalystSchema,PortfolioManagerSchema,GeneralAgentSchema,FXAgentSchema,CentralBankSchema
#from tools import FRED_Chart
import os
from dotenv import load_dotenv
from state import AgentState
import tweepy
from fpdf import FPDF
from extra_tools import chart_agent

load_dotenv()


available_tools = [

    StructuredTool.from_function(

        name = "Chart_Tool", #Cannot have underscore because the LLM is particular on syntax

        func = chart_agent,

        schema = SonarInput,

        description="Performs a BHP test on different zones"

    ),

]



class DeepResearchAgent:


    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)

        self.llm = llm
        self.api_key = api_key
        self.report = {}




    def macro_analyst_agent(self,State):
        llm = self.llm.with_structured_output(MacroAnalystSchema)
        prompt = State.prompt
        result = llm.invoke([
            {'role':'system','content' : system_macro_prompt},
            {'role':'user', 'content': f"Provide an macro investment pitch for {prompt}"}
        ])
        # Store output in dedicated attribute
        State.macro_analyst = result.pitch
        # Only update State.prompt if result.pitch is a string
        if isinstance(result.pitch, str):
            State.prompt = result.pitch
        # Store agent description for later use
        State.agent_description  = getattr(result, 'created_agent_description', None)
        return State

    def sector_analyst_agent(self,State):
        prompt = State.prompt
        llm = self.llm.with_structured_output(SectorAnalystSchema)
        result = llm.invoke([
            {'role':'system','content' : system_sector_research_analyst},
            {'role':'user','content' : f"Provide sector research analysis for {prompt}"}
        ])
        # Store output in dedicated attribute
        State.sector_analyst = result.research
        return State

    def central_bank_agent(self,State):
        llm = self.llm.with_structured_output(CentralBankSchema)
        system_prompt = system_central_bank_prompt
        prompt = State.prompt
        result = llm.invoke([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Provide analysis as a central bank analyst for {prompt}"}
        ])
        State.central_bank_analyst = result.research
        return State

    def fx_research_agent(self, State, prompt = None):
        prompt = State.prompt
        llm = self.llm.with_structured_output(FXAgentSchema)
        system_prompt =  system_fx_research_prompt
        result = llm.invoke([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Provide FX research analysis on {prompt}"}
        ])
        State.fx_research_agent = result.research
        return State

    def agent(self,State):
        prompt = State.prompt
        llm = self.llm.with_structured_output(GeneralAgentSchema)
        system_prompt = system_agent_prompt.format(
            role = getattr(State, 'agent_description', ''),
            context = prompt
        )
        result = llm.invoke([
            {'role':'system', 'content': system_prompt},
            {'role':'user', 'content': f'Provide analysis based on {prompt}'}
        ])
        State.agent = result.research
        print("This is the state", State)
        return State

    def portfolio_manager(self,State):
        prompt = State.prompt
        llm = self.llm.with_structured_output(PortfolioManagerSchema)
        system_prompt = system_portfolio_manager_prompt.format(
            macro_agent = getattr(State, 'macro_analyst', None),
            sector_agent = getattr(State, 'sector_analyst', None),
            fx_agent = getattr(State, 'fx_research_analyst', None),
            central_bank_agent = getattr(State, 'central_bank_analyst', None),
            agent = getattr(State, 'agent', None)
        )
        result = llm.invoke([
            {'role':'system','content':system_prompt},
            {'role': 'user', 'content': f'Provide analysis based on {prompt}'}
        ])
        State.portfolio_manager = result.portfolio_summary
        return State
        
    def create_pdf_report(self, results, filename="investment_report.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Agentic AI Research Report", ln=True, align='C')
        pdf.ln(10)
        for agent_name, result in results.items():
            pdf.set_font("Arial", style='B', size=12)
            pdf.cell(200, 10, txt=f"{agent_name}:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=str(result))
            pdf.ln(5)
        pdf.output(filename)
        print(f"PDF report saved as {filename}")

    def run(self, prompt):
        
        workflow = StateGraph(AgentState)
        workflow.add_node("macro_analyst", self.macro_analyst_agent)
        workflow.add_node("sector_analyst", self.sector_analyst_agent)
        workflow.add_node("central_bank_analyst", self.central_bank_agent)
        workflow.add_node("fx_research_analyst", self.fx_research_agent)
        workflow.add_node("portfolio_manager", self.portfolio_manager)
        workflow.add_node("agent", self.agent)
        # Edges
        workflow.add_edge(START, "macro_analyst")
        workflow.add_edge("macro_analyst","sector_analyst")
        workflow.add_edge("sector_analyst", "central_bank_analyst")
        workflow.add_edge("central_bank_analyst", "fx_research_analyst")
        workflow.add_edge("fx_research_analyst", "agent")
        workflow.add_edge("agent","portfolio_manager")

        workflow.add_edge("portfolio_manager", END)
        # Compile and run the workflow
        app = workflow.compile()
        for item in prompt:
            print("This is item", item)
       
        initial_state = AgentState(
            prompt = prompt
        )
        result = app.invoke(input = initial_state)
        # Collect results for PDF
        results = {}
        if hasattr(result, 'macro_analyst'): results['Macro Analyst'] = getattr(result, 'macro_analyst')
        if hasattr(result, 'sector_analyst'): results['Sector Analyst'] = getattr(result, 'sector_analyst')
        if hasattr(result, 'central_bank_analyst'): results['Central Bank Analyst'] = getattr(result, 'central_bank_analyst')
        if hasattr(result, 'fx_research_analyst'): results['FX Research Analyst'] = getattr(result, 'fx_research_analyst')
        if hasattr(result, 'portfolio_manager'): results['Portfolio Manager'] = getattr(result, 'portfolio_manager')
        self.create_pdf_report(results)




deepsearch = DeepResearchAgent()
