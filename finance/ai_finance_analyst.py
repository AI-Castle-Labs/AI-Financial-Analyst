from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from prompt import system_source_prompt,system_macro_prompt,system_sector_research_analyst,system_central_bank_prompt,system_fx_research_prompt,system_agent_prompt,system_portfolio_manager_prompt
from schema import PlannerAgentSchema,SonarInput,MacroAnalystSchema,SectorAnalystSchema,PortfolioManagerSchema,GeneralAgentSchema,FXAgentSchema,CentralBankSchema
#from tools import FRED_Chart
import os
from dotenv import load_dotenv
from state import AgentState
import tweepy
from fpdf import FPDF
from extra_tools import chart_agent
import json

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
    

    def planner_agent(self,State):

        planner_llm = self.llm.with_structured_output(PlannerAgentSchema)
        prompt = State.prompt
        result = planner_llm.invoke([
            {'role':'system', 'content': State.prompt},
            {'role':'user','content':f"Provide a plan for each agent based on {prompt}"}
        ])
        print ("This is the result",result)
        State.title = result.title
        State.macro_agent_task = result.macro_agent
        State.sector_analyst_agent_task = result.sector_agent
        State.central_bank_agent_task = result.central_bank_agent
        State.fx_research_agent_task = result.fx_research_agent

        return State



    def macro_analyst_agent(self,State):
        llm = self.llm.with_structured_output(MacroAnalystSchema)
        prompt = State.macro_agent_task
        result = llm.invoke([
            {'role':'system','content' : system_macro_prompt},
            {'role':'user', 'content': f"Provide an macro investment pitch for {prompt}"}
        ])
        # Store output in dedicated attribute (list form for AgentState)
        State.macro_analyst_agent = [result.pitch]
        # Only update State.prompt if result.pitch is a string
        if isinstance(result.pitch, str):
            State.prompt = result.pitch
        # Store agent description for later use
        State.agent_description  = getattr(result, 'created_agent_description', None)
        return State

    def sector_analyst_agent(self,State):
        prompt = State.sector_analyst_agent_task
        llm = self.llm.with_structured_output(SectorAnalystSchema)
        result = llm.invoke([
            {'role':'system','content' : system_sector_research_analyst},
            {'role':'user','content' : f"Provide sector research analysis for {prompt}"}
        ])
        # Store output in dedicated attribute (list form)
        State.sector_analyst_agent = [result.research]
        return State

    def central_bank_agent(self,State):
        llm = self.llm.with_structured_output(CentralBankSchema)
        system_prompt = system_central_bank_prompt
        prompt = State.central_bank_agent_task
        result = llm.invoke([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Provide analysis as a central bank analyst for {prompt}"}
        ])
        # Store as list to satisfy AgentState typing
        State.central_bank_agent = [result.research]
        return State

    def fx_research_agent(self, State, prompt = None):      
        prompt = State.fx_research_agent_task

        llm = self.llm.with_structured_output(FXAgentSchema)
        system_prompt =  system_fx_research_prompt
        result = llm.invoke([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Provide FX research analysis on {prompt}"}
        ])
        State.fx_research_agent = [result.research]
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
        State.agent = [result.research]
        return State

    def portfolio_manager(self,State):
        prompt = State.prompt
        llm = self.llm.with_structured_output(PortfolioManagerSchema)
        # Use correct state attribute names (those ending with _agent)
        system_prompt = system_portfolio_manager_prompt.format(
            macro_agent = getattr(State, 'macro_analyst_agent', None),
            sector_agent = getattr(State, 'sector_analyst_agent', None),
            fx_agent = getattr(State, 'fx_research_agent', None),
            central_bank_agent = getattr(State, 'central_bank_agent', None),
            agent = getattr(State, 'agent', None)
        )
        result = llm.invoke([
            {'role':'system','content':system_prompt},
            {'role': 'user', 'content': f'Provide analysis based on {prompt}'}
        ])
        State.portfolio_manager_agent = [result.portfolio_summary]
        return State

    def _format_state_value(self, val):
        """Normalize a State attribute into a printable string for the PDF without Python-style brackets.
        - lists -> join items separated by double newline
        - dicts -> pretty-printed JSON without surrounding brackets
        - primitives -> str()
        """
        try:
            if val is None:
                return ""
            # If this is already a list (e.g., [result.research]), join elements
            if isinstance(val, list):
                parts = []
                for v in val:
                    if isinstance(v, dict):
                        parts.append(json.dumps(v, ensure_ascii=False, indent=2))
                    else:
                        parts.append(str(v))
                # join without any bracket chars
                return "\n\n".join(parts)
            if isinstance(val, dict):
                # Pretty print dict content
                return json.dumps(val, ensure_ascii=False, indent=2)
            return str(val)
        except Exception:
            return str(val)

    def create_pdf_report(self, results, filename="investment_report.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt=results.get('Investment Pitch', 'No Title'), ln=True, align='C')
        pdf.ln(6)

        # List of agent fields to include in the PDF, in order
        agent_fields = [
            ('macro_analyst_agent', 'Macro Analyst'),
            ('sector_analyst_agent', 'Sector Analyst'),
            ('central_bank_agent', 'Central Bank Analyst'),
            ('fx_research_agent', 'FX Research Analyst'),
            ('agent', 'General Agent'),
            ('portfolio_manager_agent', 'Portfolio Manager'),
        ]

        # Print each agent's output, even if None (show 'No output' if missing)
        for attr, label in agent_fields:
            content = results.get(label)
            if not content:
                content = "No output."
            pdf.set_font("Arial", 'B', 13)
            pdf.cell(0, 8, txt=label, ln=True)
            pdf.ln(2)
            pdf.set_font("Arial", '', 11)
            for paragraph in str(content).split('\n\n'):
                for line in paragraph.split('\n'):
                    line = line.strip()
                    pdf.multi_cell(0, 7, txt=line)
                pdf.ln(3)
            pdf.set_line_width(0.2)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(6)

        pdf.output(filename)
        print(f"PDF report saved as {filename}")

    def run(self, prompt):
        workflow = StateGraph(AgentState)

        workflow.add_node("planner",self.planner_agent)
        workflow.add_node("macro_analyst", self.macro_analyst_agent)
        workflow.add_node("sector_analyst", self.sector_analyst_agent)
        workflow.add_node("central_bank_analyst", self.central_bank_agent)
        workflow.add_node("fx_research_analyst", self.fx_research_agent)
        workflow.add_node("portfolio_manager", self.portfolio_manager)
        # Node name 'agent' conflicts with state key 'agent'; use different node id
        workflow.add_node("general_agent", self.agent)
        # Edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner","macro_analyst")
        workflow.add_edge("macro_analyst","sector_analyst")
        workflow.add_edge("sector_analyst", "central_bank_analyst")
        workflow.add_edge("central_bank_analyst", "fx_research_analyst")
        workflow.add_edge("fx_research_analyst", "general_agent")
        workflow.add_edge("general_agent","portfolio_manager")
        workflow.add_edge("portfolio_manager", END)
        # Compile and run the workflow
        app = workflow.compile()

        initial_state = AgentState(
            prompt = prompt
        )
        result = app.invoke(input = initial_state)
        print("This is result",result)
        # Collect results for PDF
        results = {}
        # First, Investment Pitch (original prompt)
        results['Investment Pitch'] = result['title']

        # Known field name variants to pick from State (order preserved)
        fields = [
            ('macro_analyst', 'Macro Analyst'),
            ('macro_analyst_agent', 'Macro Analyst'),
            ('sector_analyst', 'Sector Analyst'),
            ('sector_analyst_agent', 'Sector Analyst'),
            ('central_bank_analyst', 'Central Bank Analyst'),
            ('central_bank_agent', 'Central Bank Analyst'),
            ('fx_research_analyst', 'FX Research Analyst'),
            ('fx_research_agent', 'FX Research Analyst'),
            ('agent', 'General Agent'),
            ('portfolio_manager', 'Portfolio Manager'),
            ('portfolio_manager_agent', 'Portfolio Manager'),
        ]

        for valuea, valueb in fields:
            if valuea in result:
                results[valueb] = result[valuea]
              

        self.create_pdf_report(results)




deepsearch = DeepResearchAgent()