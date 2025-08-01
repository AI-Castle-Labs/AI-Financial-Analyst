from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from prompt import system_source_prompt,system_macro_prompt,system_sector_research_analyst,system_central_bank_prompt
from schema import Classification_outputSchema,MacroAnalystSchema,SectorAnalystSchema
#from tools import FRED_Chart
import os
from dotenv import load_dotenv
from state import AgentState
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)

def search_agent(prompt):

    llm = llm.with_structured_output(Classification_outputSchema)
    system_prompt = system_source_prompt

    result = llm.invoke([
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"You are a macro analyst and you are analyzing {prompt} for a firm provide 2 data points, be very specific"}
    ])
    print(result)
    output_structure = {'data_source':result.datasource, 'name_point':result.name_point, 'data_point': result.data_point}

    print(FRED_Chart(result.name_point,result.data_point))


def macro_analyst_agent():
    llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)


    llm = llm.with_structured_output(MacroAnalystSchema)
    
    result = llm.invoke([
        {'role':'system','content' : system_macro_prompt},
        {'role':'user', 'content': f"Provide an macro investment pitch"}
    ])
    return result


def sector_analyst_agent(prompt):
    llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)

    llm = llm.with_structured_output(SectorAnalystSchema)

    result = llm.invoke([
        {'role':'system','content' : system_sector_research_analyst},
        {'role':'user','content' : f"Provide sector research analysis for {prompt}"}
    
    ])
    return result

print(sector_analyst_agent(prompt = "Investigate the performance of the US manufacturing and services sectors compared to the Eurozone, " \
"focusing on PMI data and recent earnings reports to gauge economic momentum."))

def central_bank_agent(prompt):
    llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key) 

    system_prompt = system_central_bank_prompt

    result = llm.invoke([
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Provide analysis as a central bank analyst for {prompt}"}
    ])
    return result




workflow = StateGraph(AgentState)


workflow.add_node("macro_analyst", macro_analyst_agent)
workflow.add_node("sector_analyst",sector_analyst_agent)
workflow.add_node("central_bank_analyst", central_bank_agent)





