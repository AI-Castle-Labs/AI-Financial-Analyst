from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from prompt import system_source_prompt,system_macro_prompt,system_equity_analyst
from schema import Classification_outputSchema,MacroAnalystSchema
#from tools import FRED_Chart
import os
from dotenv import load_dotenv
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


def macro_analyst(prompt):
    llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)


    llm = llm.with_structured_output(MacroAnalystSchema)
    
    result = llm.invoke([
        {'role':'system','content' : system_macro_prompt},
        {'role':'user', 'content': f"Provide macro economic analysis for {prompt}"}
    ])
    return result.insight

def equity_analyst(prompt):
    lllm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)

    result = llm.invoke([
        {'role':'system','content' : system_equity_analyst},
        {'role':'user','content' : f"Provide equity research analysis for {prompt}"}
    
    ])



if __name__ == "__main__":
    macro_analyst("C&I Loans at Small Domestically Chartered Banks")
