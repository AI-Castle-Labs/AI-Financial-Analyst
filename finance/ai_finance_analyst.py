from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from prompt import system_source_prompt,system_macro_prompt,system_sector_research_analyst,system_central_bank_prompt,system_fx_research_prompt,system_agent_prompt
from schema import Classification_outputSchema,MacroAnalystSchema,SectorAnalystSchema
#from tools import FRED_Chart
import os
from dotenv import load_dotenv
from state import AgentState
import tweepy
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
    return result.created_agent_description

print(macro_analyst_agent())

def sector_analyst_agent(prompt):
    llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)

    llm = llm.with_structured_output(SectorAnalystSchema)

    result = llm.invoke([
        {'role':'system','content' : system_sector_research_analyst},
        {'role':'user','content' : f"Provide sector research analysis for {prompt}"}
    
    ])
    return result


def central_bank_agent(prompt):
    llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key) 

    system_prompt = system_central_bank_prompt

    result = llm.invoke([
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Provide analysis as a central bank analyst for {prompt}"}
    ])
    return result

def fx_research_agent():
    llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)

    system_prompt =  system_fx_research_prompt

    result = llm.invoke([
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': "Provide FX research analysis on {prompt}"}
    ])
    return result

def agent(State):
    llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)

    system_prompt = system_agent_prompt.format(
        role = State.role
    )

    result = llm.invoke([
        {'role':'system', 'content': system_prompt},
        {'role':'user', 'content': 'Provide analysis based on given information'}
    ])

    return result

def summarizer_agent():
    llm = init_chat_model("gpt-4o-2024-08-06", temperature = 0.0, model_provider = "openai",api_key = api_key)
    system_prompt = system_summarizer_prompt.format(

    )


workflow = StateGraph(AgentState)
workflow.add_node("macro_analyst", macro_analyst_agent)
workflow.add_node("sector_analyst", sector_analyst_agent)
workflow.add_node("central_bank_analyst", central_bank_agent)
workflow.add_node("fx_research_analyst", fx_research_agent)

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
result = app.invoke({"input": "Your initial prompt here"})
print(result)



api_key = os.getenv("X_API_KEY")
api_key_secret = os.getenv("X_API_KEY_SECRET")
access_token = os.getenv("X_ACCESS_TOKEN")
access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("token")

client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_key_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)
