from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from schema import OutputSchema,PlannerSchema,Memorystate,WebSearchState
from prompt import system_planning_prompt,user_planning_prompt,system_memory_prompt
from langchain.tools import Tool
from tools import long_term_memory,use_serp_tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI  # or whatever model wrapper you're using
from langchain.tools import Tool
from langchain_core.output_parsers import JsonOutputParser



api_key = "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"

llm = init_chat_model("gpt-4o-2024-08-06",temperature = 0.0, model_provider = "openai",api_key = api_key)


available_tools = [
    Tool(
        name="web_search",
        func=use_serp_tool,
        description="Performs a Google search when required."
    ),
    Tool(
        name="long_term_memory",
        func=long_term_memory,
        description="Accesses or stores data in Qdrant vector database for long-term memory."
    )
]


def use_serp(query):
    return (use_serp_tool(query))



def memory_agent():
    query = "Research on what is the prospect of spx in the future"
    steps = "Use the websearch tool and rewrite the query into 2 words"
    tools_provided = available_tools

    llm_router = llm.with_structured_output(Memorystate)
    
    system_prompt = system_memory_prompt.format(
        plan = steps,
        tools_provided = tools_provided
    )

    user_prompt = user_planning_prompt.format(
        query = query
    )
    agent_executor = initialize_agent(
        available_tools,
        llm = llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True,
        system_message=system_prompt,
        max_iteration = 1
    )
    result = agent_executor.run(user_prompt)
    structured_result = llm_router.invoke(result)
    print(structured_result)
     

class CompassAI:
    def __init__(
        self,
        query: str,
        report_type: str,
        query_domains: list[str] | None,
        report_format: str = None
    ):
        self.query = query
        self.report_type = report_type
        self.query_domains = query_domains
        self.report_format = report_format

    def research(self):
        return self.query
    
a = CompassAI(
    query = "name",
    report_type = "pdf",
    query_domains = ["a"],
    report_format= "pdf"
)
print(a.research())