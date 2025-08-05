from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from schema import OutputSchema,PlannerSchema,Memorystate,WebSearchState
from prompt import system_planning_prompt,user_planning_prompt,system_memory_prompt, system_agent_prompt
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from schema import OutputSchema,PlannerSchema,Memorystate,WebSearchState, OtherState
from langchain.tools import Tool
from tools import long_term_memory,use_serp_tool,retrieve_document
from langchain.agents import initialize_agent, AgentType
from state import SynthesizerState



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


class CompassAI:
    def __init__(
            self,
            query : str,
            report_type : str,
            query_domains : list[str] | None,
            report_format : str = None
    ):
        """Initialize a Compass Instance
        
        Args:
            query(str) : The research query or question
            report_type(str) : Type of research
            tone(Tone) : Tone of research
            report_format : Type of Report whether it is PDF,XCEL
        """
    
def start_research(State):
    return State
    
def planner_agent(query):

    
    State = "As a planning agent, in the past you have not been to the point, be to the point"
    
    system_prompt = system_planning_prompt.format(
        background = State 
    )

    user_prompt = user_planning_prompt.format(
        query = query
    )
    planner_schema_json = PlannerSchema.model_json_schema()
    llm_router = llm.with_structured_output(planner_schema_json)
    result = llm_router.invoke(
        [
            {'role':'system','content':system_prompt},
            {'role':'system','content':user_prompt},
        ]
    )
    result = result['steps']
    
    for agent_name, agent_plan in result.items():
         print(agent_name, agent_plan)
         if agent_name == "Memory Agent":

             memory_state = Memorystate(
                 query = query,
                 steps = agent_plan,
             )
             memory_agent(memory_state)
         elif agent_name == "Synthesizer Agent":
             synthesizer_state =SynthesizerState(
                 query = query,
                 steps = agent_plan
             )


    
planner_agent("What is global macro? I want a detailed analysis on what is it and why is it important go deep with the research")


def memory_agent():
    plan = "Do the research the way you want it"
    query = "Is there an S&P 500 Index Effect"

    llm_router = llm.with_structured_output(OutputSchema)
    retrieve_tool = Tool(
        name = "retrieve document",
        func = retrieve_document,
        description = "Retrieves a document from Qdrant",
        is_single_input = True
    )
    tools = [retrieve_tool]
    Agent = initialize_agent(
        tools = tools,
        llm = llm,
        verbose = True,
        max_iteration = 1,
        handle_parsing_errors = True,
    )
    system_prompt  = system_memory_prompt.format(
        plan = plan
    )
    user_prompt = query

    result = Agent.invoke(
        [
            {'role':'system','content':system_prompt},
            {'role':'user','content':user_prompt},
        ]
    )
    return result

def web_search_agent():
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

    return (structured_result.model_dump_json)



    
def other_agent(State):
    query = State.task
    task = State.task
    role = State.role
    plan = State.plan

    #LLM embeded with the output state
    llm = llm.with_structured_output(OtherState)

    #System-Prompt 
    system_prompt = system_agent_prompt.format(
        plan = plan,
        task = task,
        role = role
    )
    user_prompt = user_prompt
    result = llm.invoke(
        [
            {'role':'system','content':system_prompt},
            {'role':'user','content': user_prompt}
        ]
    )
    return result

def synthesizer_agent(State):
    plan = State.plan
    memory_research = State.memory_research
    other_agent_1 = State.other_research_1
    other_agent_2 = State.other_research_2







"""
workflow = StateGraph(OutputSchema)
workflow.add_node("start_node",start_node)
#workflow.add_node("classification_agent", classification_agent)


workflow.set_entry_point("start_node")
workflow.add_edge("start_node","classification_agent")
workflow.add_edge("classification_agent",END)


"""

"""


workflow.add_node("citetool", cite_tool)
workflow.add_node("bullet_node_tool",bullet_note_tool)
workflow.add_node("questionanswertool",questionanswertool)




workflow.add_conditional_edges(
    "classification",
    lambda state : "classification",
    {
        "summarize": "summarize",
        "bulletnode":"bullet_node_tool",
        "answerquestiontool":"Answerquestiontool",
        
    }
)

workflow.add_node("summarize","citetool")
workflow.add_node("bulletnode","citetool")
workflow.add_node("questionanswertool","citetool")
"""

app = workflow.compile()
initial_state = {}
final_state = app.invoke({})