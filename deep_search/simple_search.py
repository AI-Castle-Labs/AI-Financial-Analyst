from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from agent.schema.schema import ClassificationSchema
from agent.simple_agent import system_classification_prompt,system_cite_prompt,system_bullet_prompt
from state.state import currentstate
from tools.tools import pdf_extractor
import os

"""
Tools for simple search
1)SearchTool
2)SummarizeTool
3)AnswerQuestionTool
4)CiteTool
5)BulletNoteTool

"""



llm = init_chat_model("gpt-4",temperature = 0.0, model_provider = "openai",api_key = api_key)

def questionanswertool(state):
    reasoning = state.reasoning
    document = state.docuemnt
    plan = state.plan





def bullet_note_tool(document,state,reasoning,plan):
    context = state.context
    system_prompt = system_bullet_prompt(
        information = document,
        Plan = plan,
        Reasoning = reasoning
    )
    llm.with_structured_output()




def cite_tool(document, state,reasoning): #decide whether this is a tool or agent
    document = state.document
    system_prompt  = system_cite_prompt(
        information = document,
        document = state.document
    )

    


def classification_agent(state):
    document = state.document
    memory = state.memory
    system_prompt = system_classification_prompt(
        information = document,
        memory = memory
    )
    user_prompt = user_prompt

    response = llm.with_structured_output(ClassificationSchema)
    print(response)



def start_node(state):
    state.document = pdf_extractor()
    return state



    





workflow = StateGraph(currentstate)
workflow.add_node("start_node",start_node)
workflow.add_node("classification_agent", classification_agent)


workflow.set_entry_point("start_node")
workflow.add_edge("start_node","classification_agent")
workflow.add_edge("classification_agent",END)


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