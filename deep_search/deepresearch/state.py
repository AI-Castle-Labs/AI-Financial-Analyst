from typing import TypedDict, List, Optional
from typing import Literal,Optional,Dict
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langgraph.graph import MessagesState
from typing import List
from langchain.output_parsers import PydanticOutputParser
from serpapi import GoogleSearch
import os
from pydantic import BaseModel, Field
from typing import List, Optional
from tools import long_term_memory



class Plannerstate(BaseModel):
    query : str = Field(
        description= "The query the user has asked"
    )
    steps : str = Field(
        description= "The steps the agent has to follow"
    )
    tools_provided : str = Field(
        description= "The set of tools to be used"
    )
    citations : str = Field(
        description= "The link to the source of the information"
    )

class Memorystate(BaseModel):
    """The schema for memory"""
    query : str = Field(
        description= "The query the user has asked"
    )
    steps: str = Field(
        description= "The steps the agent has to follow"
    )
    tools_provided : str = Field(
        description= "The set of tools to be used"
    )
    citations : str = Field(
        description = "The link to the source of the information"
    )


class WebSearchState(BaseModel):
    """The schema for web-search"""
    query : str = Field(
        description= "The query the user has asked"
    )
    steps: str = Field(
        description =  "The steps the agent has to follow"
    )
    tools_provided : str= Field(
        description= "The tools the agent can use"
    )
    citations : str = Field(
        description= "The link to the source of the information"
    )

class OtherState(BaseModel):
    """Schema for other agents"""
    role : str =Field(
        description = "The role assigned for the task"
    )

    steps: str = Field(
        description= "The steps the agent has to follow"
    )
    tools_provided : str = Field(
        description= "The set of tools to be used"
    )
    citations : str = Field(
        description= "The link to the source of the information"
    )
class SynthesizerState(BaseModel):
    memory_research : str = None
    other_agent_1_research : str = None
    other_agent_2_research : str = None

#parser = PydanticOutputParser()


class MasterState(BaseModel):
    planner: Plannerstate
    Memory : Memorystate
    Websearch : WebSearchState
    OtherState : OtherState



