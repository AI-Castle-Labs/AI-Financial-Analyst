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




class OutputSchema(BaseModel):
    """Anaylze the output for the syntesis agent"""
    research: str = Field (
        description = "Research on what is found on the specific subject"
    )
    summary : str = Field(
        description = "A summary of the research done by the agent on the subject"
    )
    confidence_score : float = Field(
        description = "Output score on based on confidence of the response"
    )
    

class Citation(BaseModel):
    """The schema for citation of the sources"""
    title:str = Field(
        description = "Title of the source"
    )
    url:str = Field(
        description = "Link to the source"
    )
    source:str = Field(
        description= "Name of the source"
    )

class PlannerSchema(BaseModel):
    """Output schema for the planning agent"""
    role: str = Field(
        ...,  # mark as required
        description="Role of the agent, etc Planner Agent, Memory Agent, Web-search agent etc"
    )
    steps: Dict[str, str] = Field(
        ...,  # mark as required
        description="Step-by-step process on what the agent should be following for the task. "
                    "The name of the agent should be the key and its role the value"
    )
    tools_required: str = Field(
        ...,  # mark as required
        description="Mention the tools required from the available set of tools for each agent to use"
    )



class Memorystate(BaseModel):
    """The schema for memory"""
    query : str = Field(
        description= "The query the user has asked"
    )
    steps: str = Field(
        description= "The steps the agent has to follow"
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

#parser = PydanticOutputParser()



