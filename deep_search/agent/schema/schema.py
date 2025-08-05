from pydantic import BaseModel, Field
from typing import Literal,Optional,Dict
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langgraph.graph import MessagesState
from typing import List


class ClassificationSchema(BaseModel):
    """Analyze the message and provide a classification of the message"""

    reasoning : str = Field(
        description= "Step-by-steap reasoning behind the classification"
    )
    classification : Literal["summarize","citetool","BulletTool"] = Field(
        description="The classification of an email: 'summarize' for summarizing  document, "
        "'citetool' for citing information of the document, "
        "'BulletTool' for queries that need a bullet point output",
    )
    '''
    Plan below is to give a plan on what the next AI agent should do based on the classification
    if summarize,
    should be detailed steps on what should be summarized giving importance to messages
    
    if citetool,
    steps on where it should find the information

    if bullettool:
    steps on bulleted answers to take
    '''
    Plan: str= Field(
        description = "There should be a detailed plan based on the classification"
    )

class BulletPointOutput(BaseModel):
    text:str = Field(...,description = "The content of the bullet point")
    confidence: Optional[float] = Field(None, description= "Confidence score between (0-1) on every bullet point")
