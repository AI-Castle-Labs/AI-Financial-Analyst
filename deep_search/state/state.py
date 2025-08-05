from pydantic import BaseModel, Field
from typing import Literal,Optional,Dict
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langgraph.graph import MessagesState





class currentstate(BaseModel):
    query: str = None
    document: str = None
    memory: str = None
