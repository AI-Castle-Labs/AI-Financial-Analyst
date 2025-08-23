from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from langgraph.channels import LastValue
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from langgraph.channels import LastValue

class AgentState(BaseModel):

    
    title : str | None = None
    
    macro_thesis: str | None  = None
    
    instructions: Optional[Dict[str, Any]] = None


    #Response from every agent

    central_bank_agent : Optional[list[str]] = None
    macro_analyst_agent : Optional[list[str]] = None
    sector_analyst_agent : Optional[list[str]] = None
    fx_research_agent : Optional[list[str]] = None
    portfolio_manager_agent : Optional[list[str]] = None
    agent : Optional[list[str]] = None

    agent_description: Annotated[Any, LastValue] = None



    prompt: Optional[str] = None


    result : Optional[list[str]] = None