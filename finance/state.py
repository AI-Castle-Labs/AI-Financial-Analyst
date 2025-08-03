from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from langgraph.channels import LastValue



class AgentState:

    macro_thesis: str | None  = None
    
    instructions: Optional[Dict[str, Any]] = None


    #Response from every agent

    central_bank_agent : Optional[list[str]] = None
    macro_analyst_agent : Optional[list[str]] = None
    sector_analyst_agent : Optional[list[str]] = None
    fx_research_agent : Optional[list[str]] = None
    agent : Optional[list[str]] = None

    agent_description : str = None



    prompt : str = None


    