from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from langgraph.channels import LastValue



class AgentState:

    macro_thesis: str | None  = None
    
    instructions: Optional[Dict[str, Any]] = None