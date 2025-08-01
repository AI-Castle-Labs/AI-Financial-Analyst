from typing import List, Tuple
from altair import Description
import pandas as pd

from langchain.chat_models import init_chat_model

from pydantic import BaseModel, Field




#Q/A Output Schema

class Classification_outputSchema(BaseModel):

    """Output Schmea for Q/A"""

    datasource: str = Field(

        description = "Data source for the respective data point"

    )
    name_point : str = Field(
        description= "The name of the datapoint for example for FEDFUNDS it would be FRED Fed Funds Rate"
    )

    data_point : str = Field(

        description = "The name of the data point for the respective source. Example for Fred Fed Funds Rate, output would be FEDFUNDS"

    )


class MacroAnalystSchema(BaseModel):
    """Output Schema for Macro Analyst"""

    datasource : str = Field (
        description= "Data source for the respective data point"
    )
    
    insight : str = Field (
        description = "The daily insight on what has happened"
    )

class EquityResearchAnalystSchema(BaseModel):
    """Output Schema for Equity Research Analyst"""

    ticker: str = Field(
        description="Stock ticker symbol (e.g., AAPL, MSFT)"
    )
    company_name: str = Field(
        description="Full company name"
    )
    sector: str = Field(
        description="Industry sector (e.g., Technology, Healthcare)"
    )
    rating: str = Field(
        description="Analyst rating (e.g., Buy, Hold, Sell)"
    )
    price_target: float = Field(
        description="Target price for the stock"
    )
    summary: str = Field(
        description="Summary of the analyst's view and key drivers"
    )