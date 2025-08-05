from typing import List, Tuple
from altair import Description
import pandas as pd

from langchain.chat_models import init_chat_model

from pydantic import BaseModel, Field
from typing import Dict,Any

from typing_extensions import Annotated



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
    
    pitch : str = Field (
        description = "The macro investment pitch on what is a good trade right now"
    )
    instructions : str = Field (
        description = "Provide instructions to sub-agent to perform further research"
    )
    created_agent_description : str = Field(
        description = "For the newly created agent provide the information about it including description, task and other information"
    )

class SectorAnalystSchema(BaseModel):
    """Output Schema for Equity Research Analyst"""
    sector: str = Field(
        description="Industry sector (e.g., Technology, Healthcare)"
    )
    rating: str = Field(
        description="Analyst rating (e.g., Buy, Hold, Sell)"
    )
    price_target: float = Field(
        description="Target price for the stock"
    )
    research: str = Field(
        description="Research on the specific sector"
    )


class QuantAnalystSchema(BaseModel):
    """Output Schema for Quant Analyst"""

    strategy_name: str = Field(
        description="Name of the quantitative strategy"
    )
    universe: str = Field(
        description="Asset universe (e.g., S&P 500, global equities)"
    )
    signal: str = Field(
        description="Quantitative signal or factor (e.g., momentum, value)"
    )
    performance: dict = Field(
        description="Performance metrics (e.g., {'return': 0.12, 'volatility': 0.08, 'sharpe': 1.5})"
    )
    insight: str = Field(
        description="Key insight or interpretation of the results"
    )

class GeneralAgentSchema(BaseModel):
    """Output Schema for General Agent"""

    research: str = Field(
        description= "The research done by the agent"
    )

class FXAgentSchema(BaseModel):
    """Output Schema for FX Agent"""

    research : str = Field(
        description = "Key research and insights on FX"
    )


class CentralBankSchema(BaseModel):
    """Output Schema for Central Bank Agent"""

    outlook : str = Field(
        description = "Outlook for the respective sector"
    )
    research : str = Field(
        description= "Research for the respective sector"
    )
    Source : str = Field(
        description = "Source of the respective research"
    )

class PerformanceMetrics(BaseModel):
    return_: float = Field(..., description="Portfolio return")
    volatility: float = Field(..., description="Portfolio volatility")
    sharpe: float = Field(..., description="Portfolio Sharpe ratio")

    class Config:
        extra = "forbid"  # This sets additionalProperties: false

class PortfolioManagerSchema(BaseModel):
    portfolio_summary: str = Field(
        description="Summary of the current portfolio holdings and allocation"
    )
    performance: PerformanceMetrics = Field(
        ..., description="Portfolio performance metrics"
    )
    risk_analysis: str = Field(
        description="Analysis of portfolio risks and exposures"
    )
    recommendations: str = Field(
        description="Portfolio manager's recommendations for changes or actions"
    )
    source: str = Field(
        description="Source of the data or research used for the analysis"
    )

    class Config:
        extra = "forbid"
