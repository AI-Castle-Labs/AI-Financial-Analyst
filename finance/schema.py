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


import tweepy
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("X_API_KEY")
api_key_secret = os.getenv("X_API_KEY_SECRET")
access_token = os.getenv("X_ACCESS_TOKEN")
access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("token")

client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_key_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)

# --- Post the tweet ---
response = client.create_tweet(text="Hello world from RAG@UIUC 🚀")

