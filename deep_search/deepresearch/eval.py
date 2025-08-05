from deepresearch.agent import CompassAI
import asyncio
import os
import argparse
from typing import Callable, List, TypeVar
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
from evals.simple_evals.simpleqa_eval import SimpleQAEval





async def evaluate_single_query(query: str):
    research = CompassAI(
        query = query,
        report_type = "pdf",
        tone = "professional"
    )
    output = await research.start_research()
    