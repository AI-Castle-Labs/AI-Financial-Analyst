from langchain.tools import Tool
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import openai



class MySearchTool(BaseTool):

