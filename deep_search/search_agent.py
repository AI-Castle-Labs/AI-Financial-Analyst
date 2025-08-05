from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List
import uuid
import os
import qdrant_client
from qdrant_client.http.models import PointStruct, Distance, VectorParams, SearchRequest
from sentence_transformers import SentenceTransformer
from fastapi.responses import JSONResponse
from tools.tools import pdf_extractor 
from finance.agent import finance_agent




"""
Search api should be able to provide advanced vector search answers to user queries

Call API -> Document and api key -> Conversation -> Conversation

"""

def finance_agent(task):
    if (task):
        finance_agent(task)


def classification(State):
    if (State.classification == "Finance"):
        finance_agent(State.task)
    
    elif (State.classification == "Medicine"):
        medicine_agent(State.task)
    
    elif (State.classification == "Other"):
        other_agent(State.task)





def upload_document(api_key, file):
    ""
    
    if not api_key:
        print("No valid api_key")
    
    upload = pdf_extractor(api_key,file)



