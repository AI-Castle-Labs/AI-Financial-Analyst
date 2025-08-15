import anthropic
import logging
import time
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

def chart_agent(self,data_point):
    llm = self.llm.with_structured_output(Classification_outputSchema)
    system_prompt = system_source_prompt

    result = self.llm.invoke([
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"You are a macro analyst and you are analyzing {prompt} for a firm provide 2 data points, be very specific"}
    ])
    print(result)
    output_structure = {'data_source':result.datasource, 'name_point':result.name_point, 'data_point': result.data_point}
    print(FRED_Chart(result.name_point,result.data_point))



def ask_sonar(ideas):
    api_key = os.getenv("PERPLEXITY_API_KEY")

    if not api_key:
        print("⚠️ PERPLEXITY_API_KEY not found in environment variables")
        # Provide a direct fallback for testing
        api_key = "API-KEY"  # Replace with your key

    print("Entering sonar")
    model = "sonar-pro"  # Use online model for most current data
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    system_prompt = f"""
    <role>
    You are an AI Investment Pitch Analyst responsible for findinging investment pitch for potential investments.
    </role>

    <context>
    Find a potential investment idea for an AI Financial Anlayst to provide recommendations for
    
    Focus on finding:
    1. Current Market Events
    2. Valuations concerns for the asset class
    3. Different Asset classes not just equities
    4. Look internationally as well not just the US
  

    IMPORTANT: You MUST include proper markdown hyperlinks for your sources like [Source Name](https://example.com)
    Don't just paste plain URLs - format them as proper clickable links using markdown.
    
    For each point, cite at least one specific source with a hyperlink.
    Include at least 5 different sources total.
    
    Format your response as:
    
    ## 1)Potential Idea
    [Research findings with hyperlinks]
    [Citation Link]
    
    ## 2) Fundamentals
    [Research findings with hyperlinks]
    [Citation Link]
    
    
    ## 3)Risk
    [Research findings with hyperlinks]
    [Citation Link]
    
    
    ## 4) Investment Horizon
    [Research findings with hyperlinks]
    [Citation Link]
    
    ##5) Long/Short
    [Research findings with hyperlinks]
    [Citation Link]
    
    
    </context>
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research this venture idea thoroughly: {title} - {description}"}
        ]
    )
    print("perplexity search results", response.choices[0].message.content)
    return response.choices[0].message.content
import os
from dotenv import load_dotenv

def tavilly_search(query):
    client = TavilyClient(os.getenv("tavily_api_key"))
    response = client.search(
        query=query
    )
    return response    

def claude_search():
     return 0



"""
import  sdmxreturbn 
from msal import PublicClientApplication

 

# parameter values for authorization and data requests

client_id = '446ce2fa-88b1-436c-b8e6-94491ca4f6fb'

authority = 'https://imfprdb2c.b2clogin.com/imfprdb2c.onmicrosoft.com/b2c_1a_signin_aad_simple_user_journey/'

scope = 'https://imfprdb2c.onmicrosoft.com/4042e178-3e2f-4ff9-ac38-1276c901c13d/iData.Login'

# authorize and retrieve access token

app = PublicClientApplication(client_id,authority=authority)

token = None

token = app.acquire_token_interactive(scopes=[scope])

# define header for a request

header = {'Authorization': f"{token['token_type']} {token['access_token']}"}

 

# retrieve data

IMF_DATA = sdmx.Client('IMF_DATA')


 

# convert to pandas

cpi_df = sdmx.to_pandas(data_msg)



"""

