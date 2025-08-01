from fredapi import Fred
import matplotlib.pyplot as plt
import sdmx
import pandas as pd

"""

def FRED_Chart(name_point, data_point):


    fred = Fred(api_key = '53a8c45b1e8169b89b2070221bf0773d')

    # Step 2: Get GDP data
    gdp = fred.get_series(data_point)

    # Step 3: Plot it
    gdp.plot(title= name_point)
    plt.xlabel("Date")
    plt.ylabel("Billions of Dollars")
    plt.grid(True)
    plt.show()



import  sdmx

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