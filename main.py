import pandas as pd
import datetime
from httpx import Client
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel
from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#OPENAI_MODEL = ""
GROQ_MODEL = GroqModel('llama-3.3-70b-versatile', api_key=GROQ_API_KEY)

class Product(BaseModel):
    brand_name: str = Field(title='Brand Name', description='The brand name of the product')
    product_name: str = Field(title='Product Name', description='The name of the product')
    price: Optional[str] = Field(title='Price', description='The price of the product')
    rating_count: Optional[int] = Field(title='Rating Count', description='The rating count of the product')

class Results(BaseModel):
    dataset: list[Product] = Field(title='Datasets', description='The list of products')

web_scraping_agent = Agent(
    name='Web Scraping Agent',
    model=GROQ_MODEL,
    system_prompt=("""
        Your task is to convert a data string into a list of dictionaries.

        Step 1. Fetch the HTML text from the given URL using the fetch_html_text() function.
        Step 2. Takes the output from Step 1 and clean it up for the final output
        """),
    retries=2,
    result_type=Results,
    model_settings=ModelSettings(
        max_tokens=8000,
        temperature=0.1
    ),
)

@web_scraping_agent.tool_plain(retries=1)
def fetch_html_text(url: str) -> str:
    """
    Fetch the HTML text from the given URL.

    Args:
        url: str - The page's URL to fetch the HTML text from.

    Returns:
        str: The HTML text from the given URL
    """
    print('Calling URL:', url)
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.302.98 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.5'
  }
    with Client(headers=headers) as client:
        response = client.get(url, timeout=20)
        if response.status_code != 200:
            return f'Failed to fetch the HTML text from {url} with status code {response.status_code}'

        soup = BeautifulSoup(response.text, 'html.parser')
        with open('soup.txt', 'w', encoding='utf-8') as f:
            f.write(soup.get_text())
        print('Soup file saved')
        return soup.get_text().replace('\n', '').replace('\r', '')

@web_scraping_agent.result_validator
def validate_result(result: Results) -> Results:
    print('Validating result...')
    if isinstance(result, Results):
        print('Result is valid')
        return result
    print('Result is not valid')
    return None

def main() -> None:
    prompt = 'https://www.ikea.com/nl/en/cat/best-sellers/'
    
    try:
        response = web_scraping_agent.run_sync(prompt)
        if response.data is None:
            # raise UnexpectedModelBehavior('No data returned from the model')
            return None

        print('-' * 50)
        print('Input_tokens:', response.usage().request_tokens)
        print('Output_tokens:', response.usage().response_tokens)
        print('Total_tokens:', response.usage().total_tokens)

        lst = []
        for item in response.data.dataset:
            lst.append(item.model_dump())
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        df = pd.DataFrame(lst)
        df.to_csv(f'product_listings_{timestamp}.csv', index=False)
    except UnexpectedModelBehavior as e:
        print(e)

if __name__ == '__main__':
    main()