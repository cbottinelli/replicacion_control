from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

U_MODEL = "gpt-4.1-mini"
H_MODEL = "gpt-4.1-mini"
T_MODEL = "gpt-3.5-turbo"
T_INSTRUCT_MODEL = "gpt-3.5-turbo-instruct"