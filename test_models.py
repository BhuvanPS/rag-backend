import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)
