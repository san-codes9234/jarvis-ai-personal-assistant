import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env next to this file (or resolve relative to project root as needed)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

PPLX_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PPLX_KEY:
    raise RuntimeError("PERPLEXITY_API_KEY missing in .env")

# Export a Perplexity-compatible client
pplx_client = OpenAI(api_key=PPLX_KEY, base_url="https://api.perplexity.ai")

def ask_perplexity(messages, model: str = "sonar-pro"):
    """
    messages: list of dicts like [{"role":"system","content":"..."},{"role":"user","content":"..."}]
    returns: assistant text
    """
    resp = pplx_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return resp.choices[0].message.content
