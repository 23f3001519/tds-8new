from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "llama3.2"


class InvoiceRequest(BaseModel):
    text: str


class InvoiceResponse(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str


@app.post("/extract", response_model=InvoiceResponse)
def extract(req: InvoiceRequest):

    prompt = f"""
Extract the invoice information.

Return ONLY valid JSON.

Schema:
{{
"vendor":"",
"amount":0,
"currency":"",
"date":"YYYY-MM-DD"
}}

Invoice:

{req.text}
"""

    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
        },
        timeout=60,
    )

    content = r.json()["choices"][0]["message"]["content"]

    # Remove markdown if present
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(content)
    except Exception:
        data = {
            "vendor": "",
            "amount": 0,
            "currency": "USD",
            "date": "1970-01-01",
        }

    return InvoiceResponse(**data)