"""Can we route the brain through AssemblyAI's LLM Gateway instead of Gemini direct?

Same reasoning, but it keeps more of the pipeline on the sponsor's stack.
"""
import json, os, time, urllib.request
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["ASSEMBLYAI_API_KEY"]
URL = "https://llm-gateway.assemblyai.com/v1/chat/completions"

PROMPT = ("BOOK CONTEXT: An eigenvector is a vector whose direction is unchanged "
          "when a matrix is applied to it; only its length scales.\n\n"
          "THEY ASK: wait, what did that last sentence mean?")


def call(model: str):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Two sentences, spoken aloud, no markdown."},
            {"role": "user", "content": PROMPT},
        ],
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": KEY, "Content-Type": "application/json"},
    )
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req))
    return time.perf_counter() - t, r


if __name__ == "__main__":
    for model in ["gemini-2.5-pro", "claude-sonnet-4-6", "gpt-5.2", "gemini-2.5-flash"]:
        try:
            dt, r = call(model)
            txt = r["choices"][0]["message"]["content"].strip().replace("\n", " ")
            print(f"{model:20} OK   {dt:5.2f}s  {txt[:130]}")
        except Exception as e:
            detail = e.read().decode()[:160] if hasattr(e, "read") else str(e)[:160]
            print(f"{model:20} FAIL {detail}")
