"""Verify every vendor key before you rely on it in a live demo."""
import os
from dotenv import load_dotenv

load_dotenv()


def check_assemblyai():
    import assemblyai as aai
    aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]
    page = aai.Transcriber().list_transcripts()
    return f"account reachable, {len(page.transcripts)} prior transcripts"


def check_gemini():
    from google import genai
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    names = [m.name for m in c.models.list()]
    embed = [n for n in names if "embedding" in n]
    return f"{len(names)} models, embedding available: {bool(embed)}"


def check_elevenlabs():
    from elevenlabs.client import ElevenLabs
    c = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    sub = c.user.subscription.get()
    left = sub.character_limit - sub.character_count
    return f"{left:,} characters left on {sub.tier}"


if __name__ == "__main__":
    for name, fn in [("AssemblyAI", check_assemblyai), ("Gemini", check_gemini),
                     ("ElevenLabs", check_elevenlabs)]:
        try:
            print(f"{name:12} OK   {fn()}")
        except Exception as exc:
            print(f"{name:12} FAIL {type(exc).__name__}: {str(exc)[:110]}")
