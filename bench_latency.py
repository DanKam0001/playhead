"""Time-to-first-spoken-word. The number a viewer actually feels."""
import os, time
from dotenv import load_dotenv
load_dotenv()
from playhead.brain import Brain
from playhead.library import Library
from playhead.voice import ElevenLabsVoice

Q, T = "Wait, what did that last sentence mean?", 30.0

def main():
    lib = Library("data/chapter4.db")
    brain = Brain(os.environ["GEMINI_API_KEY"], lib)
    voice = ElevenLabsVoice(os.environ["ELEVENLABS_API_KEY"], os.getenv("ELEVENLABS_VOICE_ID",""))

    t0 = time.perf_counter()
    reply = brain.answer(Q, T)
    t_llm = time.perf_counter() - t0
    # First byte of audio back from ElevenLabs = when sound could start.
    t1 = time.perf_counter()
    gen = voice._synthesize(reply)
    first = next(iter(gen))
    t_tts = time.perf_counter() - t1
    print(f"  LLM full reply     {t_llm:5.2f}s  ({len(reply)} chars)")
    print(f"  TTS first audio    {t_tts:5.2f}s  ({len(first)} bytes)")
    print(f"  TOTAL to first word{t_llm + t_tts:5.2f}s")
    lib.close()

if __name__ == "__main__":
    main()
