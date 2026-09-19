"""Generate a short technical 'audiobook' to develop and demo against."""
import os
from dotenv import load_dotenv
from playhead.voice import ElevenLabsVoice

load_dotenv()

CHAPTER = (
    "Chapter four. Eigenvectors and the shape of data. "
    "When we apply a matrix to a vector, two things usually happen at once. "
    "The vector gets stretched, and it gets rotated into a new direction. "
    "But for any given matrix there exist a handful of special vectors that refuse to rotate. "
    "Apply the matrix to one of these, and it comes back pointing exactly the way it started, "
    "merely longer or shorter than before. These are the eigenvectors, and the factor by which "
    "each one is scaled is its eigenvalue. "
    "This matters more than it first appears. If you think of a matrix as a transformation of space, "
    "the eigenvectors are the axes along which that transformation is simplest. "
    "Everything the matrix does can be described as stretching along those axes. "
    "Principal component analysis takes exactly this idea and points it at a cloud of data. "
    "It builds the covariance matrix of the data, finds its eigenvectors, and treats them as "
    "the natural axes of the cloud. The eigenvector with the largest eigenvalue points along the "
    "direction in which the data varies most. That is your first principal component. "
    "The second points along the next most variable direction, at right angles to the first, and so on. "
    "Keep only the first few, and you have compressed your data while throwing away remarkably little. "
    "The catch, and it is a real one, is that principal component analysis only ever finds "
    "directions that are straight lines. If the structure in your data curves, no rotation of the axes "
    "will capture it, and you will need something rather more sophisticated."
)

if __name__ == "__main__":
    v = ElevenLabsVoice(os.environ["ELEVENLABS_API_KEY"], os.getenv("ELEVENLABS_VOICE_ID", ""))
    print(f"[demo] synthesizing {len(CHAPTER)} characters...")
    path = v.to_wav(CHAPTER, "audio/chapter4.wav")
    import soundfile as sf
    info = sf.info(path)
    print(f"[demo] wrote {path}: {info.duration:.0f}s")
