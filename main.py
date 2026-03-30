import os
import re
import speech_recognition as sr
import webbrowser
import pyttsx3
import musicLibrary
from dotenv import load_dotenv
from openai import OpenAI  # Perplexity quickstart uses OpenAI-compatible SDK

# Load .env located next to this file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Read Perplexity key from .env
PPLX_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PPLX_KEY:
    raise RuntimeError("PERPLEXITY_API_KEY missing in .env")

# Initialize client for Perplexity (quickstart shows the OpenAI-compatible SDK usage)
# If your environment requires a full base path, you can try: base_url="https://api.perplexity.ai"
client = OpenAI(api_key=PPLX_KEY, base_url="https://api.perplexity.ai")

recognizer = sr.Recognizer()
engine = pyttsx3.init()


def speak(text: str):
    """Speak a short plain-text response."""
    engine.say(text)
    engine.runAndWait()


def _clean_text(text: str) -> str:
    """Return plain text by removing HTML, citation markers and extra whitespace.

    - strips HTML tags
    - removes citation brackets like [1], (1)
    - removes common 'Sources:' or 'References' footers if present
    - collapses repeated whitespace/newlines
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove citation markers like [1], [12], (1), (see source)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)

    # Remove common 'Sources:' or 'References' sections (keep only sentence before them)
    text = re.split(r"\n\s*(Sources|References|Citations)[:\n]", text, flags=re.IGNORECASE)[0]

    # Remove leftover URLs
    text = re.sub(r"https?://\S+", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def aiProcess(command: str) -> str:
    """Call Perplexity (sonar-pro) and return a single clean string answer.

    The function is defensive: it handles missing fields and exceptions and always
    returns a short plain-text string (never raw JSON).
    """
    try:
        completion = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": "You are Jarvis. Keep responses short and plain text (no markdown, no citations)."},
                {"role": "user", "content": command}
            ],
            # optional: temperature=0.2, max_tokens=200
        )

        # Defensive extraction
        text = ""
        if hasattr(completion, "choices") and completion.choices:
            choice = completion.choices[0]
            # older/newer SDKs expose message either as .message.content or .text
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                text = choice.message.content
            elif hasattr(choice, "text"):
                text = choice.text
            else:
                # fallback to str()
                text = str(choice)
        else:
            # fallback to top-level fields sometimes used by SDKs
            text = getattr(completion, "text", "") or str(completion)

        # Clean the text to plain answer
        answer = _clean_text(text)
        if not answer:
            return "I couldn't generate an answer. Try rephrasing your question."
        return answer

    except Exception as e:
        # Log minimal error locally for debugging, but return a friendly plain-text message
        print(f"AI error: {e}")
        return "Sorry, I couldn't get an answer right now."


def processCommand(c: str):
    cl = c.lower()
    if "open google" in cl:
        webbrowser.open("https://google.com")
    elif "open facebook" in cl:
        webbrowser.open("https://facebook.com")
    elif "open youtube" in cl:
        webbrowser.open("https://youtube.com")
    elif "open linkedin" in cl:
        webbrowser.open("https://linkedin.com")
    elif cl.startswith("play"):
        parts = cl.split()
        if len(parts) > 1 and parts[1] in musicLibrary.music:
            webbrowser.open(musicLibrary.music[parts[1]])
        else:
            speak("Song not found.")
    else:
        output = aiProcess(c)
        print("Jarvis reply:", output)  # useful for debugging
        speak(output)


if __name__ == "__main__":
    speak("Initializing Jarvis...")
    r = sr.Recognizer()
    while True:
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                print("Listening for wake word...")
                audio = r.listen(source, timeout=3, phrase_time_limit=2)
            word = r.recognize_google(audio)
            if word.lower() == "jarvis":
                speak("Yes?")
                with sr.Microphone() as source:
                    print("Jarvis active...")
                    r.adjust_for_ambient_noise(source, duration=0.3)
                    audio = r.listen(source, timeout=6, phrase_time_limit=6)
                    command = r.recognize_google(audio)
                processCommand(command)
        except Exception as e:
            # Print only; don't speak every internal error
            print(f"Error: {e}")
            # small sleep could be added to avoid tight error loop
            continue