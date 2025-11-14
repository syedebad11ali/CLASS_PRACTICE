import winsound
import speech_recognition as sr
import pyttsx3
import os
import asyncio

from agents import (
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    Agent,
    function_tool,
)
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Init components
recognizer = sr.Recognizer()
engine = pyttsx3.init()
load_dotenv()

# === Speak Function ===
def do_speak(text):
    engine.say(text)
    engine.runAndWait()

speak = function_tool(do_speak)  # Tool-wrapped version

# === Ask Function (for mic input) ===
def ask_text(prompt_text="What do you want to do today?", retry=3):
    for _ in range(retry):
        do_speak(prompt_text)
        winsound.Beep(1000, 300)
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            do_speak("Sorry, I couldn't catch that.")
        except sr.RequestError:
            do_speak("System is down. Please try again later.")
            return None
    do_speak("Too many failed attempts.")
    return None

@function_tool
def ask():
    return ask_text()

# === Load API Key ===
user_choice = input("Do you want to use default API key? (Press Enter to use default or input your key): ")
Api_key = user_choice.strip() if user_choice.strip() else os.getenv("Gemini_api_key")

if not Api_key:
    raise ValueError("API key not found. Please set it in .env or input manually.")

# === Gemini Client ===
Client = AsyncOpenAI(
    api_key=Api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=Client,
)

config = RunConfig(
    model=model,
    model_provider=Client,
    tracing_disabled=True,
)

# === Agent Loop ===

VoiceAgent = Agent(
    name="Jarvis the Voice",
    instructions="You are a very good listener. Your task is to get input by sound and give a very accurate and precise answer.",
    tools=[speak, ask],
    )

task = ask_text()
while not task:
    task= ask_text()
    if task.strip() == "":
        task=do_speak("What your name")
        break
    else:
        task=do_speak(task)
        break

result = Runner.run(
    VoiceAgent,
    task,
    run_config=config
    )

if result and isinstance(result.output, str):
    do_speak(result.output)
else:
    do_speak("Sorry, I couldn't process that request.")

