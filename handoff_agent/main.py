
import os 
from agents import OpenAIChatCompletionsModel,RunConfig,Runner,Agent
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
from voice_agent import VoiceAgent,task

load_dotenv()

user_choice=input("Do you want to use default api press enter else input API KEY:")
if user_choice.strip() == "":
    Api_key=os.getenv("Gemini_api_key")
else:
    Api_key = user_choice


if not Api_key:
    raise ValueError("Error: API key was not found. Insert an API key.")

Client=AsyncOpenAI(
    api_key=Api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model=OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=Client,

)
config=RunConfig(
    model=model,
    model_provider=Client,
    tracing_disabled=True,
)

async def main():
    Deciding_Agent=Agent(
        name="Decider",
        instructions = "You are agents who decide if the task is in voice or in text after deciding you handoff the work to the voice agent who can solve the task more efficiently ",
        model= model,
        handoffs=[VoiceAgent]
    )
    # while True:
    #     task=input("input a task or type quit/q to exit: ")
    #     if task.lower() in ["q","quit"]:
    #         break
    #     elif task.strip() == "":
    #         print("Please Enter a task or follow the instructions")
    #         task=input("input a task or type quit/q to exit: ")
            


    result=await Runner.run(Deciding_Agent,task,run_config=config)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
