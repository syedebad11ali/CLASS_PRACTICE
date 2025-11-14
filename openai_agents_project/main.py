from agents import (Agent,RunConfig,Runner,OpenAIChatCompletionsModel, 
function_tool,
input_guardrail,GuardrailFunctionOutput,InputGuardrailTripwireTriggered,
output_guardrail,InputGuardrailResult,OutputGuardrailResult)
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio
from typing import TypedDict
from agents.tool_context import RunContextWrapper
from agents import AgentHooks, RunContextWrapper
import random as ra
import emoji as e
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

load_dotenv()

Gemini_api_key=os.getenv("Api_key")
if not Gemini_api_key:
    raise ValueError(f"The api key give from {Gemini_api_key} was not valid")





Client=AsyncOpenAI(

    api_key=Gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",

)
model=OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=Client,
)
runconfig=RunConfig(
    
    model=model,
    model_provider=Client,
    tracing_disabled=True
)


class UserContext(TypedDict,total=False):
    """this calss just hold the user name"""
    name : str

class Appropriate_Language(BaseModel):
    is_appropriate_language: bool
    reasoning: str

guardrail_agent = Agent( 
    name="Guardrail check",
    instructions="Check if the user is using appropriate language and is being respectable",
    output_type=Appropriate_Language,
)


@input_guardrail
async def bad_word_check(ctx:RunContextWrapper[UserContext], agent:Agent[UserContext], input_data) ->GuardrailFunctionOutput:
    
    user_input=""
    if isinstance(input_data,list) and len(input_data)>0:
        user_input=input_data[-1].get("content","")
    elif isinstance(input_data,dict):
        user_input=input_data.get("content","")
    else:
        user_input=str(input_data)
    
    input_data= await Runner.run(guardrail_agent,input=user_input,context=ctx.context,run_config=runconfig)
    
    is_clean = input_data.final_output.is_appropriate_language

    if not is_clean:
        # Tripwire = inappropriate language found
        return GuardrailFunctionOutput(
            output_info="Please use appropriate language 🙏",
            tripwire_triggered=True
        )
    
    # Otherwise, pass through normally
    return GuardrailFunctionOutput(
        output_info="Language check passed",
        tripwire_triggered=False
    )
@output_guardrail
def polite_closing(ctx, agent, output_data):
    if isinstance(output_data, list):
        text = " ".join(str(item) for item in output_data)
    else:
        text = str(output_data)

    # Just pass the text as-is (don’t wrap in dict)
    return GuardrailFunctionOutput(
        output_info=text,
        tripwire_triggered=False,
    )

def decorate_output(text: str, enable: bool) -> str:
    """
    Optionally add extra emojis at start and end if emoji_enabled.
    """
    emojis = ["🙂", "😄", "🤖", "🚀", "🌟", "🎉"]
    clean = text.strip()
    if enable:
        return f"{ra.choice(emojis)} {clean} {ra.choice(emojis)}"
    return clean

@function_tool
def Wheather(city:str) ->str:
    "returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

# @function_tool
# def unavliable_language(language:str) ->str:
#     """Return an statement based on language and tell it that the agents doesn,t work on this task """
#     return f"I am unable to genrate word or any thing in the {language}"






def dynamic_instruction(context: RunContextWrapper[UserContext], agent: Agent[UserContext]) -> str:
    raw_input = getattr(context, "_input", None)
    user_name = context.context.get("name", "Dynamic Agent")
    
    # Extract the last user message content
    last_message = None
    if raw_input and "context" in raw_input and len(raw_input["context"]) > 0: #simply sayong if raw input doesn't have context the last message become an empty string
        last_message = raw_input["context"][1].get("content", "").lower()
    else:
        last_message = ""
        
    print(f"Agent starting with last user input: {last_message}")

    base_instruction = (
    "You have access to tools, "
    "but only use them if absolutely needed to answer correctly. "
    "If you can answer without tools, respond directly."
    )
    if any(phrase in last_message for phrase in ["what can you do", "what are you capable of"]):
        return f"{user_name}, you are a translator. {base_instruction}"

    if "summarize" in last_message:
        return f"{user_name}, You are a professional summarizing expert with 20 years experience."
    elif "translate" in last_message:
        return f"{user_name}, You are a professional translator, who can translate anything."
    elif "not understand" in last_message or "didn't understand" in last_message:
        return f"{user_name}, You are a helpful assistant, explain as if a 5 year old could understand, be polite and helpful."
    elif "explain" in last_message:
        return f"{user_name}, You are a helpful agent who can assist on any task and generate information based on user needs."
    
    else:
        return f"{user_name}, you are a general-purpose helpful assistant."


# watashi wa ringo ga suki desu demo banana mou daisuki desu    tanslate into english
class MyAgentHooks(AgentHooks):
    async def on_start(self, context: RunContextWrapper, agent: Agent):
        raw_input = getattr(context, "_input", None)
        print(f"Agent starting with input: {raw_input}")
    async def on_tool_start(self, tool_context, agent, func_tool):
        tool_input = getattr(tool_context, 'input', None)
        print(f"Calling tool '{func_tool.name}' with input: {tool_input}")

    async def on_tool_end(self, tool_context, agent, func_tool, output):
        print(f"Tool '{func_tool.name}' returned: {output}")

    async def on_end(self, context: RunContextWrapper, agent: Agent, result):
        print(f"Agent finished with output: {result}")

async def main():
    My_agent=Agent(
        name="Dynamic Agent",
        
        instructions=dynamic_instruction, 
        model=model,
        tools=[Wheather],
        hooks=MyAgentHooks(),
        tool_use_behavior="auto",
        input_guardrails=[bad_word_check],
        output_guardrails=[polite_closing]
        
    )

    warning_left:int=3
    emoji_enabled:bool=False
    chat_history : list = [] 
    while True:
        task:str=input("Enter a task:")
        
        if task.strip() == "" or task.lower() in ["q","quit"]:
            break

        if task.lower() in ["remove emoji", "disable emoji"]:
            emoji_enabled = False
            print("Emoji decoration disabled permanently.")
            continue
        elif task.lower() in ["add emoji", "enable emoji"]:
            emoji_enabled = True
            print("Emoji decoration enabled permanently.")
            continue

        chat_history.append({"role":"user","content":task})           
        try:
            result=await Runner.run(
                My_agent,
                input=chat_history,
                context={"user":"Ebad"},
                run_config=runconfig,)
        except InputGuardrailTripwireTriggered as e:
            warning_left-=1
            print("\n\tPlease use appropriate language!\n")
            if warning_left > 0:
                print(f"You have {warning_left} warnings left")
            else:
                print("No warning left")
                break
            continue
        
        raw_output = result.final_output_as(str)
        final = decorate_output(raw_output,emoji_enabled)
        print(f"\n{final}\n")
if __name__=="__main__":
    asyncio.run(main())