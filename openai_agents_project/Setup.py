import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent, RunConfig, Runner, OpenAIChatCompletionsModel, handoffs, RunContextWrapper
)

load_dotenv()
API_KEY = os.getenv("Api_key")
if not API_KEY:
    raise RuntimeError("set Api_key in .env")

# --- client + model + runconfig ---
Client = AsyncOpenAI(api_key=API_KEY,base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=Client)
runconfig = RunConfig(model=model, model_provider=Client, tracing_disabled=True)

# --- simple agents + handoff example (same structure as your code) ---
SalesAgent = Agent(name="Sales Agent",
                   instructions="You are a sales assistant. Help with pricing/upgrades.",
                   model=model)

async def handle_sales_handoff(ctx: RunContextWrapper, args_json: str):
    print("\n--- Handoff Triggered: Support -> Sales ---")
    print("args_json:", args_json)
    return SalesAgent

SupportAgent = Agent(
    name="Support Agent",
    instructions="You are a helpful support assistant. If the user asks about pricing, hand off.",
    handoffs=[
        handoffs.Handoff(
            tool_name="handoff_to_sales",
            tool_description="handoff to sales",
            input_json_schema={},
            on_invoke_handoff=handle_sales_handoff,
            agent_name="Sales Agent",
        )
    ],
    model=model
)

conversation_history = []

# --- safe helper to print any event chunk text/delta ---
def extract_chunk_text(ev):
    # ev might be an object with .delta / .text, or a dict with keys
    # Try common variants safely:
    text = None
    try:
        # object-like: ev.delta or ev.text
        text = getattr(ev, "delta", None) or getattr(ev, "text", None)
        # some SDKs store delta as dict: {'choices': [{'delta': {'content': '...'}}]}
        if text is None:
            if hasattr(ev, "choices") and ev.choices:
                # object-like choice
                c = ev.choices[0]
                delta = getattr(c, "delta", None) or (c.get("delta") if isinstance(c, dict) else None)
                if isinstance(delta, dict):
                    text = delta.get("content") or delta.get("text")
        # dict-like event
        if text is None and isinstance(ev, dict):
            # try many shapes
            text = ev.get("text") or (ev.get("choices") and ev["choices"][0].get("delta", {}).get("content"))
    except Exception:
        text = None
    # If text is a dict/choice object, dig deeper
    if isinstance(text, dict):
        return text.get("content") or text.get("text")
    return text

# --- attempt multiple streaming strategies ---
async def stream_with_runner_instance(agent, conversation, context):
    """
    Preferred: create a Runner instance and try runner.run_stream(...)
    """
    # Create instance runner - many SDKs expect Runner(agent, client=Client)
    try:
        runner_inst = Runner(agent, client=Client)
    except TypeError:
        # some Runner constructors differ; fallback to no-inst creation
        runner_inst = None

    if runner_inst and hasattr(runner_inst, "run_stream"):
        async for ev in runner_inst.run_stream(input=conversation, context=context, run_config=runconfig):
            chunk = extract_chunk_text(ev)
            if chunk:
                print(chunk, end="", flush=True)
        return True

    # Some SDKs implement 'stream' instead of 'run_stream' on the instance
    if runner_inst and hasattr(runner_inst, "stream"):
        async for ev in runner_inst.stream(input=conversation, context=context, run_config=runconfig):
            chunk = extract_chunk_text(ev)
            if chunk:
                print(chunk, end="", flush=True)
        return True

    return False

async def stream_with_runner_class(agent, conversation, context):
    """
    Try class-level Runner.run_stream(...) (some older examples call Runner.run_stream(...))
    """
    run_stream_fn = getattr(Runner, "run_stream", None) or getattr(Runner, "stream", None)
    if run_stream_fn:
        # call it with the same signature user used for Runner.run earlier
        async for ev in run_stream_fn(agent, input=conversation, context=context, run_config=runconfig):
            chunk = extract_chunk_text(ev)
            if chunk:
                print(chunk, end="", flush=True)
        return True
    return False

async def raw_model_stream(conversation, context):
    """
    Fallback: stream directly from the Gemini/OpenAI client.
    This does NOT run Agent-tool-handling logic (no automatic handoffs or tool calls),
    but it will stream text from the model.
    """
    messages = []
    # convert conversation list to simple messages if it's a list of dicts
    for item in conversation:
        if isinstance(item, dict) and "role" in item and "content" in item:
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            # best-effort convert
            messages.append({"role": "user", "content": str(item)})

    # This call shape works for many AsyncOpenAI clients: .chat.completions.create(..., stream=True)
    # Correct way to stream
    async for chunk in await Client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages,
        stream=True
):
    # Access the content directly
        content = chunk.choices[0].delta# or chunk.text depending on the type of chunk
        if content:
            print(content, end="", flush=True)

async def streaming_turn(agent, conversation, context):
    # Try instance runner streaming first
    ok = await stream_with_runner_instance(agent, conversation, context)
    if ok:
        return
    # Try class-level runner streaming
    ok = await stream_with_runner_class(agent, conversation, context)
    if ok:
        return
    # Fallback to raw model stream (no agent tool/handoff handling)
    print("\n[warning] streaming via raw model (no agent tool/handoff handling)\n")
    await raw_model_stream(conversation, context)

# --- main REPL ---
async def main():
    print("Streaming REPL (type q or quit to exit)")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"q", "quit", "exit"}:
            break
        conversation_history.append({"role": "user", "content": user})
        print("Assistant: ", end="", flush=True)
        await streaming_turn(SupportAgent, conversation_history, {"user_id": "guest"})
        print("\n")  # newline and separate turns

if __name__ == "__main__":
    asyncio.run(main())
