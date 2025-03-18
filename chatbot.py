import chainlit as cl
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv, find_dotenv
import os

from openai import OpenAI

load_dotenv(find_dotenv())

# Initialize the OpenAI API client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")




@cl.on_chat_start
async def on_chat_start():
        # Step 1: Provider
    provider = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # Step 2: Model
    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=provider,
    )

    # Step 3: Config Define at Run Time
    config = RunConfig(
        model=model,
        model_provider=provider,
        tracing_disabled=True,
    )

    # Step 4: Agent
    agent: Agent = Agent(
        name="Assistant", 
        instructions='''As a telephone directory inquiry operator, you are responsible for professionally handling incoming calls and providing accurate contact information. Begin each call with a polite greeting
        Speak clearly and listen attentively to understand the caller’s request. 
        Confirm the details by asking for clarification if necessary, 
        search the directory system efficiently, and provide the correct number, 
        ensuring that the caller receives accurate information. If the number is unavailable, 
        suggest an alternative, such as the main reception number. For multiple inquiries, 
        prioritize based on urgency and politely ask the caller to hold if additional time is needed. 
        If faced with an impatient caller, remain calm, professional, and rephrase responses for clarity. 
        In case of unresolved issues, escalate them to a supervisor. Maintain logs of frequently requested numbers,
        report any outdated or incorrect information for updates, and always follow security protocols when handling 
        sensitive contact details. Your role is crucial in ensuring callers receive prompt and reliable assistance
        while maintaining professionalism and efficiency.''',
    )

    # Step 5: Set the chat history, config, and agent in the user session

    cl.user_session.set("chat-history", [])  
    cl.user_session.set("config", config)
    cl.user_session.set("agent", agent) 
    await cl.Message(content="Hello, this is the Phone number Inquiry Service. How may I assist you?").send()


@cl.on_message
async def main(message: cl.Message):
    # Step 6: Send a thinking message
    msg = cl.Message(content="Thinking...")
    await msg.send()

    # Step 7: Retrieve the agent, config, and chat history from the user session
    agent: Agent = cl.user_session.get("agent") 
    config: RunConfig = cl.user_session.get("config")

    # Step 8: Retrieve the chat history from the session.
    history = cl.user_session.get("chat-history") or []
    
  
    # Step:9 Append the user's message to the history.
    history.append({"role": "user", "content": message.content})

    try:
        # Step 10: Call the Runner.run_sync method to generate a response
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        result = Runner.run_sync(starting_agent = agent,
                    input=history,
                    run_config=config)
        
        # Step 11: Retrieve the response from the result object
        response_content = result.final_output

        # Step 12: Update the thinking message with the actual response
        msg.content = response_content
        await msg.update()

        # Step 13: Update the session with the new history.
        cl.user_session.set("chat-history", result.to_input_list())

        # Step 14: (Optional) Log the interaction
        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")
        print("=" * 50)

    except Exception as e:
        # Step 15: Handle any exceptions that might occur during the execution
        msg.content = f"An error occurred: {e}"
        await msg.update()
        print(f"Error: {e}")







    # Step 16: Handle the response as needed    
    # For example, you can send the response back to the user or store it for later use.
    # For simplicity, we'll just print the response here.
    print(f"Assistant: {response_content}")










