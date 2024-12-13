from langchain.agents import AgentType, initialize_agent
from langchain.agents.structured_chat.prompt import SUFFIX
from langchain_openai import OpenAI, ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import SystemMessage
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

from tools import prepare_tools
from prompts import SYSTEM_MESSAGE


#@cl.cache
def get_memory():
    return ConversationBufferWindowMemory(memory_key="chat_history", k=1)


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-4", "gpt-4-0613", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o-mini-2024-07-18", "gpt-4o", "gpt-4o-2024-11-20"],
                initial_index=1,
            ),
            Switch(
                id="Streaming",
                label="OpenAI - Stream Tokens",
                initial=True
            ),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=0,
                min=0,
                max=2,
                step=0.1,
            )
        ]
    ).send()
    await setup_agent(settings)


@cl.on_settings_update
async def setup_agent(settings):
    print(f"Setup agent with following settings: {settings}")
    
    llm = ChatOpenAI(
        temperature=settings["Temperature"],
        streaming=settings["Streaming"],
        callbacks=[StreamingStdOutCallbackHandler()] if settings["Streaming"] else None,
        model=settings["Model"],
    )
    memory = get_memory()
    _SUFFIX = "Chat history:\n{chat_history}\n\n" + SUFFIX
    agent_kwargs = {
        "suffix": _SUFFIX,
        "system_message": SystemMessage(content=SYSTEM_MESSAGE),
        "input_variables": ["input", "agent_scratchpad", "chat_history"]
    }
    tools = prepare_tools(model_name=settings["Model"])

    agent = initialize_agent(
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        tools=tools,
        agent_kwargs=agent_kwargs,
        verbose=True,
        handle_parsing_errors=True,
        max_execution_time=None,
    )
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    res = await cl.make_async(agent.run)(
        input=message.content, callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True,)]
    )
    await cl.Message(content=res).send()
