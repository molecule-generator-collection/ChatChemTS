from langchain.agents import AgentType, initialize_agent
from langchain.agents.structured_chat.prompt import SUFFIX
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

from tools import prepare_tools
from prompts import SUFFIX_MASTER


template = """Question: {question}

Answer: Let's think step by step."""

@cl.cache
def get_memory():
    return ConversationBufferMemory(memory_key="chat_history")

@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"],
                initial_index=1,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=0,
                min=0,
                max=2,
                step=0.1,
            ),
        ]
    ).send()
    await setup_agent(settings)


@cl.on_settings_update
async def setup_agent(settings):
    print(f"Setup agent with following settings: {settings}")
    
    llm = ChatOpenAI(
        temperature=settings["Temperature"],
        streaming=settings["Streaming"],
        model=settings["Model"],
    )
    memory = get_memory()
    #_SUFFIX = "Chat history:\n{chat_history}\n\n" + SUFFIX + SUFFIX_MASTER
    _SUFFIX = SUFFIX_MASTER
    tools = prepare_tools()

    agent = initialize_agent(
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        tools=tools,
        agent_kwargs={
            "suffix": _SUFFIX,
            #"input_variables": ["input", "agent_scratchpad", "chat_history"]
        },
        verbose=True,
        handle_parsing_errors=True,
    )
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    res = await cl.make_async(agent.run)(
        input=message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res).send()
