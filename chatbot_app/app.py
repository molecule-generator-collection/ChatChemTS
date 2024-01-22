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
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-4"],
                initial_index=2,
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
            ),
            Slider(
                id="MaxExecutionTime",
                label="OpenAI - Max Execution Time",
                initial=300,
                min=60,
                max=600,
                step=10,
            )
        ]
    ).send()
    await setup_agent(settings)

    #res = await cl.AskActionMessage(
    #    content="Save this reward file?",
    #    actions=[
    #        cl.Action(name="save", value="save", label="✅ Save it"),
    #        cl.Action(name="continue", value="continue", label="▶️ Continue chatting")
    #    ],
    #    timeout=90, 
    #).send()

    #if res and res.get("value") == "save":
    #    filename = await cl.AskUserMessage(
    #        content="Type a reward file name: ",
    #        timeout=300,
    #    ).send()
    #    if filename:
    #        await cl.Message(
    #            content=f"Reward file is saved at /mnt/data/{filename['content']}"
    #        ).send()
    

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
    #_SUFFIX = SUFFIX
    agent_kwargs = {
        "suffix": _SUFFIX,
        "system_message": SystemMessage(content=SYSTEM_MESSAGE),
        "input_variables": ["input", "agent_scratchpad", "chat_history"]
    }
    tools = prepare_tools()

    agent = initialize_agent(
        llm=llm,
        #agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        tools=tools,
        agent_kwargs=agent_kwargs,
        verbose=True,
        handle_parsing_errors=True,
        max_execution_time=settings["MaxExecutionTime"],
    )
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    res = await cl.make_async(agent.run)(
        input=message.content, callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True,)]
    )
    await cl.Message(content=res).send()
