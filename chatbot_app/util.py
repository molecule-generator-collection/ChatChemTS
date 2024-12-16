from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def prepare_chat_model(settings):
    chat_model = ChatOpenAI(
        temperature=settings["Temperature"],
        streaming=settings["Streaming"],
        callbacks=[StreamingStdOutCallbackHandler()] if settings["Streaming"] else None,
        model=settings["Model"],
    )
    return chat_model
