import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.prompt import PROMPT
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.tools import Tool
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from pydantic.v1 import BaseModel, Field

from prompts import PREFIX_REWARD, PREFIX_CONFIG


def prepare_tools(verbose=True):
    root_dir = "./chatbot_data"
    os.makedirs(root_dir, exist_ok=True)
    file_tools = FileManagementToolkit(root_dir=root_dir, 
                                       selected_tools=[
                                           'read_file',
                                           'write_file',
                                           'list_directory',
                                           'move_file']).get_tools()

    return [
        create_reward_generator_tool(),
        create_config_generator_tool(),
    ] + file_tools


class RewardGeneratorInput(BaseModel):
    query: str = Field(description="The string argument for this tool")
    

def create_reward_generator_tool(verbose=True):
    agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613", streaming=True),
        tool=PythonREPLTool(),
        verbose=verbose,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix=PREFIX_REWARD,
    )
    description = """Usefule for when you need to generate reward functions of molecule generators, such as ChemTSv2, related to the query. 
        The input to this tool should be a `str` format, not `dict`.
        The input should be a phrase, not a single word."""
    reward_tool = Tool.from_function(
        func=agent_executor.run,
        name="reward_generator",
        return_direct=True,
        description=description,
        args_schema=RewardGeneratorInput,
    )
    return reward_tool


class ConfigGeneratorInput(BaseModel):
    query: str = Field(description="The string argument for this tool")


def create_config_generator_tool(verbose=True):
    llm_chain = ConversationChain(
        llm=ChatOpenAI(temperature=0, streaming=True),
        verbose=verbose,
        prompt=PromptTemplate.from_template(PREFIX_CONFIG)+PROMPT,
    )
    description = """Usefule for when you need to write config files of molecule generators, such as ChemTSv2, related to the query. 
        The input to this tool should be a `str` format, not `dict`.
        The input should be a phrase, not a single word."""
    config_tool = Tool.from_function(
        func=llm_chain.run,
        name="config_generator",
        return_direct=True,
        description=description,
        args_schema=ConfigGeneratorInput,
    )
    return config_tool

