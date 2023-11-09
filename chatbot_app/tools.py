import os
import http.client
import urllib.parse
from typing import Type

from langchain.chains import ConversationChain
from langchain.chains.conversation.prompt import PROMPT
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.tools import Tool, BaseTool
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from pydantic.v1 import BaseModel, Field

from prompts import PREFIX_REWARD, PREFIX_CONFIG


def prepare_tools(verbose=True):
    root_dir = "."
    os.makedirs(root_dir, exist_ok=True)
    file_tools = FileManagementToolkit(root_dir=root_dir, 
                                       selected_tools=[
                                           'read_file',
                                           'write_file',
                                           'list_directory',]).get_tools()

    return [
        create_reward_generator_tool(),
        create_config_generator_tool(),
    ] + file_tools + [ChemTSv2ApiTool()]


class RewardGeneratorInput(BaseModel):
    query: str = Field(..., description="The string argument for this tool")
    

def create_reward_generator_tool(verbose=False):
    agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613", streaming=False),
        tool=PythonREPLTool(),
        verbose=verbose,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix=PREFIX_REWARD,
    )
    description = """This tool writes or generates a reward function of ChemTSv2, related to the query. 
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
    query: str = Field(..., description="The string argument for this tool")


def create_config_generator_tool(verbose=False):
    llm_chain = ConversationChain(
        llm=ChatOpenAI(temperature=0, streaming=False),
        verbose=verbose,
        prompt=PromptTemplate.from_template(PREFIX_CONFIG)+PROMPT,
    )
    description = """This tool writes / generates a config file of ChemTSv2, related to the query. 
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


class ChemTSv2ApiInput(BaseModel):
    config_file_path: str = Field(..., description="Path to a config file to be loaded in ChemTSv2")


class ChemTSv2ApiTool(BaseTool):
    name = "chemtsv2_api_tool"
    description = "This tool runs ChemTSv2 application."
    args_schema: Type[BaseModel] = ChemTSv2ApiInput
    
    def __init__(self):
        super(ChemTSv2ApiTool, self).__init__()
    
    def _run(self, config_file_path: str) -> str:
        params = urllib.parse.urlencode({
            'config_file_path': config_file_path})
        headers = {
            'Accept': 'application/json',
            }
        #conn = http.client.HTTPConnection('localhost', 8001)
        conn = http.client.HTTPConnection('api_chemtsv2', 8001)
        conn.request('POST', '/run/?' + params, headers=headers)
        response = conn.getresponse()
        status = response.status
        content = response.read().decode()
        conn.close()

        return f"{status} {content}"
    
    async def _arun(self, config_file_path: str) -> str:
        raise NotImplementedError
    