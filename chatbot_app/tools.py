import http.client
import os
from pathlib import Path
from typing import Optional, Type
import urllib.parse

from langchain.chains import ConversationChain
from langchain.chains.conversation.prompt import PROMPT
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.tools import Tool, BaseTool

# from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate

from prompts import PREFIX_REWARD, PREFIX_CONFIG
from util import prepare_chat_model


def prepare_tools(settings, verbose=True):
    return [
        create_reward_generator_tool(settings=settings),
        create_config_generator_tool(settings=settings),
    ] + [ChemTSv2ApiTool(), PredictionModelBuilder(), AnalysisTool(), WriteFileTool()]


class RewardGeneratorInput(BaseModel):
    query: str = Field(..., description="The string argument for this tool")


def create_reward_generator_tool(settings, verbose=False):
    agent_executor = create_python_agent(
        llm=prepare_chat_model(settings=settings),
        tool=PythonREPLTool(),
        verbose=verbose,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix=PREFIX_REWARD,
    )
    description = """This tool writes or generates a reward function of ChemTSv2, related to the query. 
        The input to this tool should be a `str` format, not `dict`.
        The input must be a instruction sentence, not a single word."""
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


def create_config_generator_tool(settings, verbose=False):
    llm_chain = ConversationChain(
        llm=prepare_chat_model(settings=settings),
        verbose=verbose,
        prompt=PromptTemplate.from_template(PREFIX_CONFIG) + PROMPT,
    )
    description = """This tool writes or generates a config file of ChemTSv2, related to the query.
        The input to this tool should be a `str` format, not `dict`.
        The input must be a instruction sentence, not a single word."""
    config_tool = Tool.from_function(
        func=llm_chain.run,
        name="config_generator",
        return_direct=True,
        description=description,
        args_schema=ConfigGeneratorInput,
    )
    return config_tool


class ChemTSv2ApiInput(BaseModel):
    config_file_path: str = Field(
        ..., description="Path to a config file to be loaded in ChemTSv2"
    )


class ChemTSv2ApiTool(BaseTool):
    name = "chemtsv2_api_tool"
    description = "This tool runs ChemTSv2 application."
    args_schema: Type[BaseModel] = ChemTSv2ApiInput
    return_direct: bool = False

    def __init__(self):
        super(ChemTSv2ApiTool, self).__init__()

    def _run(self, config_file_path: str) -> str:
        params = urllib.parse.urlencode({"config_file_path": config_file_path})
        headers = {
            "Accept": "application/json",
        }
        conn = http.client.HTTPConnection("api_chemtsv2", os.getenv("CHEMTS_PORT"))
        conn.request("POST", "/run/?" + params, headers=headers)
        response = conn.getresponse()
        status = response.status
        content = response.read().decode()
        conn.close()

        return f"{status} {content}"

    async def _arun(self, config_file_path: str) -> str:
        raise NotImplementedError


class PredictionModelBuilderInput:
    pass


class PredictionModelBuilder(BaseTool):
    name = "flaml_prediction_model_builder_tool"
    description = """This tool returns a URL of an application for building a prediction model using FLAML.
                     No arguments are required to use this tool."""
    return_direct: bool = True

    def __init__(self):
        super(PredictionModelBuilder, self).__init__()

    def _run(self) -> str:
        return f"Open [FLAML Model Builder](http://localhost:{os.getenv('MODEL_BUILDER_PORT')}) in your browser to create prediction models."

    async def _arun(self) -> str:
        raise NotImplementedError


class AnalysisToolInput:
    pass


class AnalysisTool(BaseTool):
    name = "molecule_generation_analysis_tool"
    description = """This tool returns a URL of an application for analyzing the molecule generation result.
                     No arguments are required to use this tool."""
    return_direct: bool = True

    def __init__(self):
        super(AnalysisTool, self).__init__()

    def _run(self) -> str:
        return f"Open [ChatChemTS Analysis App](http://localhost:{os.getenv('ANALYSIS_PORT')}) in your browser to analyze the molecule generation result."

    async def _arun(self) -> str:
        raise NotImplementedError


# The below two classes is based on and modified from the original repository at [source].
# [source]: https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/file_management/write.py
class WriteFileInput(BaseModel):
    """Input for WriteFileTool."""

    file_path: str = Field(..., description="name of file")
    text: str = Field(..., description="text to write to file")
    append: bool = Field(
        default=False, description="Whether to append to an existing file."
    )


class WriteFileTool(BaseFileToolMixin, BaseTool):
    """Tool that writes a file to disk."""

    name: str = "write_file"
    args_schema: Type[BaseModel] = WriteFileInput
    description: str = "Write file to disk"
    root_dir: str = "./shared_dir/"
    return_direct: bool = True

    def _run(
        self,
        file_path: str,
        text: str,
        append: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            write_path = self.get_relative_path(file_path)
        except FileValidationError:
            return INVALID_PATH_TEMPLATE.format(arg_name="file_path", value=file_path)
        try:
            write_path.parent.mkdir(exist_ok=True, parents=False)
            mode = "a" if append else "w"
            with write_path.open(mode, encoding="utf-8") as f:
                f.write(text)
            return f"The file has been successfully saved at `{write_path.relative_to(Path('/app'))}`."
        except Exception as e:
            return "Error: " + str(e)
