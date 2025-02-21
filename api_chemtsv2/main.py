from importlib import import_module
import os
import subprocess
import yaml
from fastapi import FastAPI, Query

app = FastAPI()


@app.post("/run/")
async def run_chemtsv2(
    config_file_path: str = Query(..., description="Path to the config file"),
):
    if not config_file_path.startswith("shared_dir"):
        config_file_path = os.path.join(
            "shared_dir", config_file_path
        )  # Shared volume in Docker compose
    with open(config_file_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)

    rs = conf["reward_setting"]
    if rs["reward_module"].split(".")[0] != "shared_dir":
        return {
            "failed reason": f"[ERROR] `reward_module` does not correctly set in the {config_file_path}. "
            f"Please check which reward file you specified in the prompt when preparing the configuration file. "
            "Files and directories specified by a user during interactions with ChatChemTS are stored under the `ChatChemTS/shared_dir` directory. "
            "Please also refer to the [tutorial](https://github.com/molecule-generator-collection/ChatChemTS/wiki/Tutorial) how to generate configuration file. "
            "Just inform an user of this error and the workaround. "
            "Do not try to solve the error yourself."
        }
    try:
        _ = import_module(rs["reward_module"])
        _ = getattr(import_module(rs["reward_module"]), rs["reward_class"])
    except ModuleNotFoundError as e:
        missing_file = f"{rs['reward_module'].split('.')[-1]}.py"
        return {
            "failed reason": f"[ERROR] {missing_file} is not found in the `share_dir` "
            f"Please check the {missing_file} exists in `shared_dir`. "
            "Files and directories specified by a user during interactions with ChatChemTS are stored under the `ChatChemTS/shared_dir` directory. "
            "Please also refer to the [tutorial](https://github.com/molecule-generator-collection/ChatChemTS/wiki/Tutorial) how to generate configuration file. "
            "Just inform an user of this error and the workaround. "
            "Do not try to solve the error yourself."
        }
    except AttributeError as e:
        return {
            "failed reason": f"[ERROR] `reward_class` does not correctly set in the {config_file_path}. "
            f"Please check which reward class you specified in the prompt when preparing the configuration file. "
            "Files and directories specified by a user during interactions with ChatChemTS are stored under the `ChatChemTS/shared_dir` directory. "
            "Please also refer to the [tutorial](https://github.com/molecule-generator-collection/ChatChemTS/wiki/Tutorial) how to generate configuration file. "
            "Just inform an user of this error and the workaround. "
            "Do not try to solve the error yourself."
        }

    output_path = os.path.join(conf["output_dir"], f"result_C{conf['c_val']}.csv")
    if os.path.exists(output_path):
        return {
            "failed reason": f"[ERROR] {output_path} already exists. "
            "Need to change, move, or rename the output directory. "
            "Just inform an user of this error and the workaround. "
            "Do not try to solve the error yourself."
        }

    try:
        subprocess.run(["chemtsv2", "-c", config_file_path], check=True)
        return {"output_result_dir": conf["output_dir"]}
    except subprocess.CalledProcessError as e:
        return {"failed reason": e}


@app.get("/")
def get_root():
    return {"message": "API for running ChemTSv2"}
