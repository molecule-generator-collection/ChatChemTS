import os
import subprocess
import yaml
from fastapi import FastAPI, Query

app = FastAPI()


@app.post("/run/")
async def run_chemtsv2(
    config_file_path: str = Query(..., description="Path to the config file")):
    if not config_file_path.startswith('shared_dir'):
        config_file_path = os.path.join('shared_dir', config_file_path)  # Shared volume in Docker compose
    with open(config_file_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)

    output_path = os.path.join(conf['output_dir'], f"result_C{conf['c_val']}.csv")
    if os.path.exists(output_path):
        return {"failed reason": f"[ERROR] {output_path} already exists. "
                                  "Need to change, move, or rename the output directory. "
                                  "Just inform an user of this error and the workaround. "
                                  "Do not try to solve the error yourself."}

    try:
        subprocess.run(["chemtsv2", "-c", config_file_path], check=True)
        return {"output_result_dir": conf['output_dir']}
    except subprocess.CalledProcessError as e:
        return {"failed reason": e}
    

@app.get("/")
def get_root():
    return {"message": "API for running ChemTSv2"}
