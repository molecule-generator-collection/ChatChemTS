import os
import subprocess
import yaml
from fastapi import FastAPI, Query

app = FastAPI()


@app.post("/run/")
async def run_chemtsv2(
    config_file_path: str = Query(..., description="Path to the config file")):
    config_file_path = os.path.join('files', config_file_path)  # Shared volume in Docker compose
    with open(config_file_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)

    try:
        subprocess.run(["chemtsv2", "-c", config_file_path], check=True)
        return {"output_result_dir": conf['output_dir']}
    except subprocess.CalledProcessError as e:
        return {"failed reason": e}
    

@app.get("/")
def get_root():
    return {"message": "Hello World"}
