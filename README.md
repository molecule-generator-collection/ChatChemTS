# ChatChemTS

[![DOI](https://zenodo.org/badge/706037831.svg)](https://doi.org/10.5281/zenodo.14498884)

ChatChemTS is an open-source LLM-based web application for using an AI-based molecule generator, [ChemTSv2](https://github.com/molecule-generator-collection/ChemTSv2).

<div align="center">
  <img src=img/toc.png width="70%">
</div>

## Demo Videos

<img src="https://github.com/molecule-generator-collection/ChatChemTS/assets/29348731/50049eb6-d2c1-4f74-9830-f6c98ccf9ff8" width="32%"> <img src="https://github.com/molecule-generator-collection/ChatChemTS/assets/29348731/a5cd8614-030b-4386-83bf-cc06508bd158" width="32%"> <img src="https://github.com/molecule-generator-collection/ChatChemTS/assets/29348731/04ed00bc-daf7-43fa-bae1-09635871e6d6" width="32%">

:arrow_left: ChatChemTS   :arrow_up: Analysis tool   :arrow_right: Prediction model builder

## How to Start

### Confirmed OS & CPU Architecture

The below OS with CPU architecture is confirmed.

- Linux
  - `Ubuntu` (22.04.2 LTS,  AMD EPYC 7443P `x86_64`)
- macOS
  - `Ventura` (13.6.8, Intel Core i9 `x86_64`; 13.7.1, Apple M2 `arm64`)
  - `Sonoma` (14.4.1, Apple M2 `arm64`)
- Windows (requires WSL2. Detailes in the software requirement section)
  - `11 Pro` (23H2, Intel Core i9-11900K `amd64`)
  - `11 Home` (23H2, AMD Ryzen 7 5700U `amd64`)
  - `10 Pro` (22H2, Intel Core i7-10870H `amd64`)

### Software Requirement

- Docker: >= version 24
- Git
- WSL2 (Windows Only; Please refer to the official document: [How to install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install))

>[!NOTE]
>For Mac and Windows users, Docker Desktop is easy way to install Docker into your laptop.
>Refer to the following links:
>
>- Windows: [Docker Desktop WSL 2 backend on Windows](https://docs.docker.com/desktop/features/wsl/)
>- Mac: [Install Docker Desktop on Mac](https://docs.docker.com/desktop/setup/install/mac-install/)

>[!CAUTION]
>If you use Docker Desktop, you must agree the [Docker Desktop license agreement](https://docs.docker.com/subscription/desktop-license/).
>
>- Docker Desktop is free for small businesses (fewer than 250 employees AND less than $10 million in annual revenue), personal use, education, and non-commercial open source projects.
>- Otherwise, it requires a paid subscription for professional use.
>
>ChatChemTS itself is a non-commercial open source project and licensed under the MIT License.

### Installation

At first, open `Terminal (Mac & Linux)` or `PowerShell (Windows)`.  

>[!IMPORTANT]
>Before proceeding the next step, make sure Docker is properly started.
>
>- On `Windows` or `macOS`: Veryfy that Docker Desktop is running.
>- On `macOS with Apple Silicon`: Must disable `Use Rosetta for x86_64/amd64 emulation on Apple Silicon` option if the option is enabled and restart your Mac to ensure that the change takes effect. ref. [Change your Docker Desktop settings](https://docs.docker.com/desktop/settings-and-maintenance/settings/)
>- On `Windows`: ensure you switch to the WSL2 environment with the following command: `wsl --distribution Ubuntu`.
>Note that the distribution may not be named `Ubuntu`.
>You can check which distribution is actually installed by running `wsl --list`.
>- On `Linux`: Ensure the Docker daemon is running.

#### Local Laptop

```bash
git clone https://github.com/molecule-generator-collection/ChatChemTS.git
cd ChatChemTS
# You must set your OpenAI API key in `.env` file.
# The `.env` file is located in the root of the ChatChemTS repository.
bash ./manage_app.sh deploy
```

>[!NOTE]
> If you encounter the error related to `IncompleteRead Error`, try to rerun `bash ./manage_app.sh deploy`.

#### Remote Server

If you want to deploy ChatChemTS on a remote server, you will need to set up port forwarding for ports 8000 to 8003 to connect your local laptop to the remote server as follows.

```bash
ssh -L 8000:localhost:8000 -L 8001:localhost:8001 -L 8002:localhost:8002 -L 8003:localhost:8003 YOUR_REMOTE_SERVER
# Follow the same steps as in procedure `Local laptop`.
```

When ChatChemTS is successfully deployed, you can see the below messages.

```bash
 ✔ Network chatchemts_chatchemts_network  Created                                             0.1s 
 ✔ Volume "chatchemts_shared_volume"      Created                                             0.0s 
 ✔ Container chatchemts-api_chemtsv2-1    Started                                             0.7s 
 ✔ Container chatchemts-model_builder-1   Started                                             0.7s 
 ✔ Container chatchemts-analysis-1        Started                                             0.7s 
 ✔ Container chatchemts-chatbot-1         Started                                             0.0s 
ChatChemTS is now running! Access it at http://localhost:8000
```

>[!TIP]
>If you want to change the port numbers used for deployment, you can freely edit the port numbers in the `.env` file.
>If you are deploying on a remote server, make sure to update the local forwarding port numbers accordingly to match the changes.
>This is particularly useful for avoiding errors that may occur if applications like Jupyter Notebook are already using any or all of the ports from 8000 to 8003.

## How to Use

### Open ChatChemTS App
Once ChatChemTS is running, open [http://localhost:8000](http://localhost:8000) in your web browser.  
>[!TIP]
>- For first-time users, please check out the [Tutorial](https://github.com/molecule-generator-collection/ChatChemTS/wiki/Tutorial) page for guidance.
>- Files and directories specified by a user during interactions with ChatChemTS are stored under the `ChatChemTS/shared_dir` directory.

### Managing the Application

Below is a simple guide on how to manage ChatChemTS Docker environments using `manage_app.sh`.
This script lets you easily set up, run, stop, and clean up the Docker environment without dealing directly with Docker commands.

#### Basic Commands

```bash
bash ./manage_app.sh {deploy|start|stop|clean|help}
```

- `deploy`: Sets up and starts the application using docker-compose.
- `start`: (Re)starts the application if it’s already set up.
- `stop`: Stops the application (but keeps the setup files).
- `clean`: Completely removes the application setup, including any data created.
- `help`: Shows a quick reference for these commands.

#### Examples

- To start using the application for the first time:

```bash
bash ./manage_app.sh deploy
```

After this, open [http://localhost:8000](http://localhost:8000) in your web browser.

- To start using the application:

```bash
bash ./manage_app.sh start
```

- To stop the application (for example, when you are done using it):

```bash
bash ./manage_app.sh stop
```

- To remove everything and return to a clean state (note: the contents of `shared_dir` will not be affected.):

```bash
bash ./manage_app.sh clean
```

## For Developer

Please refer to the [GitHub Wiki](https://github.com/molecule-generator-collection/ChatChemTS/wiki/Developer-Document).

## Package Dependency

>[!NOTE]
>ChatChemTS is automatically deployed using Docker Compose (commands are written in `manage_app.sh`), thus you don't need to prepare its computational environment manually.

<details>
  <summary>Click to show/hide requirements</summary>

- python: 3.11
- openai: 1.9.0
- langchain: 0.11
- chainlit: 1.0.101
- rdkit: 2023.9.1
- streamlit: 1.30.0
- pandas: 2.1.4
- flaml[automl]: 2.1.1
- matplotlib:
- scikit-learn:
- chembl_webresource_client:
- fastapi: 0.109.0
- chemtsv2: 1.0.2
- mols2grid

</details>

## How to Cite

```bibtex
@article{Ishida2024,
  title = {Large Language Models Open New Way of AI-Assisted Molecule Design for Chemists},
  url = {http://dx.doi.org/10.26434/chemrxiv-2024-1p82f},
  DOI = {10.26434/chemrxiv-2024-1p82f},
  journal = {ChemRxiv},
  author = {Ishida, Shoichi and Sato, Tomohiro and Honma, Teruki and Terayama, Kei},
  year = {2024},
  month = apr 
}
```

## License

This package is distributed under the MIT License.
