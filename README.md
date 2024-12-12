# ChatChemTS
ChatChemTS is an open-source LLM-based web application for using an AI-based molecule generator, [ChemTSv2](https://github.com/molecule-generator-collection/ChemTSv2). 

<div align="center">
  <img src=img/toc.png width="70%">
</div>

## Demo Videos

<img src="https://github.com/molecule-generator-collection/ChatChemTS/assets/29348731/50049eb6-d2c1-4f74-9830-f6c98ccf9ff8" width="32%"> <img src="https://github.com/molecule-generator-collection/ChatChemTS/assets/29348731/a5cd8614-030b-4386-83bf-cc06508bd158" width="32%"> <img src="https://github.com/molecule-generator-collection/ChatChemTS/assets/29348731/04ed00bc-daf7-43fa-bae1-09635871e6d6" width="32%">

- left: ChatChemTS
- middle: Analysis tool
- right: Prediction model builder

## How to Start

### Confirmed Operating System & CPU Architecture

The below OS with CPU architecture is confirmed.
- Linux
  - `Ubuntu` (22.04.2 LTS,  AMD EPYC 7443P `x86_64`)
- macOS
  - `Ventura` (13.6.8, Intel Core i9 `x86_64`; 13.7.1, Apple M2 `arm64`)
  - `Sonoma` (14.4.1, Apple M2 `arm64`)
- Windows (requires WSL2. Detailes in the software requirement section)
  - `11 Pro` (23H2, Intel Core i9-11900K `amd64`)

### Software Requirement

- Docker: >= version 24
- Git
- WSL2 (Windows Only; Please refer to the official document: [How to install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install))

>[!NOTE]
>For Mac and Windows users, Docker Desktop is easy way to install Docker into your laptop.
>Refer to the following links:
- Windows: [Docker Desktop WSL 2 backend on Windows](https://docs.docker.com/desktop/features/wsl/)
- Mac: [Install Docker Desktop on Mac](https://docs.docker.com/desktop/setup/install/mac-install/)

### Installation

At first, open Terminal (Mac & Linux) or PowerShell (Windows).  

If you are using Windows, ensure you switch to the WSL2 environment with the following command:
```powershell
wsl --distribution Ubuntu
```
Note that the distribution may not be named `Ubuntu`.
You can check which distribution is actually installed by running `wsl --list`.

>[!NOTE]
>Before proceeding the next step, make sure Docker is properly started.
- On Windows or macOS: Veryfy that Docker Desktop is running.
- On Linux: Ensure the Docker daemon is running.

#### Local laptop
```bash
git clone git@github.com:molecule-generator-collection/ChatChemTS.git
cd ChatChemTS
# You must set your OpenAI API key in `.env` file.
# The `.env` file is located in the root of the ChatChemTS repository.
./deploy.sh deploy
```

#### Remote server

If you want to deploy ChatChemTS on a remote server, you will need to set up port forwarding for ports 8000 to 8003 to connect your local laptop to the remote server as follows.
```bash
ssh -L 8000:localhost:8000 -L 8001:localhost:8001 -L 8002:localhost:8002 -L 8003:localhost:8003 YOUR_REMOTE_SERVER
# Follow the same steps as in procedure `Local laptop`.
```

When ChatChemTS is successfully deployed, you can access it at [http://localhost:8000](http://localhost:8000). 

## Package dependency

>[!NOTE]
>ChatChemTS is automatically deployed using Docker Compose (commands are written in deploy.sh), thus you don't need to prepare its computational environment manually.

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
