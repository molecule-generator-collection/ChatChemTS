# ChatChemTS
ChatChemTS is an open-source LLM-based web application for using an AI-based molecule generator, ChemTSv2. 

<div align="center">
  <img src=img/toc.png width="60%">
</div>

## Demo Videos

<img src="https://github.com/molecule-generator-collection/ChatChemTS/assets/29348731/50049eb6-d2c1-4f74-9830-f6c98ccf9ff8" width="32%"> <img src="https://github.com/molecule-generator-collection/ChatChemTS/assets/29348731/a5cd8614-030b-4386-83bf-cc06508bd158" width="32%"> <img src="https://github.com/molecule-generator-collection/ChatChemTS/assets/29348731/04ed00bc-daf7-43fa-bae1-09635871e6d6" width="32%">

- left: ChatChemTS
- middle: Analysis tool
- right: FLAML prediction model builder

## Quick Start

### Hardware Requirement

- CPU: x86_64, amd64

>[!NOTE]
>arm architecture, e.g., Apple Silicon, is not currently supported

### Software Requirement

- Docker: >= version 24
- Git

### How to deploy

#### Local laptop
```bash
git clone git@github.com:sishida21/ChatChemTS.git
cd ChatChemTS
# must set your OpenAI API key in `.env` file.
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


## How to Cite

## License

This package is distributed under the MIT License.
