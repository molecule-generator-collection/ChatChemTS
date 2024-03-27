# ChatChemTS

ChatChemTS is an open-source LLM-based GUI application for using an AI-based molecule generator, ChemTSv2. 

## Quick Start

### Hardware Requirement

- CPU: x86_64, amd64

>[!NOTE]
>arm architecture, e.g., Apple Silicon, is not currently supported

### Software Requirement

- Docker: >= version 24
- Git

### How to deploy

```bash
git clone git@github.com:sishida21/ChatMolGen.git
cd ChatMolGen
# must set your OpenAI API key in `.env` file.
./deploy.sh deploy
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
