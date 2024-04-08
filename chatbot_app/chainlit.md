![chatchemts logo](public/logo_dark.png)

# Welcome to ChatChemTS!

ChatChemTS helps you to make reward and config files for AI-based molecule generators (currently support ChemTSv2).

## Available tools

When using ChatChemTS, you don't need to understand the following functions, but ChatChemTS uses them to enhance your molecule generation experience.

- `reward_generator` creates a reward script for AI-based molecule generators.
- `config_generator` creates a configuration for AI-based molecule generators.
- `chemtsv2_api` runs ChemTSv2 application using the provided configuration file.
- `flaml_prediction_model_builder` supports users to build prediction models using AutoML tool, FLAML.
- `analysis_tool` provides an analysis application to analyze the molecule generation result.
- `write_file_tool` provides a function to write a file.

## Demo prompt

Be sure to enter one-line at a time.

### STEP 1. Generate Reward file

```text
Write a reward function to maximize LogP value of molecules using ChemTSv2.
Save the above reward script as `reward_test.py`.
```

### STEP 2. Configuration generation

```text
Write a config file to generate 500 molecules using the reward file, `reward_test.py` with a Lipinski and a Radical filters. Output directory is set to `output_example`.
Save the above configuration as `config_test.yaml`.
```

### STEP 3. Run molecule generator

```text
Run ChemTSv2 using `config_test.yaml`.
```

### STEP 4. Analyze retult

```text
I want to analyze the molecule generation result.
```
