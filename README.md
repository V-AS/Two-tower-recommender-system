# Project Name: Two-tower-recommentation-system

Developer Names: Yinying Huo

Date of project start: Jan 2025

This project is to fulfill the CAS 741 project.

The folders and files for this project are as follows:

- docs - Documentation for the project

- refs - Reference material used for the project, including papers

- src - Source code

- data - Folder for datasets

- test - code for system and unit tests

- output - Folder for model outputs

## Running the Project

Create a virtual environment and install the dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Additionally, install `faiss-cpu` or `faiss-gpu` if you have an NVIDIA GPU:

The `output` folder already contains the trained model. You can use it directly without training the model again.

You can type the following to use the user interface:
```bash
python src/user_interface.py
```

If you want to see more details of the model, you can enable the debug mode:
```bash 
python src/user_interface.py --debug
```

Train the model locally:
```bash
python src/main.py --mode train --epochs 3 --batch_size 10 --embedding_dim 32
```
