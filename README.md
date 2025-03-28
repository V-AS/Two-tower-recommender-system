# Project Name: Two-tower-recommentation-system

Developer Names: Yinying Huo

Date of project start: Jan 2025

This project is to fulfill the CAS 741 project.

The folders and files for this project are as follows:

- docs - Documentation for the project

- refs - Reference material used for the project, including papers

- src - Source code

- data - Folder for datasets

- test - Test cases

## Running the Project

Create a virtual environment and install the dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Train the model:
```bash
python src/main.py --mode train --epochs 3 --batch_size 10 --embedding_dim 32
```
