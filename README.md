# Project Name: Two-tower-recommentation-system

A recommendation system based on the Two-Tower Embeddings architecture, developed for personalized book recommendations.

Developer Names: Yinying Huo

Date of project start: Jan 2025

This project is to fulfill the CAS 741 project.

## Overview

This project implements a recommendation system using a Two-Tower Embedding (TTE) model architecture. The system maps user and item features into a shared embedding space and uses Approximate Nearest Neighbor (ANN) search with FAISS to efficiently retrieve relevant items. This approach enables fast and scalable personalized recommendations even with large item catalogs.

## Directory Structure

```bash
Two-tower-recommender-system/
├── docs/                 # Documentation (SRS, MG, VnV, etc.)
├── refs/                 # Reference material and papers
├── src/                  # Source code
│   ├── hardware/         # Hardware-hiding module
│   ├── modules/          # Core modules (neural network, ANN search, etc.)
│   ├── utils/            # Utility functions and configuration
│   ├── main.py           # Primary execution script
│   └── user_interface.py # Terminal-based user interface
├── data/                 # Data files
│   ├── raw/              # Original dataset
│   └── processed/        # Preprocessed data for model training
├── tests/                # System and unit tests
│   ├── system/           # System test scripts
│   └── unit/             # Unit test scripts
├── output/               # Trained models and embeddings
└── requirements.txt      # Project dependencies
```

## Prerequisites

- Python 3.9 or higher
- PyTorch 1.9 or higher
- FAISS for vector similarity search
- Other dependencies as listed in `requirements.txt`

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/V-AS/Two-tower-recommender-system.git
  cd Two-tower-recommender-system
  ```

2. Create and activate a virtual environment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate   # On Windows: .venv\Scripts\activate
  ```
3. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

4. Install FAISS:
  ```bash
  # For CPU-only systems:
  pip install faiss-cpu
  
  # For systems with NVIDIA GPU:
  pip install faiss-gpu
  ```

## Usage

The `output` folder contains pre-trained models, so you can immediately use the recommendation system:

```bash
python src/user_interface.py
```

or

```bash
python src/user_interface.py --debug
```

## Training the Model locally
```bash
python src/main.py --mode train --epochs 3 --batch_size 10 --embedding_dim 32
```

**Note**: The training process takes approximately 5 minutes with the recommended parameters, which have been optimized for the current dataset.

## Run a specific test file locally
```bash
pytest tests/unit/test_embedding_generation.py
```





