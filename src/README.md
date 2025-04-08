# TTE RecSys Source Code

The folders and files for this project are as follows:
```
src/
├── hardware/
│   └── system_interface.py        # Hardware-hiding module Handles saving models to specific folders and loading them
├── modules/
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── model_training.py          # Uses Stochastic Gradient Descent to train the model
│   ├── embedding_generation.py    # Generates embeddings using the trained DNNs
│   ├── recommendation.py          # Produces recommended items using ANN search and embedding dot products
│   ├── neural_network.py          # Defines the DNNs for users and items
│   ├── ann_search.py              # Provides an interface for Approximate Nearest Neighbor search
│   └── vector_operations.py       # Provides efficient vector operations (dot product and normalization)
├── utils/
│   └── data_preprocessing.ipynb   # Notebook to merge the data from 3 separate csv files into one csv file.
├── main.py                        # Main execution script
└── user_interface.py
```
