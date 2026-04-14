# Next Character Bigram Model

A simple implementation of a bigram model for generating names using PyTorch. This project demonstrates character-level language modeling by learning the probability distribution of character transitions from a dataset of names.

## Overview

The model works by:
1. Reading a list of names from `names.txt`
2. Building a 27x27 transition matrix (26 letters + '.' for start/end)
3. Using the matrix to generate new names by sampling from the probability distributions

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch (torch)
- NumPy
- Matplotlib
- Jupyter Notebook

## Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook nextchar_bigram.ipynb
   ```

2. Run the cells in order to:
   - Load and explore the names dataset
   - Build the bigram frequency matrix
   - Visualize the transition probabilities
   - Generate new names

## Dataset

The `names.txt` file contains approximately 32,000 names used for training the model.

## Example Output

The model can generate names like:
- junide
- janasah
- p
- cony
- a
- nn
- kohin
- rosen
- chendra
- ahdra

