# Character-Level Name Generation (Bigram + MLP)

This repository now contains two PyTorch notebook implementations for next-character name generation from `names.txt`:

1. A count-based and neuralized bigram model
2. A multilayer perceptron (MLP) model with character embeddings and context windows

## Project Structure

- `01_BigramModel/nextchar_bigram.ipynb`
- `02_MLPModel/MLPModel.ipynb`
- `names.txt` (training corpus)
- `requirements.txt`

## Dataset

The dataset in `names.txt` contains 32,033 names.

## Models

### 1) Bigram Model

Notebook: `01_BigramModel/nextchar_bigram.ipynb`

What it does:
- Builds a 27 x 27 character transition matrix (`.` + 26 lowercase letters)
- Estimates conditional next-character probabilities
- Trains a simple single-layer neural bigram model with one-hot inputs
- Samples new names by multinomial sampling until end token `.`

Reported notebook signals:
- Training examples: 228,146 bigrams
- Baseline average negative log-likelihood (example): ~3.6142
- Single update training loss print: ~2.4647

Sample generated names:
- cexze
- momasurailezityha
- konimittain
- llayn
- staiyaubrtthrigotai

### 2) MLP Model

Notebook: `02_MLPModel/MLPModel.ipynb`

What it does:
- Uses context length `block_size = 3`
- Splits dataset into train/dev/test (80/10/10)
- Learns character embeddings and a 1-hidden-layer MLP (`tanh`)
- Optimizes cross-entropy with minibatch SGD and learning-rate decay
- Generates names autoregressively from the learned distribution

Reported notebook signals:
- Split sizes:
  - Train: `torch.Size([182625, 3])`
  - Dev: `torch.Size([22655, 3])`
  - Test: `torch.Size([22866, 3])`
- Final training loss print: ~2.3375
- Dev loss: ~2.1936
- Test loss: ~2.1966

Sample generated names:
- ceri
- kayda
- samiyah
- molie
- aneil

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Notebooks

From repository root:

```bash
jupyter notebook 01_BigramModel/nextchar_bigram.ipynb
```

or

```bash
jupyter notebook 02_MLPModel/MLPModel.ipynb
```

Run cells top-to-bottom in each notebook.

## Notes

- `nextchar_bigram.ipynb` was moved from repository root into `01_BigramModel/`.
- `readme.txt` exists in the repo but is not used for project documentation.

