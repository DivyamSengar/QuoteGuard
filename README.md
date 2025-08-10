# QuoteGuard: Speaker Verification Transformer

QuoteGuard is a project that utilizes a custom-built transformer model to verify the speakers of political quotes and speeches. It leverages advanced techniques like sparse attention, regularization, and cross-validation to achieve high accuracy in classifying text to its author.



## 📜 Description

This project contains two main components:

1.  **A Transformer-Classifier (`part1`)**: An encoder-based transformer model designed for text classification. It is trained to identify the speaker of a given text (quote or speech) from a predefined set of political figures. The model's performance is rigorously evaluated using k-fold cross-validation to ensure the stability and reliability of the accuracy metric.

2.  **A Language Model (`part2`)**: A decoder-based transformer trained for language modeling. Its purpose is to understand and generate text that mimics the style of the training data.

The project also includes an implementation of a **sparse attention mechanism** (`part3`), allowing for more efficient processing and potentially improved performance by focusing the model's attention on more relevant parts of the input text.

## ✨ Features

* **Custom Transformer Architecture**: Built from scratch using PyTorch.
* **Speaker Verification**: Classifies text to one of three political speakers.
* **Sparse Attention**: Implements a sparse attention mechanism to explore performance improvements.
* **Regularization**: Uses Dropout in the feed-forward layers to prevent overfitting.
* **Cross-Validation**: Employs k-fold cross-validation for robust accuracy assessment.

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* PyTorch
* NLTK
* scikit-learn

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repo-link>
    cd QuoteGuard
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download the NLTK `punkt` tokenizer models:
    ```python
    import nltk
    nltk.download('punkt')
    ```

### Running the Models

You can run the different parts of the project using the command line.

#### Part 1: Run the Transformer-Classifier

To train and evaluate the classifier on the test set without cross-validation:
```bash
python main.py part1
```

To run with **5-fold cross-validation**:
```bash
python main.py part1 --folds 5
```

#### Part 2: Run the Language Model

To train the decoder-based language model:
```bash
python main.py part2
```

#### Part 3: Experiment with Sparse Attention

To run the classifier and language model with a custom sparse attention pattern:
```bash
python main.py part3
```
You will be prompted to enter a sparsity pattern value (e.g., 3). This will run `part1` (with cross-validation if specified) and `part2` using the sparse attention mechanism.