import torch
import argparse
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import os
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import *
import torch.optim as optim
from utilities import *
from sklearn.model_selection import KFold
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match
the numbers mentioned in the assignment description """
batch_size = 16
block_size = 32
learning_rate = 1e-3
n_embd = 64
n_head = 2
n_layer = 4
eval_interval = 100
max_iters = 500
eval_iters = 200

n_input = 64
n_hidden = 100
n_output = 3
epochs_CLS = 15

def load_texts(directory):
    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    data, labels = zip(*batch)
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, attn_map = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
    accuracy = (100 * total_correct / total_samples)
    classifier.train()
    return accuracy

def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    decoderLMmodel.eval()
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel.lossCalc(X, Y)
        losses.append(loss.item())
        if len(losses) >= eval_iters:
            break
    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()
    decoderLMmodel.train()
    return perplexity

def main():
    args = parse_arguments()

    if args.part == 'part1':
        part1_code(-1, args.folds)
    elif args.part == 'part2':
        part2_code(-1)
    elif args.part == 'part3':
        spars_pattern = int(input("Enter sparsity pattern value (e.g., 3): "))
        part1_code(spars_pattern, args.folds)
        part2_code(spars_pattern)

def part1_code(sparsity_pattern, n_folds):
    print("Loading data and creating tokenizer...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))

    full_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    test_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)

    if n_folds > 1:
        print(f"--- Running {n_folds}-Fold Cross-Validation ---")
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_accuracies = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
            print(f"--- Fold {fold+1}/{n_folds} ---")
            train_subsampler = Subset(full_dataset, train_ids)
            val_subsampler = Subset(full_dataset, val_ids)

            train_loader = DataLoader(train_subsampler, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)

            encoder = EncodingTransformer(tokenizer.vocab_size, n_embd, block_size, n_layer, n_head, n_input, n_hidden, n_output, sparsity_pattern).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

            for epoch in range(epochs_CLS):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits, _ = encoder(xb)
                    loss = loss_fn(logits, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            accuracy = compute_classifier_accuracy(encoder, val_loader)
            print(f"Fold {fold+1} Validation Accuracy: {accuracy:.2f}%")
            fold_accuracies.append(accuracy)

        avg_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        print(f"\n--- Cross-Validation Summary ---")
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
        print(f"Standard Deviation: {std_accuracy:.2f}%")
    
    # Standard training on the full training set and evaluation on the test set
    print("\n--- Training on full data and evaluating on test set ---")
    train_loader = DataLoader(full_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    encoder = EncodingTransformer(tokenizer.vocab_size, n_embd, block_size, n_layer, n_head, n_input, n_hidden,n_output, sparsity_pattern).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    for epoch in range(epochs_CLS):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = encoder(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = compute_classifier_accuracy(encoder, test_loader)
        print(f'Epoch {epoch+1}, Test Accuracy: {accuracy:.2f}%')


def part2_code(sparsity_pattern):
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
    with open("speechesdataset/test_LM_obama.txt", 'r', encoding='utf-8') as f:
        obamaText = f.read()
    with open("speechesdataset/test_LM_hbush.txt", 'r', encoding='utf-8') as f:
        hBushText = f.read()
    with open("speechesdataset/test_LM_wbush.txt", 'r', encoding='utf-8') as f:
        wBushText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    test_LM_obama_dataset = LanguageModelingDataset(tokenizer, obamaText,  block_size)
    test_LM_hbush_dataset = LanguageModelingDataset(tokenizer, hBushText,  block_size)
    test_LM_wbush_dataset = LanguageModelingDataset(tokenizer, wBushText,  block_size)
    test_LM_obama_loader = DataLoader(test_LM_obama_dataset, batch_size=batch_size, shuffle=True)
    test_LM_hbush_loader = DataLoader(test_LM_hbush_dataset, batch_size=batch_size, shuffle=True)
    test_LM_wbush_loader = DataLoader(test_LM_wbush_dataset, batch_size=batch_size, shuffle=True)

    decoder = DecodingTransformer(tokenizer.vocab_size, n_embd, block_size, n_layer, n_head, n_hidden, sparsity_pattern)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        logits, attn_map = decoder(xb)
        a, b, c = logits.size()
        logits = logits.view(a*b, c)
        yb = yb.view(a*b)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Number of Parameters: ", sum(p.numel() for p in decoder.parameters() if p.requires_grad))
    print("Vocabulary size is", tokenizer.vocab_size)
    for i in range(100, 501, 100):
        print('train perplexity, at step ',i,'is ', compute_perplexity(decoder, train_LM_loader, i))
    print('obama perplexity at step 500, ', compute_perplexity(decoder, test_LM_obama_loader, 500))
    print('hbush perplexity at step 500, ', compute_perplexity(decoder, test_LM_hbush_loader, 500))
    print('wbush perplexity at step 500, ', compute_perplexity(decoder, test_LM_wbush_loader, 500))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run different parts of the transformer model training.")
    parser.add_argument('part', choices=['part1', 'part2', 'part3'],
                        help="Specify which task to run: 'part1', 'part2', or 'part3'.")
    parser.add_argument('--folds', type=int, default=1, help="Number of folds for cross-validation in part1.")
    return parser.parse_args()

if __name__ == "__main__":
    main()