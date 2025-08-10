import torch
import argparse
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import *
import torch.optim as optim
from utilities import *
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

spars_pattern = -1 ## A new hyperparameter (for part 3), dictating whether the model uses sparseAttention ## (via blocks) or not (if the number is negative) 

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
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
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel.lossCalc(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        # total_loss += loss.item()
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():
    args = parse_arguments()

    if args.part == 'part1':
        part1_code(-1)

    if args.part == 'part2':
        part2_code(-1)

    if args.part == 'part3':
        #Architectural Exploration part, using SparseAttentionHeads 
        spars_pattern = int(input("Please enter a positive number for a new sparsity pattern. This number will determine the sparsity of the SparseSelfAttentionHead.\nBoth the encoder and decoder will be run with this sparsity value, so comparisons can be made with part1/part2 results.\nI suggest choosing 3\n")) 
        part1_code(spars_pattern)
        part2_code(spars_pattern)

        #Performance Improvement part
        #Run part1 (manually or via the part1 commandline entry) after setting numHidden = 125, learning rate = 5e-4, and uncommenting the blocks in the part1 code regarding the optimizer.
        # part1_code(-1)


def part1_code(sparsity_pattern):
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    encoder = EncodingTransformer(tokenizer.vocab_size, n_embd, block_size, n_layer, n_head, n_input, n_hidden,n_output, sparsity_pattern).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    # new_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate) ## for performance improvement
    # optimizer = new_optimizer  ## for performance improvement
    # utility_class = Utilities(tokenizer, encoder)
    # for the classification  task, you will train for a fixed number of epochs like this:
    print("Number of Parameters: ", sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    print("Vocabulary size is", tokenizer.vocab_size)
    for epoch in range(epochs_CLS):
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            # CLS training code here
            logits, attn_map = encoder(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if epoch == 0: 
        #     utility_class.sanity_check("I am running for president.", block_size)
        accuracy = compute_classifier_accuracy(encoder, test_CLS_loader)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy}%')

def part2_code(sparsity_pattern):
        print("Loading data and creating tokenizer ...")
        texts = load_texts('speechesdataset')
        tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data

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
        # utility_class = Utilities(tokenizer, decoder)
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            # LM training code here
            logits, attn_map = decoder(xb)
            #reshape the model output and target output to properly calculate the loss:
            a, b, c = logits.size()
            logits = logits.view(a*b, c)
            yb = yb.view(a*b)
            loss = loss_fn(logits, yb)
            # if (i == 0):
            #     utility_class.sanity_check("I am running for president.", block_size)
            # Backpropagation
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
    return parser.parse_args()
if __name__ == "__main__":
    main()
