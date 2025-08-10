import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
### Encoder block should have an attention mechanism that is fully free, meaning information can be gleaned from every example unto another, but for decoders this should be masked, so only previous examples can advance any information for this example.
### After encoding information and recieiving BTC (C==64) embedding by token by example tensors, then apply self-attention heads, next feed forward layer
class SelfAttentionHead(nn.Module):
    # for our purposes, headSize should be n_embd
    #shouldMask represents a boolean of whether we are using masking or not, so that we can reuse the same class for the encoding and decoding parts
    # layer norms before attention/feedforward layers?
    def __init__(self, headSize, n_embd, shouldMask, block_size):
        super().__init__()
        self.headSize = headSize
        self.key = nn.Linear(n_embd, headSize, bias=False)
        self.query = nn.Linear(n_embd, headSize, bias=False)
        self.value = nn.Linear(n_embd, headSize, bias=False)
        self.shouldMask = shouldMask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # although the tutorial mentioned using the number of channels (C) as the scale factor, we use the scale factor
        # of the paper instead, so we scale by headSize, not channels
        weights = Q @ K.transpose(-2, -1) * (self.headSize**-0.5)
        # weights = Q @ K.transpose(-2, -1) * (C**-0.5)
        if self.shouldMask:
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = torch.softmax(weights, dim=-1)
        output = weights @ V
        return output, weights
class MultiAttentionHeads(nn.Module):
    def __init__(self, num_heads, size, n_embd, shouldMask, block_size, sparsity_pattern):
        super().__init__()
        if sparsity_pattern < 0:
            self.heads = nn.ModuleList([SelfAttentionHead(size, n_embd, shouldMask, block_size) for i in range(num_heads)])
        else: self.heads = nn.ModuleList([SparseSelfAttentionHead(size, n_embd, shouldMask, block_size, sparsity_pattern) for i in range(num_heads)])
        self.project = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        # output =  torch.cat([h(x) for h in self.heads], dim=-1)
        # return self.project(output)
        outputs = []
        attentions = []
        for h in self.heads:
            out, att = h(x)
            outputs.append(out)
            attentions.append(att)
        output = torch.cat(outputs, dim=-1)
        attention_maps = attentions
        return self.project(output), attention_maps
class FeedForwardLayer(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(FeedForwardLayer, self).__init__()
        self.first = nn.Linear(n_input, n_hidden)
        self.second = nn.ReLU()
        self.third = nn.Linear(n_hidden, n_output)
        # Writeup did not mention that we should dropout for the forward layer so we do not
    def forward(self, x):
        return self.third(self.second(self.first(x)))
class EncodingBlock(nn.Module):
    def __init__(self, num_embd, n_head, block_size, sparsity_pattern):
        super().__init__()
        headSize = num_embd // n_head
        self.Attention = MultiAttentionHeads(n_head, headSize, num_embd, False, block_size, sparsity_pattern)
        #the forward/sequential MLP layer
        self.sequential_forward = FeedForwardLayer(num_embd, num_embd*4, num_embd)
        self.ln1 = nn.LayerNorm(num_embd)
        self.ln2 = nn.LayerNorm(num_embd)
    def forward(self, x):
        attn_output, attn_map = self.Attention(x)
        x  = self.ln1(attn_output)+x 
        x = self.ln2(self.sequential_forward(x))+x 
        return x, attn_map
class EncodingTransformer(nn.Module):
    def __init__(self, vocab_size, num_embds, block_size, num_layers, num_heads, forward_inpt, forward_hid, forward_out, sparsity_pattern):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_embds)
        self.position_embedding = nn.Embedding(block_size, num_embds)
        self.layers = nn.Sequential(*[EncodingBlock(num_embds, num_heads, block_size, sparsity_pattern) for i in range(num_layers)])
        self.ln = nn.LayerNorm(num_embds)
        #the actual forward classifier that outputs the speech classes
        self.forward_classifier = FeedForwardLayer(forward_inpt, forward_hid, forward_out)
    def forward(self, x):
        B, T = x.size()
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(torch.arange(T, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        x = token_embedding + position_embedding
        attn_map_total = []
        for layer in self.layers:
            x, attn_map = layer(x)
            attn_map_total.extend(attn_map)
        x_new_dim = x.mean(dim=1)
        #should i use the layer norm here?
        logits = self.forward_classifier(self.ln(x_new_dim))
        return logits, attn_map_total
class DecodingBlock(nn.Module):
    def __init__(self, num_embd, n_head, block_size, forward_hid,sparsity_pattern):
        super().__init__()
        headSize = num_embd // n_head
        self.Attention = MultiAttentionHeads(n_head, headSize, num_embd, True, block_size, sparsity_pattern)
        #the forward/sequential MLP layer
        self.sequential_forward = FeedForwardLayer(num_embd, forward_hid, num_embd)
        self.ln1 = nn.LayerNorm(num_embd)
        self.ln2 = nn.LayerNorm(num_embd)
    def forward(self, x):
        attn_output, attn_map = self.Attention(x)
        x  = self.ln1(attn_output) + x
        x = self.ln2(self.sequential_forward(x)) + x
        return x, attn_map
class DecodingTransformer(nn.Module):
    def __init__(self, vocab_size, num_embds, block_size, num_layers, num_heads, forward_hid, sparsity_pattern):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_embds)
        self.position_embedding = nn.Embedding(block_size, num_embds)
        self.layers = nn.Sequential(*[DecodingBlock(num_embds, num_heads, block_size, forward_hid, sparsity_pattern) for i in range(num_layers)])
        self.ln = nn.LayerNorm(num_embds)
        self.lm_head = nn.Linear(num_embds, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, x):
        B, T = x.size()
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(torch.arange(T, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        x = token_embedding + position_embedding
        attn_map_total = []
        for layer in self.layers:
            x, attn_map = layer(x)
            attn_map_total.extend(attn_map)
        # don't aggregate for the decoder
        logits = self.lm_head(self.ln(x))
        #should i use the layer norm here?
        return logits, attn_map_total
    def lossCalc(self, X, Y):
        logits, attn_map_total = self.forward(X)
        a, b, c = logits.size()
        logits = logits.view(a*b, c)
        Y = Y.view(a*b)
        return self.loss_fn(logits, Y)
class SparseSelfAttentionHead(nn.Module):
    def __init__(self, headSize, n_embd, shouldMask, block_size, sparsity_pattern):
        super().__init__()
        self.headSize = headSize
        self.key = nn.Linear(n_embd, headSize, bias=False)
        self.query = nn.Linear(n_embd, headSize, bias=False)
        self.value = nn.Linear(n_embd, headSize, bias=False)
        self.shouldMask = shouldMask
        self.sparsity_pattern = sparsity_pattern
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Sparse attention weights
        weights = Q @ K.transpose(-2, -1) * (self.headSize**-0.5)
        
        if self.shouldMask:
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Apply sparsity pattern
        sparsity_mask = self.get_sparsity_mask(T)
        weights = weights.masked_fill(sparsity_mask == 0, float('-inf'))
        
        weights = torch.softmax(weights, dim=-1)
        output = weights @ V
        return output, weights
    
    def get_sparsity_mask(self, T):
        mask = torch.ones(T, T, device=self.tril.device)
        # Apply the sparsity pattern (here is an example of block sparsity)
        for i in range(T):
            for j in range(T):
                if abs(i - j) > self.sparsity_pattern:
                    mask[i, j] = 0
        return mask
