import math
import collections
from collections import Counter
import random
import re
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import torch
import os
import requests
import zipfile
from tqdm import tqdm



#Code from claude to check if the data is already downloaded and download it if it's not

class Text8:
    """The Text8 dataset."""
    def __init__(self, root='./data'):
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root)
        self.raw_file = os.path.join(root, 'text8')
        self.url = "http://mattmahoney.net/dc/text8.zip"

    def download(self):
        """Download the dataset if it's not already available."""
        if not os.path.exists(self.raw_file):
            print("Downloading Text8 dataset...")
            response = requests.get(self.url)
            zip_path = os.path.join(self.root, 'text8.zip')
            with open(zip_path, 'wb') as f:
                f.write(response.content)

            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)

            os.remove(zip_path)  # Remove the zip file after extraction
            print("Download complete and file extracted.")
        else:
            print("Text8 dataset already downloaded.")

    def read(self):
        """Read the raw text data."""
        if not os.path.exists(self.raw_file):
            print("Dataset not found. Downloading...")
            self.download()

        print("Reading Text8 dataset...")
        with open(self.raw_file, 'r', encoding='utf-8') as f:
            return f.read()



#split into the num_ngrams most common n-grams and use those as tokens, as well as single characters.
# Probably we only use 2-grams not higher n-grams, though we could modify this to use n-grams from nmin to nmax
#Claude helped me write this, modified to have a progressbar

class Tokenizer:
    def __init__(self, text, ngram_length, num_ngrams):
        self.text = text
        self.ngram_length = ngram_length
        self.num_ngrams = num_ngrams
        self.ngram_tokens = self._get_top_ngrams()
        self.vocab = {token: i for i, token in enumerate(self.ngram_tokens + list(set(self.text)))}

    def _get_top_ngrams(self):
        ngram_freq = Counter(self.text[i:i+self.ngram_length] for i in range(len(self.text) - self.ngram_length + 1))
        return [ngram for ngram, _ in ngram_freq.most_common(self.num_ngrams)]

    def tokenize(self):
        tokens = []
        i = 0
        pbar = tqdm(total=len(self.text), desc="Tokenizing")
        while i < len(self.text):
            found_ngram = False
            for ngram in self.ngram_tokens:
                if self.text[i:i+len(ngram)] == ngram:
                    tokens.append(self.vocab[ngram])
                    i += len(ngram)
                    pbar.update(len(ngram))
                    found_ngram = True
                    break
            if not found_ngram:
                tokens.append(self.vocab[self.text[i]])
                i += 1
                pbar.update(1)
        pbar.close()
        return tokens

    def decode(self, tokens):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join(inv_vocab[token] for token in tokens)


#Self explanatory

class SequenceMaker:
    def __init__(self, tokens, seq_length):
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, start_idx):
        inputs = torch.tensor(self.tokens[start_idx:start_idx+self.seq_length])
        targets = torch.tensor(self.tokens[start_idx+1:start_idx+self.seq_length+1])
        return inputs, targets


#To use sequencemaker to get sequences

class SequenceDataset(Dataset):
    def __init__(self, sequence_maker, indices):
        self.sequence_maker = sequence_maker
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.sequence_maker[self.indices[idx]]


#Makes batches of input and output sequences, allowing for the different sequences to overlap (note overlap_frac)

class DataLoader:
    def __init__(self, tokens, seq_length, batch_size, overlap_frac=0.5, num_train=100000, num_val=10000):
        self.sequence_maker = SequenceMaker(tokens, seq_length)
        self.batch_size = batch_size
        self.seq_step = int((1-overlap_frac)*seq_length)  # to use sequences that overlap
        self.num_train = num_train
        self.num_val = num_val
        
        self.train_indices, self.val_indices = self._split_indices()

    def _split_indices(self):
        seq_idx = list(range(0, len(self.sequence_maker) - self.sequence_maker.seq_length, self.seq_step))
        total_required = self.num_train + self.num_val
        if len(seq_idx) < total_required:
            warning_message = (
                f"Insufficient data: {len(seq_idx)} sequences available, "
                f"but {total_required} required. "
                f"Consider reducing num_train, num_val, "
                f"or increasing overlap_frac."
            )
            print(f"WARNING: {warning_message}")
        
        used_idx = random.sample(seq_idx, min(total_required, len(seq_idx)))
        random.shuffle(used_idx)
        
        train_idx = used_idx[:self.num_train]
        val_idx = used_idx[self.num_train:]
        
        return train_idx, val_idx

    def get_train_loader(self):
        train_dataset = SequenceDataset(self.sequence_maker, self.train_indices)
        return TorchDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self):
        val_dataset = SequenceDataset(self.sequence_maker, self.val_indices)
        return TorchDataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)