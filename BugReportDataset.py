import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class BugReportDataset(Dataset):
    def __init__(self, bug_pairs, labels, tokenizer, max_length=512):
        """
        Args:
            bug_pairs: List of tuples (bug1, bug2) containing bug report text
            labels: List of labels (1 for duplicate, 0 for non-duplicate)
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.bug_pairs = bug_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.bug_pairs)

    def __getitem__(self, idx):
        bug1, bug2 = self.bug_pairs[idx]
        
        # Tokenize bug reports
        encoding1 = self.tokenizer(
            bug1,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            bug2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input1': encoding1['input_ids'].squeeze(),
            'input2': encoding2['input_ids'].squeeze(),
            'attention1': encoding1['attention_mask'].squeeze(),
            'attention2': encoding2['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }
