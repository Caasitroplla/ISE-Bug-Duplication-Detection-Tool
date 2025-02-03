import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from dataset import BugReportDataset
from trainer import BugDuplicateDetector
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load and preprocess bug report data.
    Expected CSV format: bug1_text, bug2_text, is_duplicate
    """
    df = pd.read_csv(file_path)
    bug_pairs = list(zip(df['bug1_text'], df['bug2_text']))
    labels = df['is_duplicate'].values
    return bug_pairs, labels

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Load data
    bug_pairs, labels = load_data('bug_reports.csv')
    
    # Split data
    train_pairs, val_pairs, train_labels, val_labels = train_test_split(
        bug_pairs, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = BugReportDataset(train_pairs, train_labels, tokenizer)
    val_dataset = BugReportDataset(val_pairs, val_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )
    
    # Initialize and train the model
    detector = BugDuplicateDetector(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=300,
        learning_rate=0.001
    )
    
    # Train the model
    detector.train(train_loader, val_loader, epochs=10, patience=3)
    
    # Example prediction
    bug1 = "Application crashes when clicking the save button"
    bug2 = "Save functionality leads to program crash"
    similarity = detector.predict(bug1, bug2, tokenizer)
    print(f"Similarity score: {similarity:.4f}")
    print(f"Predicted as duplicate: {similarity > 0.5}")

if __name__ == "__main__":
    main()
