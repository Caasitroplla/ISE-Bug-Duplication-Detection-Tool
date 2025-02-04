import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from dataset import BugReportDataset
from trainer import BugDuplicateDetector
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os
import random

def load_data(dataset_path, platform='eclipse'):
    """
    Load and create bug report pairs from JSON files
    Args:
        dataset_path: Path to v02 directory
        platform: 'eclipse' or 'mozilla'
    Returns:
        tuple: (bug_pairs, labels)
    """
    bug_data = load_bug_attributes(dataset_path, platform)
    bug_pairs = []
    labels = []
    
    # Process duplicate relationships
    for bug_id, attributes in bug_data.items():
        if 'duplicate_of' in attributes and attributes['duplicate_of']:
            duplicate_id = attributes['duplicate_of']
            if duplicate_id in bug_data:
                # Create positive pair (duplicate)
                bug1_desc = bug_data[bug_id].get('description', '')
                bug2_desc = bug_data[duplicate_id].get('description', '')
                if bug1_desc and bug2_desc:  # Only add if both descriptions exist
                    bug_pairs.append((bug1_desc, bug2_desc))
                    labels.append(1)
                    
                    # Create some negative pairs for balance
                    # Take a random non-duplicate bug for negative examples
                    non_dup_ids = [bid for bid in bug_data.keys() 
                                 if bid != bug_id and bid != duplicate_id]
                    if non_dup_ids:
                        random_bug_id = random.choice(non_dup_ids)
                        random_desc = bug_data[random_bug_id].get('description', '')
                        if random_desc:
                            bug_pairs.append((bug1_desc, random_desc))
                            labels.append(0)
    
    return bug_pairs, labels

def load_bug_attributes(dataset_path, platform='eclipse'):
    """
    Load all attributes for bug reports from separate JSON files
    Args:
        dataset_path: Path to v02 directory
        platform: 'eclipse' or 'mozilla'
    Returns:
        dict: {bug_id: {attribute: value}}
    """
    platform_path = os.path.join(dataset_path, platform)
    bug_data = {}
    
    # Load each JSON file in the directory
    for filename in os.listdir(platform_path):
        if filename.endswith('.json'):
            attribute_name = filename.replace('.json', '')
            file_path = os.path.join(platform_path, filename)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Each JSON file has the attribute name as the root key
                attribute_data = data[attribute_name]
                
                # Initialize bug entries if they don't exist
                for bug_id, value in attribute_data.items():
                    if bug_id not in bug_data:
                        bug_data[bug_id] = {}
                    bug_data[bug_id][attribute_name] = value
    
    return bug_data

def process_platform(dataset_path, platform, tokenizer):
    """
    Process and train on a specific platform's dataset
    """
    print(f"\nProcessing {platform.upper()} dataset...")
    
    # Load data
    bug_pairs, labels = load_data(dataset_path, platform)
    
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
    return detector

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    dataset_path = 'DataSet/v02'
    
    # Process Eclipse dataset
    eclipse_detector = process_platform(dataset_path, 'eclipse', tokenizer)
    
    # Process Mozilla dataset
    mozilla_detector = process_platform(dataset_path, 'mozilla', tokenizer)
    
    # Example prediction using Eclipse model
    bug1 = "Application crashes when clicking the save button"
    bug2 = "Save functionality leads to program crash"
    similarity = eclipse_detector.predict(bug1, bug2, tokenizer)
    print(f"\nEclipse Model - Similarity score: {similarity:.4f}")
    print(f"Predicted as duplicate: {similarity > 0.5}")

if __name__ == "__main__":
    main()
