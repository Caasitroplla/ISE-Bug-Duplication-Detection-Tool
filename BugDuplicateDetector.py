import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class BugDuplicateDetector:
    def __init__(self, vocab_size, embedding_dim=300, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseCNN(vocab_size, embedding_dim).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, train_loader, val_loader, epochs=10, patience=3):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                input1 = batch['input1'].to(self.device)
                input2 = batch['input2'].to(self.device)
                attention1 = batch['attention1'].to(self.device)
                attention2 = batch['attention2'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input1, input2, attention1, attention2)
                loss = self.criterion(outputs.squeeze(), labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Validation phase
            val_loss, val_metrics = self.evaluate(val_loader)
            
            print(f'\nEpoch {epoch+1}/{epochs}')
            print(f'Training Loss: {total_loss/len(train_loader):.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Metrics: Precision: {val_metrics["precision"]:.4f}, '
                  f'Recall: {val_metrics["recall"]:.4f}, F1: {val_metrics["f1"]:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input1 = batch['input1'].to(self.device)
                input2 = batch['input2'].to(self.device)
                attention1 = batch['attention1'].to(self.device)
                attention2 = batch['attention2'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input1, input2, attention1, attention2)
                loss = self.criterion(outputs.squeeze(), labels)
                total_loss += loss.item()
                
                predictions = (outputs.squeeze() > 0.5).float()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            all_preds, 
            average='binary'
        )
        
        return total_loss / len(data_loader), {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict(self, bug1, bug2, tokenizer):
        self.model.eval()
        with torch.no_grad():
            # Tokenize inputs
            encoding1 = tokenizer(
                bug1,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            encoding2 = tokenizer(
                bug2,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input1 = encoding1['input_ids'].to(self.device)
            input2 = encoding2['input_ids'].to(self.device)
            attention1 = encoding1['attention_mask'].to(self.device)
            attention2 = encoding2['attention_mask'].to(self.device)
            
            similarity = self.model(input1, input2, attention1, attention2)
            return similarity.item()