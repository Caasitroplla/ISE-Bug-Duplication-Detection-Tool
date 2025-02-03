import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, num_filters=128, filter_sizes=(3, 4, 5)):
        super(SiameseCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs) 
            for fs in filter_sizes
        ])
        
        # Fully connected layers
        conv_output_size = len(filter_sizes) * num_filters
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)
    
    def forward_one(self, x, attention_mask=None):
        # Embedding layer
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(-1)
        
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        # Apply CNN and max-pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pool_out = F.max_pool1d(conv_out, conv_out.shape[2])
            conv_outputs.append(pool_out)
        
        # Concatenate all conv outputs
        concat_output = torch.cat(conv_outputs, dim=1).squeeze(-1)
        
        # Fully connected layers
        out = F.relu(self.fc1(concat_output))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        
        return out
    
    def forward(self, input1, input2, attention1=None, attention2=None):
        # Process both inputs through the same network
        out1 = self.forward_one(input1, attention1)
        out2 = self.forward_one(input2, attention2)
        
        # Calculate similarity
        distance = torch.abs(out1 - out2)
        similarity = self.fc_out(distance)
        
        return torch.sigmoid(similarity)
