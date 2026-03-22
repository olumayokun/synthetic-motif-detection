import torch
import torch.nn as nn

class DNAMotifCNN(nn.Module):
    def __init__(self, seq_length=100):
        super(DNAMotifCNN, self).__init__()
        
        # 1. The Motif Detector (Convolutional Layer)
        # in_channels=4 (A, C, G, T)
        # out_channels=16 (We are creating 16 different "sliding windows" to look for 16 different patterns)
        # kernel_size=8 (Each window is 8 DNA letters wide)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=8)
        self.relu = nn.ReLU()
        
        # 2. The Downsampler (Pooling Layer)
        # Reduces the spatial dimension by keeping only the strongest signal in a window of 4
        self.pool = nn.MaxPool1d(kernel_size=4)
        
        # 3. The Decision Maker (Fully Connected Layer)
        # We need to calculate the flattened size mathematically based on our previous layers.
        # Formula: ((seq_length - kernel_size + 1) / pool_size) * out_channels
        # ((100 - 8 + 1) / 4) -> 23 * 16 = 368
        self.flattened_size = 23 * 16 
        
        self.fc1 = nn.Linear(in_features=self.flattened_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x expected shape: (Batch, 4, 100)
        
        # Pass through the CNN filter and activate
        x = self.conv1(x)         # Shape becomes: (Batch, 16, 93)
        x = self.relu(x)
        
        # Pool to reduce dimensionality
        x = self.pool(x)          # Shape becomes: (Batch, 16, 23)
        
        # Flatten the 2D matrices into a single 1D array for the dense layer
        x = torch.flatten(x, start_dim=1) # Shape becomes: (Batch, 368)
        
        # Final classification
        x = self.fc1(x)           # Shape becomes: (Batch, 1)
        x = self.sigmoid(x)       # Output a probability between 0.0 and 1.0
        
        return x

# Quick unit test to prove the architecture works
if __name__ == "__main__":
    # Create a dummy batch of 32 DNA sequences, 4 channels (ACGT), 100 letters long
    dummy_data = torch.randn(32, 4, 100) 
    
    model = DNAMotifCNN(seq_length=100)
    predictions = model(dummy_data)
    
    print(f"Input shape: {dummy_data.shape}")
    print(f"Output shape: {predictions.shape}") # Should print torch.Size([32, 1])