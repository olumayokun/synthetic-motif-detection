import torch
from torch.utils.data import Dataset
import numpy as np
import random

class SyntheticDNADataset(Dataset):
    """
    A PyTorch Dataset that generates synthetic DNA sequences.
    Positive samples contain a specific regulatory motif.
    Negative samples are purely random background DNA.
    """
    def __init__(self, num_samples=1000, seq_length=100, motif="TATAAAA"):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.motif = motif
        self.motif_len = len(motif)
        
        # DNA mapping dictionary
        self.char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.bases = ['A', 'C', 'G', 'T']
        
        # Generate the data when the class is instantiated
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences = []
        labels = []
        
        for i in range(self.num_samples):
            # 1. Generate a random background DNA sequence
            seq_list = random.choices(self.bases, k=self.seq_length)
            
            # 2. 50% chance to inject the motif (Positive sample)
            if i % 2 == 0: 
                # Pick a random starting position for the motif
                insert_pos = random.randint(0, self.seq_length - self.motif_len)
                
                # Overwrite the random DNA with our specific motif
                for j, char in enumerate(self.motif):
                    seq_list[insert_pos + j] = char
                
                labels.append(1.0) # Functional
            else:
                labels.append(0.0) # Non-functional (Random)
                
            sequences.append("".join(seq_list))
            
        return sequences, labels

    def _one_hot_encode(self, sequence):
        """
        Converts a string like 'ACGT' into a 2D matrix of 0s and 1s.
        """
        # Convert string to list of integers [0, 1, 2, 3]
        int_seq = [self.char_to_int[char] for char in sequence]
        
        # Vectorized One-Hot Encoding using numpy (SWE performance trick!)
        one_hot = np.eye(4)[int_seq] 
        
        # np.eye creates shape (seq_length, 4). 
        # PyTorch 1D CNNs expect (Channels, seq_length), so we transpose it.
        one_hot = one_hot.T 
        
        return torch.tensor(one_hot, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        This is the magic method PyTorch uses to fetch data during training.
        """
        seq_str = self.sequences[idx]
        label = self.labels[idx]
        
        # Encode on the fly
        encoded_seq = self._one_hot_encode(seq_str)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return encoded_seq, label_tensor

# Quick unit test
if __name__ == "__main__":
    # Generate 10 dummy sequences to test the class
    dataset = SyntheticDNADataset(num_samples=10, seq_length=100)
    
    # Fetch the very first item
    first_seq, first_label = dataset[0]
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Encoded Sequence Shape: {first_seq.shape}") # Should be [4, 100]
    print(f"Label Shape: {first_label.shape}")          # Should be [1]
    print(f"Label Value: {first_label.item()}")