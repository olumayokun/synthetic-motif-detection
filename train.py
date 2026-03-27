import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import DNAMotifCNN
from dataset import SyntheticDNADataset

def train_model():
    # 1. Hardware Configuration (The SWE touch!)
    # Automatically use Apple Silicon (MPS), NVIDIA GPU (CUDA), or fallback to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🚀 Training on device: {device}")

    # 2. Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 40
    DATASET_SIZE = 2000 # 1000 positive (TATA box), 1000 negative (random)

    # 3. Load and Split the Data
    print("Loading synthetic DNA dataset...")
    full_dataset = SyntheticDNADataset(num_samples=DATASET_SIZE, seq_length=100)
    
    # 80/20 Train-Validation Split to prevent overfitting
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoaders handle the batching and shuffling automatically
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize the Model, Loss Function, and Optimizer
    model = DNAMotifCNN(seq_length=100).to(device)
    
    # Binary Cross Entropy Loss (because we output a probability between 0 and 1)
    criterion = nn.BCELoss()
    
    # Adam Optimizer (The industry standard for gradient descent)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. The Training Loop
    print("\nStarting Training...")

    # List to track metrics for plotting in the notebook
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(EPOCHS):
        model.train() # Put model in training mode
        running_loss = 0.0
        
        for sequences, labels in train_loader:
            # Move data to the correct hardware
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Step A: Zero out the old gradients
            optimizer.zero_grad()
            
            # Step B: Forward Pass (Make a prediction)
            predictions = model(sequences)
            
            # Step C: Calculate the Loss (How wrong was the prediction?)
            loss = criterion(predictions, labels)
            
            # Step D: Backward Pass (Calculate the gradients)
            loss.backward()
            
            # Step E: Update the Weights
            optimizer.step()
            
            running_loss += loss.item()
            
        # 6. The Validation Loop (Test on unseen data)
        model.eval() # Put model in evaluation mode (turns off dropout, etc.)
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # torch.no_grad() disables gradient tracking to save memory/compute
        with torch.no_grad():
            for val_seqs, val_labels in val_loader:
                val_seqs, val_labels = val_seqs.to(device), val_labels.to(device)
                
                val_preds = model(val_seqs)
                loss = criterion(val_preds, val_labels)
                val_loss += loss.item()
                
                # Convert probabilities to strict 0 or 1 guesses
                binary_preds = (val_preds > 0.5).float()
                correct_predictions += (binary_preds == val_labels).sum().item()
                total_predictions += val_labels.size(0)
                
        # Calculate Epoch Metrics
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_predictions / total_predictions) * 100

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    print("\n✅ Training Complete!")
    return model, history

if __name__ == "__main__":
    trained_model = train_model()