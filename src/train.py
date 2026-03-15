import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm # For a nice progress bar
import os

def train_model(model, train_dataset, val_dataset=None, epochs=20, batch_size=8, lr=1e-4, save_path='weights/best_model.pth'):
    """
    Main training function for the defect detection model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Loss Function and Optimizer
    # Binary Cross Entropy with Logits is standard for binary segmentation
    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Optional: Learning rate scheduler (reduces LR if performance plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_loss = float('inf')

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Progress bar wrapper
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass (Backpropagation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")

        # 3. Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if not os.path.exists('weights'):
                os.makedirs('weights')
            torch.save(model.state_dict(), save_path)
            print(f"--> Model saved to {save_path}")

        scheduler.step(epoch_loss)

    print("Training Complete.")
