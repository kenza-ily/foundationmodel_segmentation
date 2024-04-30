import torch
import torch.nn.functional as F

from metrics import dice_score, dice_binary, combined_loss
from models_mae import Autoencoder

def train(model, train_loader, optimizer,device):
    model.train()

    total_loss=0

    for images, masks in train_loader:
        images = images.to(device)  # Move images to the configured device
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader,device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_dice = 0
    total_dice_binary = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)  # Move images to the configured device
            masks = masks.to(device)
            outputs = model(images).squeeze(1)
            loss = combined_loss(outputs, masks)
            dice = dice_score(outputs, masks)
            dice_bin = dice_binary(outputs, masks)  # Calculating the binary dice score

            total_loss += loss.item()
            total_dice += dice.item()
            total_dice_binary += dice_bin.item()  # Accumulating binary dice scores

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_dice_binary = total_dice_binary / len(val_loader)  # Average binary dice score

    return avg_loss, avg_dice, avg_dice_binary


############################
# Main
############################

def main_baseline(train_loader, val_loader, device, log_file_name,epochs=30,img_size=224, patch_size=16,in_chans=3, embed_dim=128,depth=6, num_heads=4,lr=5e-3):

    # Initialize model
    model = Autoencoder(img_size, patch_size,in_chans, embed_dim,depth, num_heads)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    best_val_loss = float('inf')  # Initialize with a very high value

    for epoch in range(epochs):
        # Training Phase
        loss_train = train(model, train_loader, optimizer, device)
        print(f'Epoch {epoch + 1}: Training Loss = {loss_train:.5f}')

        # Validation phase
        val_loss, val_dice, val_dice_binary = validate(model, val_loader, device)
        print(f'Epoch {epoch + 1}: Validation Loss = {val_loss:.5f}, Validation Dice Score = {val_dice:.5f},Validation Binary Dice Score = {val_dice_binary:.5f}')

        # Check if the current validation loss is the lowest we've seen
        if val_loss < best_val_loss:
            best_model = model
            best_val_loss = val_loss
            best_epoch = epoch

    # Save model with best validation loss
    torch.save(best_model.state_dict(), f"{log_file_name.replace('.log', '')}_trained_epoch_{best_epoch + 1}.pth")
    print(f"Saved new best model with Validation Loss = {best_val_loss:.5f}")
