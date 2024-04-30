import torch
from metrics import combined_loss, pixel_accuracy, iou, sensitivity, specificity, precision, f1_score, dice_score, dice_binary

def pretrain_test(model, data_loader, device, epoch):
    model.eval()  # Set the model to evaluation mode

    # print_freq = 100
    total_mae = 0  # Accumulate MAE over the validation set
    n_samples = 0  # Count the total number of samples processed

    with torch.no_grad():  # Disable gradient computation
        for i, data in enumerate(data_loader,0):
            imgs = torch.stack(data[0])
            samples = imgs.to(device) #data[0] ->COCO change
            # samples = data[0].to(device)
            samples_patched = model.patchify(samples)

            # Forward pass
            loss, pred, _ = model(samples)

            # Calculate Mean Absolute Error
            mae = torch.abs(pred - samples_patched).mean()
            total_mae += mae.item() * samples_patched.size(0)
            n_samples += samples.size(0)

            # Log validation progress
            if i == len(data_loader) - 1: #i % print_freq == 0 or
              print(f"Epoch: [{epoch + 1}][{i}/{len(data_loader)}] "
                  f"Val Loss: {loss.item():.4f} Val MAE: {total_mae/n_samples:.4f}")

    average_mae = total_mae / n_samples
    return {"loss":loss,"mae": average_mae}

# Testing the model
def evaluate_fine_tuned_model(model, test_loader):
    print("Evaluating fine tuned model")
    print(f"Total batches expected: {len(test_loader)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    total_sensitivity = 0.0  # Recall
    total_specificity = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    total_dice_soft = 0.0
    total_dice_binary = 0.0

    # Initialize a counter to detect the last batch
    current_batch = 0
    total_batches = len(test_loader)

    with torch.no_grad():
        for images, masks in test_loader:

            images = images.to(device)  # Move images to the configured device
            masks = masks.to(device)#.squeeze(1)
            outputs = model(images)
            outputs = outputs.squeeze(1)

            preds_probs = torch.sigmoid(outputs)
            preds_binary = (preds_probs > 0.5).float()

            # Flatten the masks and predictions for metric calculations
            masks_flat = masks.view(-1).int()  # Flatten and convert to integer
            preds_flat = preds_binary.view(-1).int()  # Flatten and convert to integer

            loss = combined_loss(outputs, masks)
            acc = pixel_accuracy(preds_flat, masks_flat)
            iou_score = iou(preds_flat, masks_flat)
            sens = sensitivity(preds_flat, masks_flat)
            spec = specificity(preds_flat, masks_flat)
            prec = precision(preds_flat, masks_flat)
            f1 = f1_score(preds_flat, masks_flat)
            dice_soft = dice_score(preds_probs, masks)  # Use probabilities for soft dice calculation
            dice_bin = dice_binary(preds_probs, masks)  # Use binary predictions for hard dice calculation

            total_loss += loss.item()
            total_accuracy += acc.item()
            total_iou += iou_score.item()
            total_sensitivity += sens.item()
            total_specificity += spec
            total_precision += prec.item()
            total_f1 += f1.item()
            total_dice_soft += dice_soft.item()
            total_dice_binary += dice_bin.item()

            # save_batch(images, masks, outputs,current_batch)
            current_batch += 1

    # Calculate average metrics
    avg_loss = total_loss / current_batch
    avg_accuracy = total_accuracy / current_batch
    avg_iou = total_iou / current_batch
    avg_sensitivity = total_sensitivity / current_batch
    avg_specificity = total_specificity / current_batch
    avg_precision = total_precision / current_batch
    avg_f1 = total_f1 / current_batch
    avg_dice_soft = total_dice_soft / current_batch
    avg_dice_binary = total_dice_binary / current_batch

    print(f'Test Loss: {avg_loss:.5f}')
    print(f'Accuracy: {avg_accuracy:.5f}')
    print(f'IoU: {avg_iou:.5f}')
    print(f'Sensitivity: {avg_sensitivity:.5f}')
    print(f'Specificity: {avg_specificity:.5f}')
    print(f'Precision: {avg_precision:.5f}')
    print(f'F1 Score: {avg_f1:.5f}')
    print(f'Dice Coefficient (Soft): {avg_dice_soft:.5f}')
    print(f'Dice Coefficient (Binary): {avg_dice_binary:.5f}')
    
    test_metrics = {"test/loss": avg_loss, "test/accuracy": avg_accuracy, "test/iou": avg_iou, "test/sensitivity": avg_sensitivity, 
               "test/specificity": avg_specificity, "test/precision": avg_precision, "test/f1": avg_f1, 
               "test/dice_soft": avg_dice_soft, "test/dice_binary": avg_dice_binary}
    # if args.debug == 0:
    #     wandb.log(test_metrics)

    return test_metrics

def validate(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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