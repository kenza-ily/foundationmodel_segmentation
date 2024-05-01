import builtins
import datetime
from functools import partial
import inspect
import logging
import math
import os
import argparse
import random
import sys
import time
import torch
import wandb
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2
import numpy as np
from augment import *
from models_mae import *
from utils import *
from evaluate import *

def get_pretrain_data_loaders(args):
    print("Getting dataloaders")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    imgs_mean = np.array([0.5, 0.5, 0.5])
    imgs_std = np.array([0.5, 0.5, 0.5])
    # ----------- Transforms -----------
    # Simple resizing and rescaling
    train_transform = transforms.v2.Compose([
        transforms.v2.Resize((args.image_size, args.image_size)),
        transforms.v2.ToDtype(torch.float32, scale=True),
        transforms.v2.ToTensor(),
        transforms.v2.Normalize(imgs_mean, imgs_std)
    ])
    validation_transform = train_transform
    
    testloader = None
    valloader = None
    
    # ----------- Data Augmentation (if applicable) -----------
    if (args.noise == 'vanilla'):
        pass
    elif (args.noise == 'salt'):
        train_transform = v2.Compose([
        transforms.v2.Resize((args.image_size,args.image_size)),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.ToTensor(),
        SaltPepperTransform(),
        v2.Normalize(imgs_mean,imgs_std),
    ])
    elif (args.noise == 'gaussian'):
        transform = transforms.v2.Compose([
            transforms.v2.Resize((args.image_size,args.image_size)),
            transforms.v2.RandomHorizontalFlip(),
            transforms.v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.v2.ToDtype(torch.float32, scale=True),
            transforms.v2.ToTensor(),
            transforms.v2.Lambda(lambda x: add_gaussian_noise(x, 0, 0.1)),
            transforms.v2.Normalize(imgs_mean,imgs_std),
        ])
    
    if (args.dataset == 'coco'):
        # INSERT PATH HERE
        annotation_train_path='/cs/student/projects1/2020/ssoomro/24UCL_SelfSupervised_Segmentation/mae/annotations/instances_train2017.json'
        train_path='/cs/student/projects1/2020/ssoomro/24UCL_SelfSupervised_Segmentation/mae/train2017'
        annotation_validation_path='/cs/student/projects1/2020/ssoomro/24UCL_SelfSupervised_Segmentation/mae/annotations/instances_val2017.json'
        validation_path='/cs/student/projects1/2020/ssoomro/24UCL_SelfSupervised_Segmentation/mae/val2017'
        
        train_dataset = datasets.CocoDetection(train_path,annotation_train_path,transforms=train_transform)
        train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset, target_keys=("masks",))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,collate_fn=lambda batch: tuple(zip(*batch)),num_workers=args.num_workers,shuffle=True)
        
        validation_dataset = datasets.CocoDetection(validation_path,annotation_validation_path,transforms=validation_transform)
        validation_dataset = datasets.wrap_dataset_for_transforms_v2(validation_dataset, target_keys=("masks",))
        valloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, drop_last=False,collate_fn=lambda batch: tuple(zip(*batch)),num_workers=args.num_workers,shuffle=False)
        
        testLoader = None
        print(f"Number of training samples: {len(train_dataset)}")
    else:
        print("Your dataset does not match any of the dataset names")
        exit(-1)
    
    print("Getting data loaders completed, returning data loaders")
    return trainloader, valloader, testloader

def train_one_epoch(model: torch.nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    args=None):
    model.train(True)  # Set the model to training mode

    print_freq = 100  # Frequency of printing training status
    total_mae = 0  # To accumulate MAE over the epoch
    n_samples = 0  # To count the total number of samples processed

    # Reset optimizer gradient to zero
    optimizer.zero_grad()

    for i, data in enumerate(data_loader,0):
        # Move the samples to the specified device
        imgs = torch.stack(data[0])
        samples = imgs.to(device) #data[0] ->COCO change
        samples_patched = model.patchify(samples)

        # Forward pass
        loss, pred, mask = model(samples)

        # Calculate Mean Absolute Error
        mae = torch.abs(pred - samples_patched).mean()
        total_mae += mae.item() * samples_patched.size(0)
        n_samples += samples.size(0)

        # Print loss if it is not finite
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        # Calculate loss and perform a backward pass
        loss.backward()

        # Update the model weights
        optimizer.step()

        # Zero the gradients after updating
        optimizer.zero_grad()

        # Log training progress
        if i % print_freq == 0 or i == len(data_loader) - 1: #
            print(f"Epoch: [{epoch+1}][{i}/{len(data_loader)}] "
                  f"Loss: {loss.item():.4f} MAE: {total_mae/n_samples:.4f} LR: {optimizer.param_groups[0]['lr']:.6f}")

    average_mae = total_mae / n_samples

    # Return an image to save as example per epoch
    image = samples[0]
    image = image.cpu().numpy()  # Convert the tensor to numpy for visualization
    image = np.transpose(image, (1, 2, 0))

    # Save model checkpoint
    return {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "mae":average_mae}, image

def start_pretrain(log_file_name, trainloader, valloader, testloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    model = MaskedAutoencoderViT(img_size=args.image_size, patch_size=args.patch_size,
                                 in_chans=3, embed_dim=args.enc_projection_dim, 
                                 depth=args.enc_layers, num_heads=args.enc_num_heads,
                                 decoder_embed_dim=args.dec_projection_dim, 
                                 decoder_depth=args.dec_layers, decoder_num_heads=args.dec_num_heads,
                                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False, mask_type=args.mask_sampling, mask_ratio=args.mask_ratio, block_mask_ratio=args.block_mask_ratio)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    print("Starting PRE-Training loop")
    
    min_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    for epoch in range(args.epochs):
        # Train
        t1 = time.time()
        train_stats, image = train_one_epoch(model, trainloader,optimizer, device, epoch)
        t_train = time.time() - t1

        # Test in validation set
        t1 = time.time()
        test_stats = pretrain_test(model,valloader,device, epoch)
        t_test = time.time() - t1

        # Save example of a reconstructed image on final epoch
        if epoch == args.epochs - 1:
            image_file_name = f"{log_file_name.replace('.log', '')}_epoch_{epoch+1}.png"
            run_one_image(image, model.to('cpu'), os.path.join(args.out_dir,image_file_name))
        
        model.to(device)
        
        # Log the training and validation stats
        print(f"Training Loss: {train_stats['loss']}")
        print(f"Training MAE: {train_stats['mae']}")
        print(f"Training Time: {t_train}s")
        print(f"Validation Loss: {test_stats['loss']}")
        print(f"Validation MAE: {test_stats['mae']}")
        print(f"Validation Time: {t_test}s")

        train_metrics = {"pretrain/train_loss": train_stats['loss'], "pretrain/train_mae": train_stats['mae'], 
                   "pretrain/val_loss": test_stats['loss'], "pretrain/val_mae": test_stats['mae'],
                   "pretrain/train_time": t_train, "pretrain/val_time": t_test}
        if args.debug == 0 and args.demo == 0: 
            wandb.log(train_metrics)
        
        if (test_stats['loss'] < min_val_loss):
            print("New best model and val_loss updated")
            best_model = model
            min_val_loss = test_stats['loss']
            best_epoch = epoch

    # SAVE PRETRAINED
    torch.save(model.state_dict(), f"{log_file_name.replace('.log', '')}_trained_epoch_{best_epoch + 1}.pth")
    # SAVE PRETRAINED
    
    return model, train_metrics

def fine_tune(model, train_loader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()# Set the model to training mode

    total_loss = 0
    total_dice = 0
    total_dice_binary = 0

    

    for images, masks in train_loader:

        # Reset optimizer gradient to zero
        optimizer.zero_grad()

        images = images.to(device)  # Move images to the configured device
        masks = masks.to(device)

        # Forward pass
        outputs = model(images).squeeze(1)

        # Calculate loss
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        # Calculate other evaluation metrics
        dice = dice_score(outputs, masks)
        dice_bin = dice_binary(outputs, masks)  # Calculating the binary dice score

        total_loss += loss.item()
        total_dice += dice.item()
        total_dice_binary += dice_bin.item()  # Accumulating binary dice scores

    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_dice_binary = total_dice_binary / len(train_loader)

    return avg_loss, avg_dice, avg_dice_binary    

def start_fine_tune(fine_tune_model, train_loader, validation_loader, log_file_name, args):
    fine_tuned_optimizer = torch.optim.Adam(fine_tune_model.parameters(), lr=1e-3)
    print("Starting fine tune loop")
    min_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    for epoch in range(args.epochs):
        # Fine tuning Phase
        train_loss, train_dice, train_dice_binary = fine_tune(fine_tune_model, train_loader, fine_tuned_optimizer)

        # if (epoch + 1) % freq_info == 0:
        print(f'Epoch {epoch + 1}: Training Loss = {train_loss:.5f}, Training Dice Score = {train_dice:.5f},Training Binary Dice Score = {train_dice_binary:.5f}')
        # Validation and Saving Model
        val_loss, val_dice, val_dice_binary = validate(fine_tune_model, validation_loader)
        print(f'Epoch {epoch + 1}: Validation Loss = {val_loss:.5f}, Validation Dice Score = {val_dice:.5f},Validation Binary Dice Score = {val_dice_binary:.5f}')
        
        finetune_metrics = {"finetune/train_loss": train_loss, "finetune/train_dice": train_dice, "finetune/train_dice_binary": train_dice_binary, 
                   "finetune/val_loss": val_loss, "finetune/val_dice": val_dice, "finetune/val_dice_binary": val_dice_binary}
        if args.debug == 0 and args.demo == 0:
            wandb.log(finetune_metrics)
        
        if (val_loss < min_val_loss):
            print("New best model and val_loss updated")
            best_model = fine_tune_model
            min_val_loss = val_loss
            best_epoch = epoch
        
    # Save model with best validation loss
    torch.save(best_model.state_dict(), f"{log_file_name.replace('.log', '')}_tuned_epoch_{best_epoch + 1}.pth")
    
    return best_model, finetune_metrics