# PLEASE SEE: use --debug 1 if you would like to disable WANDB integration AND logging
# To see samples (logs, models, visual), please see samples/ directory
# Search the workspace for "INSERT PATH HERE" comment to change the path for dataset etc

# (RECOMMENDED SINCE 1 EPOCH DEMONSTRATION BY DEFAULT, NO WANDB)
# Demo Usage:
# python main.py --demo 2 | noise = vanilla, mask_sampling = random
# python main.py --demo 2 | noise = vanilla, mask_sampling = block
# python main.py --demo 3 | noise = gaussian, mask_sampling = grid
# python main.py --demo 4 | BASELINE FULLY SUPERVISED

# Other typical usage:
# For baseline    | python main.py --debug 1 --baseline 1
# Other example 1 | python main.py --noise gaussian --mask_sampling random --fine_tune_size 2 --epochs 30
# Other example 2 | python main.py --noise vanilla --mask_sampling block --block_mask_ratio 0.5 --fine_tune_size 1 --epochs 30

import builtins
import logging
import random
import sys

import wandb
import torch
import numpy as np

# Use of wildcard imports is not good practice according to PEP 8
from data_loader import getPetDataset
from train import get_pretrain_data_loaders, start_pretrain, transfer_model, start_fine_tune
from evaluate import evaluate_fine_tuned_model
from parse_and_log import start_logging, print_command_line_arguments, get_args_parser

def set_seed(SEED):
    print("Setting seed value: " + str(SEED))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def print_command_line_args():
    print("Number of arguments:", len(sys.argv))
    print("Argument List:", sys.argv)

def main(args):
    args_dict = vars(args)
    
    if args.debug == 0:
        # Initiate logging with unique identifier
        # Structure: YYYY-MM-DD_HH-MM-SS.log (and corresponding images/models)
        log_file_name = start_logging()
        print("Log file generated: " + log_file_name)
        # Override print() to print AND log the message
        original_print = print
        def custom_print(*args, **kwargs):
            original_print(*args, **kwargs)
            logging.info(' '.join(map(str, args)))
        builtins.print = custom_print
    
    if args.debug == 0 and args.demo == 4:
        print("STARTED IN DEMO MODE: NO WANDB INIT")
        print("Enabling baseline mode with 1 epoch")
        args.baseline = 1
        args.epochs = 1
    
    elif args.debug == 0 and args.demo in [1, 2, 3]:
        print("STARTED IN DEMO MODE: NO WANDB INIT")
        args.noise = "gaussian"
        args.mask_sampling = "grid"
        args.fine_tune_size = 1
        args.epochs = 1
    
    # Regular run, not in debug mode
    elif args.debug == 0 and args.demo == 0:
        # Initialize wandb run
        init_config = {
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "weight_decay": args.weight_decay,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "encoder_dim": args.enc_projection_dim,
                "enc_heads": args.enc_num_heads,
                "enc_layers": args.enc_layers,
                "decoder_dim": args.dec_projection_dim,
                "dec_heads": args.dec_num_heads,
                "dec_layers": args.dec_layers,
                "train_size": args.fine_tune_size,
                "noise": args.noise,
                "mask_sampling": args.mask_sampling,
                "mask_ratio": args.mask_ratio,
                "block_mask_ratio": args.block_mask_ratio,
                "log_file": log_file_name
            }
        
        pretrain_config = init_config
        pretrain_config['phase'] = "pretrain"
        pretrain_run = wandb.init(
            # set the wandb project where this run will be logged
            project = args.wandb_project,
            # track hyperparameters and run metadata
            config=pretrain_config)
    
    
    elif args.debug == 1 and args.demo == 0:
        # Debugging is enabled
        # Set Debugging arguments
        print("STARTED IN DEBUG MODE: NO LOG FILE GENERATED")
        print("STARTED IN DEBUG MODE: NO WANDB INIT")
        args.noise = "gaussian"
        args.mask_sampling = "grid"
        args.fine_tune_size = 0.1
        args.epochs = 1
        log_file_name = 'debug.log'


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    # Seed for reproducability, can be seen from default arg
    set_seed(args.seed)
    
    # Prints (and logs) all parameters for the experiment
    print_command_line_arguments(args, args_dict, log_file_name, args.epochs)
    
    # Get pretrain loaders 
    # (IF APPLICABLE) Data augmentation happens in this function
    trainloader, valloader, testloader = get_pretrain_data_loaders(args) 
    
    # Pretrain model and put it on CPU / GPU
    model, pretrain_metrics = start_pretrain(log_file_name, trainloader, valloader, testloader, args)
    model = model.to(device)
    if args.debug == 0 and args.demo == 0:
        wandb.finish()
    
    # Tranfer model
    fine_tune_model = transfer_model(model, args)
    
    # Gets the pet train, validation and test loaders
    # Note: train scales with --fine_tune_size argument
    pet_train_loader, pet_validation_loader, pet_test_loader = getPetDataset(args)
    
    if args.debug == 0 and args.demo == 0:
        finetune_config = init_config
        finetune_config['phase'] = "finetune"
        finetune_run = wandb.init(
            # set the wandb project where this run will be logged
            project = args.wandb_project,
            # track hyperparameters and run metadata
            config=finetune_config)
    
    fine_tune_model, finetune_metrics = start_fine_tune(fine_tune_model, pet_train_loader, pet_validation_loader, log_file_name, args)
    
    test_metrics = evaluate_fine_tuned_model(model=fine_tune_model, test_loader=pet_test_loader)
    if args.debug == 0 and args.demo == 0:
        for key,value in test_metrics.items():
            wandb.summary[key] = value

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)