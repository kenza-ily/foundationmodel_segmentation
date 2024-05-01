import argparse
import datetime
import inspect
import logging
import os
import sys

import torch

from data_loader import getPetDataset
from baseline import main_baseline

def get_args_parser():
    """Get the arguments parser"""
    parser = argparse.ArgumentParser()
    
    # DEMO Functionality 
    parser.add_argument('--demo', default=0, type=int, choices=[0, 1, 2, 3, 4], required=False)

    # Basic parameters
    parser.add_argument('--root_dir', default='./', type=str,help='Root directory')
    parser.add_argument('--wandb_project', default="comp0197-ADL-CW2-MAE", 
                        help='Use wandb to log the training process, empty for no logging')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # Dataset parameters
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for training/testing')
    parser.add_argument('--image_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Number of patches to sample from an image')
    parser.add_argument('--data_path', default='./data', type=str,
                        help='Pretrain dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Path where to save results, empty for no saving')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='Learning rate for optimizer')
    
    parser.add_argument("--noise", default="vanilla", type=str, choices=["vanilla", "gaussian", "salt"], required=False,
                        help="Type of noise: vanilla, gaussian, or salt")
    parser.add_argument("--baseline", default=0, type=int, choices=[0, 1], required=False,
                        help="Whether to run baseline model: 0 (no), 1 (yes)")
    parser.add_argument("--dataset", default="coco", type=str, choices=["coco"], required=False,
                        help="Dataset to use, currently only 'coco' is supported")
    parser.add_argument("--mask_sampling", default="random", type=str, choices=["random","grid","block", "semantic"], required=False,
                        help="Type of mask sampling: random or semantic")
    parser.add_argument("--fine_tune_size", default=1, type=float, choices=[1, 0.5, 0.1], required=False,
                        help="Size of fine-tuning dataset: 1, 0.5, or 0.1")
    parser.add_argument('--out_dir', default="", type=str)
    parser.add_argument('--block_mask_ratio', default=0.5, type=float, required=False)
    parser.add_argument('--debug', default=0, type=int, choices=[0, 1])

    # Training parameters
    parser.add_argument('--epochs', default=10, type=int)

    # Encoder and decoder parameters
    parser.add_argument('--enc_projection_dim', default=128, type=int,
                        help='Encoder projection dimension')
    parser.add_argument('--dec_projection_dim', default=64, type=int,
                        help='Decoder projection dimension')
    parser.add_argument('--enc_num_heads', default=4, type=int,
                        help='Encoder number of heads')
    parser.add_argument('--dec_num_heads', default=4, type=int,
                        help='Decoder number of heads')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help='Encoder number of layers')
    parser.add_argument('--dec_layers', default=2, type=int,
                        help='Decoder number of layers')
    return parser

def start_logging(log_file=None, log_level=logging.INFO, Prefix=None):
    """
    Initializes logging configuration.

    Args:
        log_file (str): Path to the log file. If None, a default filename with timestamp and optional prefix will be used.
        log_level (int): Logging level (default: logging.INFO).
        Prefix (str): Optional prefix for the log file name.
    """
    # If log_file is None, construct the file name with optional Prefix prefix
    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if Prefix:
            log_file = f'Prefix_{Prefix}_{timestamp}.log'
        else:
            log_file = f'new_{timestamp}.log'
            
    # Initialize logging configuration
    logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging started")
    
    # Report the name of the caller file
    caller_file = inspect.stack()[1].filename  # Get the file name of the caller
    caller_file_name = os.path.basename(caller_file)  # Extract the base name of the caller file
    logging.info(f"Logging initiated from: {caller_file_name}")
    
    # Report the Prefix if it exists
    if Prefix:
        logging.info("Prefix is " + Prefix)
        
    return log_file

def print_command_line_arguments(args, args_dict, log_file_name, epochs):
    if (args.baseline == 1):
        for i in range(1, 10):
            print("RUNNING BASELINE MODEL ONLY")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        train_loader, val_loader, test_loader = getPetDataset(args)
        main_baseline(train_loader, val_loader, device, log_file_name,epochs=epochs)
        print("Baseline finished, exiting")
        sys.exit(0)
    
    # Print all command line arguments and their values
    print("All command-line arguments and their parsed values:")
    for arg, value in args_dict.items():
        print(f"{arg}: {value}")