
import argparse
import torch
from model_utils import load_data, create_model, train_model, save_checkpoint

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network on a flower dataset.")
    parser.add_argument("data_dir", type=str, help="Directory of the dataset (should contain 'train', 'valid', and 'test' folders)")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", help="Model architecture: vgg16 (or vgg13)")
    parser.add_argument("--learning_rate", type=float, default=0.0025, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training if available")
    return parser.parse_args()

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load training and validation data
    train_loader, valid_loader, class_to_idx = load_data(args.data_dir)
    num_classes = len(class_to_idx)
    
    # Build the model
    model, optimizer, criterion = create_model(args.arch, args.hidden_units, args.learning_rate, device, num_classes)
    
    # Train the network
    train_model(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device)
    
    # Save the checkpoint
    save_checkpoint(model, args.save_dir, args.arch, class_to_idx, args.hidden_units, args.learning_rate, args.epochs, optimizer)

if __name__ == "__main__":
    main()
