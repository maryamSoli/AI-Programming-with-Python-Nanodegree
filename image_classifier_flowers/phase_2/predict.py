import argparse
import torch
import json
from model_utils import load_checkpoint, predict

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained network.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, default=None, help="Path to JSON file mapping categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available")
    return parser.parse_args()

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load the model from checkpoint
    model, _, _ = load_checkpoint(args.checkpoint, device)
    
    # Predict the class probabilities and classes
    probs, classes = predict(args.image_path, model, args.top_k)
    
    # Map classes to actual flower names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        flower_names = [cat_to_name.get(cls, cls) for cls in classes]
    else:
        flower_names = classes
    
    print("Predicted Classes:", flower_names)
    print("Probabilities:", probs)

if __name__ == "__main__":
    main()
