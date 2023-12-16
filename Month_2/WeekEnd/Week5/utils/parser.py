import argparse
import torch

def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=500)
    parser.add_argument("--num-class", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--results-folder-path", type=str, default="results")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()
