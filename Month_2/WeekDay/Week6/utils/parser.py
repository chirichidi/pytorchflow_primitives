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
    parser.add_argument("--data", type=str, default="mnist", choices=["mnist", "cifar10"], help="dataset name")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "lenet"], help="model name")
    return parser.parse_args()


def parse_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-image-path", type=str)
    parser.add_argument("--load-dir", type=str)
    parser.add_argument("--data", type=str, default="mnist", choices=["mnist", "cifar10"], help="dataset name")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "lenet"])
    return parser.parse_args()