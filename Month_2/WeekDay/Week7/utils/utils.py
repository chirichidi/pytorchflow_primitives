import json
import os
import torch

MNIST_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def make_results_dir(results_folder_path):
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)


def make_sub_results(args):
    folder_contents = os.listdir(args.results_folder_path) + ["-1"]
    max_folder_name = max([int(f) for f in folder_contents])
    new_folder_name = str(max_folder_name + 1).zfill(2)
    save_path = os.path.join(args.results_folder_path, new_folder_name)
    args.save_path = save_path
    os.makedirs(save_path)
    return save_path


def save_hparams(args):
    dict_args = vars(args).copy()
    del dict_args["device"]
    with open(os.path.join(args.save_path, "hparam.json"), "w", encoding="utf-8") as f:
        json.dump(dict_args, f, indent=4)


def image_check(image, args):
    if args.data == "mnist":
        return image
    elif args.data == "cifar10":  # 3채널 데이터
        image = torch.unsqueeze(image, dim=0)
        return image
    else:
        raise NotImplementedError


def index2classname(index, args):
    if args.data == "mnist":
        return MNIST_CLASSES[index]
    elif args.data == "cifar10":  # 3채널 데이터
        return CIFAR_CLASSES[index]
    else:
        raise NotImplementedError
