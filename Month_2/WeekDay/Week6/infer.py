# 필요한 패키지를 import
import os
import torch
import torch.nn as nn
import argparse
import json
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from utils.get_module import load_image
from utils.parser import parse_infer_args
from utils.get_module import get_transform
from utils.get_module import get_model

def main():
    args = parse_infer_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(args.load_dir, "best_model.ckpt")
    hparam_path = os.path.join(args.load_dir, "hparam.json")

    with open(hparam_path, "r", encoding='utf-8') as f:
        train_args = argparse.Namespace(**json.load(f))

    image_size = train_args.image_size
    hidden_size = train_args.hidden_size
    num_class = train_args.num_class

    image = load_image(args)
    transform = get_transform(args=args)
    image = transform(image).to(args.device)

    model = get_model(args)

    # 학습된 weight를 생성한 AI 모델 객체에 넣어주기
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)

    # 준비한 데이터를 AI 모델에 넣어주기
    output = model(image)

    # 결과로 나온 데이터를 해석 (시각화)
    output_index = torch.argmax(output).item()
    prob = torch.max(nn.functional.softmax(output, dim=1)) * 100

    print(f"모델은 이 이미지를 {prob:.4f}%의 확률로 {output_index+1} 라고 합니다.")


if __name__ == "__main__":
    main()
