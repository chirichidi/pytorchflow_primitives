# 필요한 패키지를 import
import os
import torch
torch.autograd.set_detect_anomaly(True)

from utils.parser import parse_train_args
from utils.utils import make_results_dir
from utils.utils import make_sub_results
from utils.utils import save_hparams
from utils.get_module import get_dataloaders
from utils.eval import evaluate
from utils.get_module import get_optimizer
from utils.get_module import get_criteria
from utils.get_module import get_model

def main():
    args = parse_train_args()
    image_size = args.image_size
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_class = args.num_class
    lr = args.lr
    epoch = args.epoch
    results_folder_path = args.results_folder_path
    device = args.device

    make_results_dir(results_folder_path)
    save_path = make_sub_results(args)
    save_hparams(args)

    train_loader, val_loader, test_loader = get_dataloaders(args)
    
    # AI 모델 객체 생성 (과정에서 hyper-parameter가 사용)
    model = get_model(args)
    criteria = get_criteria()
    optimizer = get_optimizer(args, model)

    best = -1
    # for loop를 기반으로 학습이 시작됨
    for ep in range(epoch):
        # [epoch]을 학습하기 위해 batch 단위로 데이터를 가져와야 함
        # 이 과정이 for loop로 진행
        for idx, (image, label) in enumerate(train_loader):
            # dataloader가 넘겨주는 데이터를 받아서
            image = image.to(device)
            label = label.to(device)

            # AI 모델에게 넘겨주고
            output = model(image)
            # 출력물을 기반으로 Loss를 구하고
            loss = criteria(output, label)
            # Loss를 바탕으로 Optimize를 진행
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 특정 조건을 제시해서, 그 조건이 만족한다면 학습의 중간 과정을 확인
            if idx % 100 == 0:
                # 평가를 진행
                acc = evaluate(model, val_loader, device)
                # acc = evaluate_per_class(model, val_loader, device)
                # acc_per_class = evaluate_per_class(model, val_loader, device)
                # 보고 싶은 수치 확인 (Loss, 평가 결과 값, 이미지와 같은 meta-data)
                print(f"Epoch : {ep}/{epoch}, step : {idx}, Loss : {loss.item():.3f}")
                # 만약 평가 결과가 나쁘지 않으면
                if best < acc:
                    print(f"이전보다 성능이 좋아짐 {best} -> {acc}")
                    best = acc
                    # 모델을 저장
                    torch.save(
                        model.state_dict(), os.path.join(save_path, "best_model.ckpt")
                    )


    final_acc = evaluate(model, test_loader, device)
    print(f"최종 test data에 해당하는 평가 결과는 {final_acc:.3f}입니다")


if __name__ == "__main__":
    main()
