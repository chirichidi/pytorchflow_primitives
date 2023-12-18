import torch

def evaluate(model, dataloader, device):
    with torch.no_grad():
        model.eval()
        corrects, totals = 0, 0
        # dataloader를 바탕으로 for문을 돌면서 :
        for image, label in dataloader:
            # 데이터와 정답을 받아서
            image, label = image.to(device), label.to(device)

            # 모델에 입력을 넣고 출력을 생성, 출력 : [0.1, 0.05, 0.05, 0.70, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]
            output = model(image)
            # 출력물을 실제 정답과 비교 가능한 숫자로 변경
            output_index = torch.argmax(output, dim=1)
            # 출력과 실제 정답을 비교 (4, 3) -> correct
            corrects += torch.sum(label == output_index).item()
            totals += image.shape[0]
    acc = corrects / totals
    model.train()
    return acc

def evaluate_per_class(model, dataloader, device, total_num_class=10):
    with torch.no_grad():
        model.eval()
        corrects, totals = torch.zeros(total_num_class), torch.zeros(
            total_num_class
        )
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            output_index = torch.argmax(output, dim=1)
            for _class in range(total_num_class):
                totals[_class] += (label == _class).sum().item()
                corrects[_class] += (
                    ((label == _class) * (output_index == _class)).sum().item()
                )

    acc = corrects / totals
    model.train()
    return acc  # 10짜리 벡터 텐서의 형태