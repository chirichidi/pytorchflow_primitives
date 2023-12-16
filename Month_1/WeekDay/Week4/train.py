# 필요한 패키지 import
import torch

from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import Compose

# hyperparameter 선언
image_size = 28
batch_size = 64
hidden_size = 500
num_classes = 10
lr = 0.001
epochs = 5
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
# 선언한 hyperparameter 를 저장

# 데이터 불러오기
transform = Compose([
    Resize((image_size, image_size)),
    ToTensor(),
])

# dataset 만들기 & 전처리하는 코드
train_val_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset, val_dataset = random_split(train_val_dataset, [50000, 10000], torch.Generator().manual_seed(42))
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

# dataloader 만들기
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# AI 모델 설계도 만들기 (class)
# init, forward 구현하기
class MyMLP(torch.nn.Module):
    def __init__(self, image_size, hidden_size, num_classes):
        super(MyMLP, self).__init__()
        self.linear_1 = torch.nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
        self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear_3 = torch.nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.view(-1, image_size*image_size)
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_2(x))
        x = self.softmax(self.linear_3(x))
        return x

# AI 모델 객체 생성 (과정에서 hyperparameter 사용)
model = MyMLP(image_size, hidden_size, num_classes)
# loss 객체 생성
criteria = torch.nn.CrossEntropyLoss()
# optimizer 객체 생성
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ------- 준비단계 끝 -------- 
# ------- 학습단계 시작 -------- 

# loop 돌면서 학습 진행
for epoch in range(epochs):
    # [epoch]을 학습하기 위해 batch 단위로 데이터를 가져와야함
    for idx, (image, label) in enumerate(train_loader):
    # 이 과정이 loop 로 진행
        # dataloader 가 넘겨주는 데이터를 받아서
        image = image.to(device)
        label = label.to(device)

        # ai 모델에게 넘겨주고
        output = model(image)
        # 출력물을 기반으로 loss 를 계산하고
        loss = criteria(output, label)
        # loss 를 바탕으로 Optimization 을 진행
        optimizer.zero_grad() # 파이토치는 기본적으로 gradient 를 계속 누적시키기 때문에, 이를 초기화 해줘야함.
        loss.backward()
        optimizer.step() # 한스텝은 W = W - lr * W.grad.  

        # 특정 조건을 제시해서, 그 조건이 만족한다면 학습의 중간 과정을 확인
        if idx % 100 == 0:
            # 평가를 진행
            pass
            # 보고 싶은 수치 확인(loss, 평가 결과 값, 이미지와 같은 meta-data 등)
            print(f"Epoch[{epoch}/{epochs}] Step[{idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
            # 만약 평가 결과가 괜찮으면
                # 모델 저장
        pass