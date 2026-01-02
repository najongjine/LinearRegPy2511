import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# ==========================================
# 1. CSV 데이터 읽기 및 전처리
# ==========================================
print("--- [1] 데이터 준비 단계 ---")
try:
    df = pd.read_csv('ulca_admission_data.csv')
except FileNotFoundError:
    print("오류: 'ulca_admission_data.csv' 파일을 찾을 수 없습니다.")
    exit()

gre = torch.FloatTensor(df['gre'].values)
gpa = torch.FloatTensor(df['gpa'].values)
rank = torch.FloatTensor(df['rank'].values)

# 정규화
x_data = torch.stack([gre / 800.0, gpa / 4.0, rank / 4.0], dim=1)
y_data = torch.FloatTensor(df['admit'].values).reshape(-1, 1)

# ★ 데이터 확인 로그
print(f"전체 입력 데이터 Shape: {x_data.shape}")  # (1000, 3) 예상
print(f"입력 데이터 예시 (첫번째 사람): {x_data[0]}")
print(f"정답 데이터 예시 (첫번째 사람): {y_data[0]}")

# ==========================================
# 2. 모델 정의 (로그 기능 추가)
# ==========================================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 3개 입력 -> 16개로 뻥튀기
        self.layer1 = nn.Linear(3, 16) 
        self.relu = nn.ReLU()
        # 16개 -> 1개로 압축
        self.layer2 = nn.Linear(16, 1) 
        self.sigmoid = nn.Sigmoid()
        
        # ★ 디버그용 플래그 (처음 한 번만 찍기 위해)
        self.print_debug = True

    def forward(self, x):
        # ---------------------------------------------------------
        # 학습 루프에서 호출될 때마다 실행되지만, 로그는 맨 처음에만 찍힘
        # ---------------------------------------------------------
        if self.print_debug:
            print("\n--- [DEBUG] Forward Pass (데이터 흐름 추적) ---")
            print(f"1. [Input] 들어오기 전 Shape: {x.shape}")
            print(f"   ㄴ 값 예시(상위 1개): {x[0].tolist()}")

        x = self.layer1(x)
        if self.print_debug:
            print(f"2. [Layer1] Linear 통과 후 Shape: {x.shape} (3->16 확장)")
            print(f"   ㄴ 값 예시: {x[0].detach().numpy().round(3)}") # 보기 좋게 반올림

        x = self.relu(x)
        if self.print_debug:
            print(f"3. [ReLU] 활성화 함수 통과 후 (음수 제거):")
            print(f"   ㄴ 값 예시: {x[0].detach().numpy().round(3)}")

        x = self.layer2(x)
        if self.print_debug:
            print(f"4. [Layer2] Linear 통과 후 Shape: {x.shape} (16->1 압축)")
            print(f"   ㄴ 값 예시: {x[0].detach().numpy().round(3)}")

        x = self.sigmoid(x)
        if self.print_debug:
            print(f"5. [Sigmoid] 최종 출력 (0~1 확률):")
            print(f"   ㄴ 값 예시: {x[0].item():.4f}")
            print("--------------------------------------------\n")
            self.print_debug = False # ★ 로그 껐음 (이제부터 출력 안함)

        return x

# ==========================================
# 3. 모델 가중치(Weight) Shape 확인
# ==========================================
model = SimpleNet()

print("\n--- [2] 모델 내부 가중치(파라미터) 확인 ---")
print(f"Layer1 가중치(W) Shape: {model.layer1.weight.shape} (Out:16, In:3)")
print(f"Layer1 편향(b) Shape  : {model.layer1.bias.shape} (16)")
print(f"Layer2 가중치(W) Shape: {model.layer2.weight.shape} (Out:1, In:16)")
print(f"Layer2 편향(b) Shape  : {model.layer2.bias.shape} (1)")

# ==========================================
# 4. 학습 시작
# ==========================================
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

print("\n--- [3] 학습 시작 (Training) ---")
for epoch in range(2001):
    # forward() 가 실행될 때, 첫 번째 루프에서만 내부 로그가 쫙 찍힘
    hypothesis = model(x_data)
    
    loss = criterion(hypothesis, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# ==========================================
# 5. 모델 저장
# ==========================================
torch.save(model.state_dict(), 'ulca_admission_model.pth')
print("\n모델 저장 완료: ulca_admission_model.pth")