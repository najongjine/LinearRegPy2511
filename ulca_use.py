import torch
import torch.nn as nn

# 1. 모델 구조 정의 (train.py와 똑같아야 함)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(3, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# 2. 모델 껍데기 생성 및 가중치 로드
model = SimpleNet()

# 저장된 파일 불러오기
try:
    model.load_state_dict(torch.load('ulca_admission_model.pth'))
    model.eval() # 평가 모드로 전환 (중요)
    print("모델 로드 성공!")
except FileNotFoundError:
    print("오류: admission_model.pth 파일이 없습니다. train.py를 먼저 실행하세요.")
    exit()

# 3. 실전 예측 함수
def predict_admission(gre, gpa, rank):
    # 입력값 정규화 (학습때랑 똑같이 나눠줘야 함)
    inputs = torch.FloatTensor([[gre/800.0, gpa/4.0, rank/4.0]])
    
    with torch.no_grad(): # 예측할 땐 기울기 계산 필요 없음
        result = model(inputs)
    
    probability = result.item() * 100
    print(f"\n[스펙] GRE: {gre}, GPA: {gpa}, Rank: {rank}")
    print(f"▶ 합격 확률: {probability:.2f}%")
    print(f"▶ 결과 예측: {'합격' if probability > 50 else '불합격'}")

# --- 테스트 ---
# 내 점수 넣어서 확인
predict_admission(790, 3.8, 1) # 공부 잘한 애
predict_admission(300, 2.5, 4) # 공부 안한 애