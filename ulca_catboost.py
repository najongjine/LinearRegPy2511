# 1. 라이브러리 설치 (필요한 경우 주석(#)을 지우고 실행하세요)
# !pip install catboost pandas scikit-learn

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 2. 데이터 불러오기
# 업로드하신 파일명을 그대로 사용합니다.
df = pd.read_csv('ulca_admission_data.csv')

# 데이터 확인
print("데이터 샘플:")
print(df.head())
print("-" * 30)

# 3. 데이터 전처리 (Features와 Target 분리)
# X: 입력 변수 (gre, gpa, rank)
# y: 타겟 변수 (admit)
X = df[['gre', 'gpa', 'rank']]
y = df['admit']

# rank는 숫자로 되어 있지만 의미상 '등급'이므로 범주형 변수(Categorical Feature)로 지정합니다.
# CatBoost는 이를 지정해주면 더 똑똑하게 학습합니다.
cat_features = ['rank']

# 학습용(Train)과 테스트용(Test) 데이터 분리 (8:2 비율)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. CatBoost 모델 생성 및 학습
model = CatBoostClassifier(
    iterations=500,         # 반복 횟수
    learning_rate=0.05,     # 학습률
    depth=6,                # 트리 깊이
    cat_features=cat_features, # 범주형 변수 지정
    verbose=100             # 100번마다 학습 과정 출력
)

print("모델 학습을 시작합니다...")
model.fit(X_train, y_train)

# 5. 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("-" * 30)
print(f"모델 정확도: {accuracy:.4f}")
print("\n분류 보고서:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 6. 성적에 따른 합격 예측 함수 만들기
# ---------------------------------------------------------
def predict_admission(gre, gpa, rank):
    """
    새로운 점수를 입력받아 합격 확률을 예측하는 함수
    """
    input_data = pd.DataFrame({
        'gre': [gre],
        'gpa': [gpa],
        'rank': [rank]
    })
    
    # 확률 예측 (0일 확률, 1일 확률)
    prediction_prob = model.predict_proba(input_data)
    
    # 결과 예측 (0 또는 1)
    prediction = model.predict(input_data)
    
    fail_prob = prediction_prob[0][0] * 100
    pass_prob = prediction_prob[0][1] * 100
    
    print(f"=== 예측 결과 (GRE: {gre}, GPA: {gpa}, Rank: {rank}) ===")
    if prediction[0] == 1:
        print(f"결과: 🟢 합격 예측 (확률: {pass_prob:.1f}%)")
    else:
        print(f"결과: 🔴 불합격 예측 (확률: {fail_prob:.1f}%)")
    print("-" * 50)

# === 사용 예시 ===
# 여기에 원하는 점수를 넣어보세요.
# rank=1 (좋은대학) ~ rank=4 (안좋은대학)

# 예시 1: 성적이 높고 좋은 대학 출신
predict_admission(gre=780, gpa=3.9, rank=1)

# 예시 2: 성적은 보통이고 대학 등급이 낮은 경우
predict_admission(gre=500, gpa=3.0, rank=4)