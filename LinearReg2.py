import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. 현실적인 다차원 데이터 생성 (Data Generation)
np.random.seed(42)
n_samples = 100 # 데이터 개수를 좀 늘렸습니다

# (1) 공부 시간 (0 ~ 10시간)
study_hours = np.random.rand(n_samples) * 10 

# (2) 과목 난이도 (1:쉬움 ~ 5:어려움, 정수)
subject_difficulty = np.random.randint(1, 6, n_samples)

# (3) 학생 컨디션/수면 시간 (평균 6시간, 표준편차 2)
condition = np.random.normal(6, 2, n_samples)
condition = np.clip(condition, 0, 12) # 0~12시간 사이로 자름

# [정답 생성 공식 (현실 반영)]
# 점수 = 기본점수 + (공부효율) - (난이도 페널티) + (컨디션 영향) + [예측불가 노이즈]
# 노이즈(* 10)을 크게 줘서 데이터가 지저분하게 만듦
scores = 30 + (6 * study_hours) - (8 * subject_difficulty) + (3 * condition) + np.random.randn(n_samples) * 10

# 점수가 0~100점 사이가 되도록 보정 (현실성)
scores = np.clip(scores, 0, 100)

# DataFrame 생성
df = pd.DataFrame({
    'Study_Hours': study_hours,
    'Difficulty': subject_difficulty,
    'Condition': condition,
    'Scores': scores
})

print("--- Data Sample ---")
print(df.head())
print("\n--- Correlation (상관계수) ---")
print(df.corr()['Scores']) # 각 변수가 점수랑 얼마나 관계있는지 확인

# 2. 모델 학습 (Model Training)
# X는 이제 컬럼이 3개입니다 (다차원)
X = df[['Study_Hours', 'Difficulty', 'Condition']]
y = df['Scores']

model = LinearRegression()
model.fit(X, y)

# 3. 예측 및 결과 확인
predicted_scores = model.predict(X)

print("\n--- 학습 결과 ---")
# 다차원이므로 기울기(coef)가 변수 개수만큼 나옵니다.
features = ['공부시간', '난이도', '컨디션']
for i, col_name in enumerate(features):
    print(f"{col_name}의 영향력(기울기): {model.coef_[i]:.2f}")
    
print(f"절편 (기본 점수): {model.intercept_:.2f}")
print(f"결정계수(R^2, 정확도): {model.score(X, y):.2f}") 
# R^2가 1에 가까울수록 완벽, 현실 데이터면 0.5~0.7 정도 나올 수 있음

# 4. 시각화 (Visualization)
# 변수가 3개라 선 하나로 표현 불가능 -> '실제값 vs 예측값' 비교 그래프 사용
plt.figure(figsize=(8, 6))

plt.scatter(y, predicted_scores, color='green', alpha=0.6)
# 완벽하게 예측했다면 모든 점이 이 빨간 점선 위에 있어야 함
plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Perfect Fit')

plt.title('Actual Scores vs Predicted Scores')
plt.xlabel('Actual Scores (Truth)')
plt.ylabel('Predicted Scores (Model)')
plt.legend()
plt.grid(True)
plt.show()

# 5. 모델 저장
import joblib
joblib.dump(model, 'study_multivar_reg.pkl')
print("\n모델 저장이 완료되었습니다: study_multivar_reg.pkl")