import pandas as pd
import numpy as np

# 재현성을 위해 시드 설정
np.random.seed(42)

# 데이터 개수 설정 (1000명)
n_samples = 1000

# 1. Feature 데이터 생성
# GRE: 200~800점 사이 정수
gre = np.random.randint(290, 801, n_samples)
# GPA: 2.0~4.0 사이 실수
gpa = np.round(np.random.uniform(2.0, 4.0, n_samples), 2)
# Rank: 1~4 (대학 레벨, 1이 가장 높음)
rank = np.random.randint(1, 5, n_samples)

# 2. Target (admit) 데이터 생성 로직
# 합격 확률 계산: (GRE 반영) + (GPA 반영) - (Rank 반영: 숫자가 클수록 감점) + (랜덤 노이즈)
# 이 수식은 단순히 데이터의 경향성을 만들기 위함입니다.
score = (gre / 800) * 0.45 + (gpa / 4.0) * 0.45 - (rank * 0.05) + np.random.normal(0, 0.05, n_samples)

# 상위 40% 정도를 합격(1)으로 설정
threshold = np.percentile(score, 60) 
admit = (score > threshold).astype(int)

# 3. 데이터프레임 생성
df = pd.DataFrame({
    'admit': admit,
    'gre': gre,
    'gpa': gpa,
    'rank': rank
})

# 4. CSV 파일로 저장
df.to_csv('admission_data.csv', index=False)

print("데이터 생성 완료! 상위 5개 데이터 확인:")
print(df.head())