"""
https://www.kaggle.com/datasets/meirnizri/covid19-dataset
"""
import kagglehub
import shutil
import os

""" 데이터 다운로드 """
# 1. 내 프로젝트 루트에 저장할 폴더 이름 설정
local_dataset_dir = "./covid_data"

# 2. 이미 다운로드 받았는지 확인 (Skip 로직)
if os.path.exists(local_dataset_dir):
    print(f"✅ 데이터셋이 이미 존재합니다. 다운로드를 건너뜁니다. ({local_dataset_dir})")
    
else:
    print("⬇️ 데이터셋 다운로드를 시작합니다 (kagglehub)...")
    
    # 3. kagglehub로 다운로드 (일단 캐시 폴더에 받아짐)
    cache_path = kagglehub.dataset_download("meirnizri/covid19-dataset")
    
    print(f"📦 캐시된 경로에서 프로젝트 폴더로 복사 중...")
    
    # 4. 캐시된 데이터를 내 프로젝트 폴더로 복사
    # copytree는 폴더 전체를 복사합니다.
    shutil.copytree(cache_path, local_dataset_dir)
    
    print(f"✅ 완료! 데이터가 프로젝트 경로에 저장되었습니다: {local_dataset_dir}")

""" 데이터 다운로드 END """

""" Pandas 로 다운로드 받은 데이터 읽기"""
import pandas as pd
import os
# 1. 파일 경로 설정 (이미지에서 확인한 경로)
csv_file_path = "./covid_data/Covid Data.csv"

# 2. 파일이 있는지 확인 후 읽기
if os.path.exists(csv_file_path):
    print(f"📂 파일 읽기 시작: {csv_file_path}")
    
    # 데이터 로드
    df = pd.read_csv(csv_file_path)
    
    print("✅ 데이터 로드 성공!")
    print(f"📊 데이터 크기(행, 열): {df.shape}")
    
    # -------------------------------------------------------
    # [추가] 아까 이야기한 타겟 데이터(정답지) 만들기
    # classification: 1~3은 확진(1), 4 이상은 비확진(0)
    # -------------------------------------------------------
    df['is_covid'] = df['CLASIFFICATION_FINAL'].apply(lambda x: 1 if x < 4 else 0)

    # 함수 정의: '9999-99-99'면 0(생존), 아니면 1(사망)
    def check_death(date):
        if date == '9999-99-99':
            return 0
        else:
            return 1

    # 새로운 컬럼 'is_dead' 생성
    df['is_dead'] = df['DATE_DIED'].apply(check_death)

    # 처리가 끝났으니 원래 날짜 컬럼은 삭제 (모델에 방해됨)
    df = df.drop(columns=['DATE_DIED'])

    print(df['is_dead'].value_counts())
    
    # 결과 확인 (처음 5줄)
    print("\n[데이터 미리보기 (상위 5개)]")
    print(df.head())
    
    # 정답 비율 확인
    print("\n[정답 클래스 비율]")
    print(df['is_covid'].value_counts())

else:
    print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요: {csv_file_path}")
""" Pandas 로 다운로드 받은 데이터 읽기 END"""

""" X 와 y 로 나누기 """
# 1. y (정답) 설정: 코로나 양성 여부
y = df['is_covid']

# 2. X (입력/문제지) 설정
# 정답인 'is_covid' 제외
# 정답의 원본인 'CLASIFFICATION_FINAL'도 반드시 제외 (이거 안 빼면 정확도 100% 나옴 -> Data Leakage)
X = df.drop(columns=['is_covid', 'CLASIFFICATION_FINAL'])

print(f"✅ 데이터 분리 완료!")
print(f"X (입력 데이터) 크기: {X.shape}")
print(f"y (정답 데이터) 크기: {y.shape}")

# X에 어떤 컬럼들이 남았는지 확인
print("\n[X 컬럼 목록 (모델에 들어갈 항목들)]")
print(X.columns.tolist())
""" X 와 y 로 나누기 END """

""" 훈련 데이터와 test 데이터 나누기 """
from sklearn.model_selection import train_test_split

# 1. 데이터 분리 (Train: 80%, Test: 20%)
# shuffle=True: 데이터를 무작위로 섞습니다 (기본값이 True이지만 명시했습니다)
# random_state=42: 실행할 때마다 똑같이 섞이도록 고정 (재현성 확보)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    shuffle=True, 
    random_state=42
)

print("✅ 데이터 분리 완료!")
print(f"훈련용 데이터(X_train): {X_train.shape}")
print(f"테스트용 데이터(X_test):  {X_test.shape}")
print(f"훈련용 정답(y_train): {y_train.shape}")
print(f"테스트용 정답(y_test):  {y_test.shape}")
""" 훈련 데이터와 test 데이터 나누기 END """


""" 모델에 데이터 때려넣기 """
# https://gemini.google.com/share/97c9ff213c86
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # 모델 저장을 위한 라이브러리

# 1. 모델 생성 (LinearRegression 대신 분류용인 LogisticRegression 사용)
# max_iter=1000: 데이터가 많으면 학습 횟수를 좀 늘려줘야 에러가 안 납니다.
model = LogisticRegression(max_iter=1000)

print("🚀 모델 학습을 시작합니다... (데이터 때려넣는 중)")

# 2. 모델 학습 (fit) -> 기출문제(X_train)와 정답(y_train)을 줌
model.fit(X_train, y_train)
print("✅ 모델 학습 완료!")

# 3. 검증 (predict) -> 시험문제(X_test)를 풀게 시킴
print("📝 테스트 데이터로 예측 중...")
y_pred = model.predict(X_test)

# 4. 정확도 및 성능 평가
acc = accuracy_score(y_test, y_pred)
print("-" * 30)
print(f"🏆 정확도 (Accuracy): {acc * 100:.2f}%")
print("-" * 30)

# [추가 정보] 혼동 행렬 (맞춘 개수 상세 확인)
# [[진짜음성맞춤, 가짜양성(틀림)],
#  [가짜음성(틀림), 진짜양성맞춤]]
cm = confusion_matrix(y_test, y_pred)
print("📊 혼동 행렬 (Confusion Matrix):")
print(cm)

# 5. 모델 저장하기 (내 컴퓨터에 파일로 저장)
model_filename = 'covid_prediction_model.pkl'
joblib.dump(model, model_filename)
print(f"\n💾 모델이 저장되었습니다: {model_filename}")
""" 모델에 데이터 때려넣기 END """