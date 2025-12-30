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
from catboost import CatBoostClassifier # 로지스틱 대신 캣부스트 임포트
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
# 1. CatBoost에게 "이 컬럼들은 숫자가 아닙니다"라고 알려줄 리스트 작성
# AGE(나이)와 is_dead(우리가 만든 0/1)를 제외한 모든 명목형 변수들
cat_features_names = [
    'USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'PNEUMONIA', 'PREGNANT', 
    'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 
    'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU', 'INTUBED'
]

# 데이터프레임에서 이 컬럼들이 몇 번째 열(index)에 있는지 찾기
# (CatBoost는 컬럼 이름 대신 인덱스를 좋아합니다)
cat_features_indices = [X.columns.get_loc(col) for col in cat_features_names if col in X.columns]

print(f"🐱 범주형 변수 {len(cat_features_indices)}개를 식별했습니다.")

# 2. 모델 생성
model = CatBoostClassifier(
    iterations=500,        # [중요] 기본값 1000 -> 500으로 절반 축소
    depth=6,               # 트리 깊이 (너무 깊으면 느려짐, 적당히 6)
    random_state=42, 
    verbose=50,            # [중요] 0 -> 50 (50번 돌 때마다 로그 찍힘, 멈춘 거 아님을 확인)
    early_stopping_rounds=20 # 성능 안 오르면 20번 기다리다 그냥 종료 (시간 절약)
)

print("🚀 모델 학습을 시작합니다... (범주형 처리 적용됨)")

# 3. 모델 학습 (fit) - 여기서 cat_features를 꼭 넣어줘야 함!
model.fit(
    X_train, y_train, 
    cat_features=cat_features_indices  # <--- 핵심 포인트!
)
print("✅ 모델 학습 완료!")

# 4. 검증 및 평가
print("📝 테스트 데이터로 예측 중...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("-" * 30)
print(f"🏆 정확도 (Accuracy): {acc * 100:.2f}%")
print("-" * 30)

cm = confusion_matrix(y_test, y_pred)
print("📊 혼동 행렬 (Confusion Matrix):")
print(cm)

# 5. 모델 저장
model_filename = 'covid_catboost_advanced.pkl'
joblib.dump(model, model_filename)
print(f"\n💾 모델이 저장되었습니다: {model_filename}")
""" 모델에 데이터 때려넣기 END """