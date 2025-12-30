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
    
    # 결과 확인 (처음 5줄)
    print("\n[데이터 미리보기 (상위 5개)]")
    print(df.head())
    
    # 정답 비율 확인
    print("\n[정답 클래스 비율]")
    print(df['is_covid'].value_counts())

else:
    print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요: {csv_file_path}")
""" Pandas 로 다운로드 받은 데이터 읽기 END"""