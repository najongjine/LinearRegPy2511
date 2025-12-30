"""
https://www.kaggle.com/datasets/meirnizri/covid19-dataset
"""
import kagglehub
import shutil
import os

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