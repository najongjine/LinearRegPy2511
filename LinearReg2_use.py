import joblib
import pandas as pd

# 1. 모델 로드 수정
# LinearReg2.py에서 저장한 파일명('study_multivar_reg.pkl')으로 변경해야 합니다.
loaded_model = joblib.load('study_multivar_reg.pkl')
print("로딩 완료! 다차원 예측 시스템 가동.\n")

# 2. 입력 데이터 설정 (변수가 3개 필요함)
hours = 10.0       # 공부 시간
difficulty = 3     # 난이도 (1:쉬움 ~ 5:어려움)
condition = 8.0    # 컨디션 (수면시간 등)

# 3. 입력 데이터를 DataFrame으로 변환
# [중요] 학습할 때 사용한 Feature(컬럼) 3개를 모두 넣어야 에러가 안 납니다.
input_data = pd.DataFrame({
    'Study_Hours': [hours],
    'Difficulty': [difficulty],
    'Condition': [condition]
})

# 4. 예측 수행
predicted_score = loaded_model.predict(input_data)

# 5. 결과 출력
print(f"--------------------------------")
print(f"🕒 공부 시간 : {hours}시간")
print(f"🔥 과목 난이도: {difficulty} (1~5)")
print(f"😊 컨디션    : {condition} (수면시간)")
print(f"--------------------------------")
print(f"💯 예상 점수 : {predicted_score[0]:.2f}점")
print(f"--------------------------------\n")