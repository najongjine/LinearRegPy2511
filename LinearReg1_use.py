import joblib
import pandas as pd


loaded_model = joblib.load('study_linear_reg.pkl')
print("로딩 완료! 예측 시스템 가동.\n")
hours = float(2)
input_data = pd.DataFrame({'Study_Hours': [hours]})
# 3. 입력 데이터를 DataFrame으로 변환
# [중요] 훈련할 때 DataFrame을 줬으니, 예측할 때도 똑같은 모양(컬럼명)으로 줘야 합니다.
input_data = pd.DataFrame({'Study_Hours': [hours]})

# 4. 예측 수행
predicted_score = loaded_model.predict(input_data)

# 5. 결과 출력
print(f"--------------------------------")
print(f"🕒 공부 시간: {hours}시간")
print(f"💯 예상 점수: {predicted_score[0]:.2f}점")
print(f"--------------------------------\n")
