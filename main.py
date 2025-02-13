from fastapi import FastAPI
import numpy as np
import pandas as pd
import os
import requests
import datetime
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "0213_xgboost_stock_model.json")

xgb_model = XGBRegressor()
xgb_model.load_model(model_path)

# 🔹 2️⃣ 사용자 지정 종목 리스트
user_top_10_stocks = ['고구마켓 주식회사', '토마토탈 주식회사', '(주)딸기사세요', '감자 주식회사', '나주배랑깨',
                      '배추컴퍼니', '양파이낸스', '(주)블루베리굿', '무야호 농업회사법인', '감귤로벌 주식회사']

# 🔹 3️⃣ One-Hot Encoding 설정
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(np.array(user_top_10_stocks).reshape(-1, 1))  # 종목 리스트를 학습

# 🔹 4️⃣ 기상청 API를 통해 실시간 기온 및 강수량 가져오기
def get_weather_data():
    BASE_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
    SERVICE_KEY = ""

    # 🔹 오늘 날짜 및 정시 기준 시간 설정
    today = datetime.datetime.now().strftime("%Y%m%d")
    current_time = datetime.datetime.now()
    base_time = current_time.replace(minute=0, second=0, microsecond=0).strftime("%H%M")

    # 🔹 API 요청 파라미터 설정
    params = {
        "serviceKey": SERVICE_KEY,
        "numOfRows": 10,
        "pageNo": 1,
        "dataType": "JSON",
        "base_date": today,
        "base_time": base_time,
        "nx": 63,  # 전주시 좌표
        "ny": 89,  # 전주시 좌표
    }

    try:
        # 🔹 API 요청 및 응답 데이터 받기
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        # 🔹 기온(T1H)과 강수량(RN1) 추출
        weather_data = {}
        for item in data['response']['body']['items']['item']:
            if item['category'] == 'T1H':  # 기온
                weather_data['temperature'] = float(item['obsrValue'])
            elif item['category'] == 'RN1':  # 강수량
                weather_data['rainfall'] = float(item['obsrValue'])

        # 🔹 만약 강수량 데이터가 없다면 기본값 0.0으로 설정
        return {
            "temperature": weather_data.get("temperature", 0.0),
            "rainfall": weather_data.get("rainfall", 0.0),
        }

    except Exception as e:
        print(f"⚠️ 기상청 API 요청 실패: {e}")
        return {"temperature": 0.0, "rainfall": 0.0}  # API 요청 실패 시 기본값 반환

# 🔹 5️⃣ FastAPI 엔드포인트: 주가 예측
@app.get("/predict")
def predict():
    # 🔹 실시간 날씨 데이터 가져오기
    weather = get_weather_data()
    today_weather_array = np.array([[weather["temperature"], weather["rainfall"], np.log1p(weather["rainfall"])]])

    predictions = []
    for stock in user_top_10_stocks:
        # 🔹 One-Hot Encoding 적용 (각 종목별로 다르게 변환)
        stock_one_hot = encoder.transform([[stock]])  # 종목별로 다른 인코딩 적용

        # 🔹 입력 데이터 결합
        today_input = np.concatenate((today_weather_array, stock_one_hot), axis=1)

        # 🔹 XGBoost 모델 예측
        predicted_price_log = xgb_model.predict(today_input)[0]
        predicted_price = np.expm1(predicted_price_log)  # 로그 변환 복원

        # 🔹 ±2% 변동 적용
        delta_percentage = 0.02
        predicted_price_variation = np.random.uniform(
            predicted_price * (1 - delta_percentage),
            predicted_price * (1 + delta_percentage)
        )

        # 🔹 반올림 적용 (10원 단위)
        predicted_price = round(predicted_price / 10) * 10
        predicted_price_variation = round(predicted_price_variation / 10) * 10

        predictions.append({
            "stock": stock,
            "predicted_close": predicted_price,
            "real_time_variation": predicted_price_variation
        })

    return {
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "weather": weather,  # 🔹 기온 및 강수량 추가
        "predictions": predictions
    }
