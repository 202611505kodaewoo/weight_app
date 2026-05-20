# 남자 모델만 출력 

import streamlit as st
import numpy as np
import joblib

st.title("신체 정보를 이용한 몸무게 예측 머신러닝 모델")
st.write("신체 정보를 입력하면 몸무게를 예측합니다.")

model = joblib.load("weight_model_male.pkl")

st.sidebar.header("머신러닝 모델 설계 실습 (다중회귀)")

height = st.slider("키 (cm)", 140.0, 190.0, 170.0)
neck = st.slider("목 둘레 (cm)", 25.0, 50.0, 37.0)
head = st.slider("머리 둘레 (cm)", 50.0, 65.0, 57.0)
waist = st.slider("젖가슴 둘레 (cm)", 70.0, 130.0, 100.0)

X = np.array([[height, neck, head, waist]])

if st.button("몸무게 예측하기"):
    prediction = model.predict(X)
    st.success(f"예측 몸무게 : {prediction[0]:.1f} kg")