import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

TARGET = "top20_spender"
TRAIN_PATH = "결과물/dataset_train.csv"

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="고지출자 예측 웹서비스", layout="wide")

JOB_LABEL_MAP = {
    1: "관리자",
    2: "전문가 및 관련 종사자",
    3: "사무 종사자",
    4: "서비스 종사자",
    5: "판매 종사자",
    6: "농림어업 숙련 종사자",
    7: "기능원 및 관련 기능 종사자",
    8: "장치, 기계 조작 및 조립 종사자",
    9: "단순 노무 종사자",
    10: "군인",
    11: "학생",
    12: "전업주부",
    13: "기타"
}

JOB_NAME_TO_CODE = {v: k for k, v in JOB_LABEL_MAP.items()}

@st.cache_data
def load_train_and_columns():
    df = pd.read_csv(TRAIN_PATH)
    X_cols = [col for col in df.columns if col != TARGET]
    return df, X_cols

def build_input_row(X_cols, user_inputs):
    """
    X_cols: 학습에 사용된 전체 feature 컬럼명 리스트
    user_input: 사용자 입력값을 담은 딕셔너리
    return: 사용자 입력값을 기반으로 한 DataFrame (1행) - 컬럼 100% 동일
    """
    x = pd.DataFrame([{c: 0 for c in X_cols}])

    if "TRIP_DAYS" in x.columns:
        x.loc[0, "TRIP_DAYS"] = user_inputs.get("TRIP_DAYS", 0)
        
    for col_prefix, value in [("AGE_GROUP", user_inputs["AGE_GRP"]),
                              ("JOB", user_inputs["JOB"]),
                              ("TRAVEL_STATUS_RESIDENCE", user_inputs["RESIDENCE"]),
                              ("TRAVEL_STATUS_DESTINATION", user_inputs["DESTINATION"]),
                              ("TRAVEL_STATUS_ACCOMPANY", user_inputs["ACCOMPANY"]),
                              ("TRAVEL_LIKE_SIDO_1", user_inputs["SIDO"]),
                              ("TRAVEL_LIKE_SGG_1", user_inputs["SGG"]),
    ]:
        col_name = f"{col_prefix}_{value}"
        if col_name in x.columns:
            x.loc[0, col_name] = 1
    return x

def get_unique_values_from_ohe(X_cols, prefix):
    """
    X_cols: 학습에 사용된 전체 feature 컬럼명 리스트
    prefix: OHE 컬럼명 접두사 (예: "AGE_GROUP", "JOB", ...)
    return: prefix에 해당하는 OHE 컬럼들의 고유값 리스트
    """
    unique_values = []
    for col in X_cols:
        if col.startswith(prefix + "_"):
            unique_values.append(col.replace(prefix + "_", ""))
    return sorted(list(set(unique_values)))

# def job_display_and_code_options(X_cols):
#     """
#     학습 데이터의 JOB_1.0, JOB_2.0 ... 컬럼만 추출해서
#     화면 표시명과 실제 모델 코드를 연결
#     """

#     job_names = []
#     name_to_code = {}

#     for col in X_cols:

#         # JOB_으로 시작
#         if not col.startswith("JOB_"):
#             continue

#         # JOB_ 제거
#         code = col[len("JOB_NM_"):]

#         # JOB_NM_10.0 같은 컬럼 제외
#         try:
#             code_int = int(float(code))
#         except (ValueError, TypeError):
#             continue

#         # 코드에 해당하는 직업명
#         job_name = JOB_LABEL_MAP.get(code_int)

#         if job_name is None:
#             continue

#         job_names.append(job_name)
#         name_to_code[job_name] = code

#     return job_names, name_to_code
def job_display_and_code_options(X_cols):
    """
    모델이 실제로 학습한 JOB_*.? 컬럼에서 코드 목록을 만들고, 
    화면에는 직업명만 보여주되 선택 결과는 코드로 반환할 수 있게 준비
    """
    job_codes = get_unique_values_from_ohe(X_cols, "JOB_NM")

    job_names = []
    for code in job_codes:
        code_int = int(float(code))
        job_names.append(JOB_LABEL_MAP.get(code_int, code))

    name_to_code = {}
    for code, name in zip(job_codes, job_names):
        name_to_code[name] = code
    return job_names, name_to_code

st.title("고지출자(Top 20%) 예측 웹서비스")
st.caption("입력한 여행/개인 정보를 기반으로 고지출자 여부(0/1)와 확률을 예측합니다.")

train_df, X_cols = load_train_and_columns()

age_options = get_unique_values_from_ohe(X_cols, "AGE_GROUP")
res_options = get_unique_values_from_ohe(X_cols, "TRAVEL_STATUS_RESIDENCE")
dest_options = get_unique_values_from_ohe(X_cols, "TRAVEL_STATUS_DESTINATION")
accompany_options = get_unique_values_from_ohe(X_cols, "TRAVEL_STATUS_ACCOMPANY")
sido_options = get_unique_values_from_ohe(X_cols, "TRAVEL_LIKE_SIDO_1")
sgg_options = get_unique_values_from_ohe(X_cols, "TRAVEL_LIKE_SGG_1")

job_name_options, job_name_to_code = job_display_and_code_options(X_cols)

st.subheader("1) 사용자 입력")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age_grp = st.selectbox("연령대(AGE_GRP)", age_options if age_options else ["20대", "30대", "40대", "50대", "60대 이상"])
    job_name = st.selectbox(
    "직업(JOB)",
    job_name_options if job_name_options else list(JOB_NAME_TO_CODE.keys())
)

job_code = job_name_to_code[job_name]

with col2:
    residence = st.selectbox("거주지(RESIDENCE)", res_options if res_options else ["서울특별시"])
    destination = st.selectbox("여행지(DESTINATION)", dest_options if dest_options else ["부산광역시"])
    accompany = st.selectbox("동행자(ACCOMPANY)", accompany_options if accompany_options else ["나홀로여행"])

with col3:
    trip_days = st.number_input("여행기간(TRIP_DAYS)", min_value=1, max_value=60, value=3, step=1)
    threshold = st.slider("고지출자 확률 임계값(Threshold)", 0.1, 0.9, 0.5, 0.05)
    show_input_vector = st.checkbox("모델 입력 벡터(원핫) 보기", value=False)

with col4:
    sido = st.selectbox("선호 여행지 시도(SIDO)", sido_options if sido_options else ["서울특별시"])
    sgg = st.selectbox("선호 여행지 시군구(SGG)", sgg_options if sgg_options else ["강남구"])

payload = {
    "AGE_GRP": age_grp,
    "JOB": job_code,
    "RESIDENCE": residence,
    "DESTINATION": destination,
    "ACCOMPANY": accompany,
    "TRIP_DAYS": trip_days,
    "SIDO": sido,
    "SGG": sgg,
    "threshold": threshold
}

st.subheader("2) 예측 결과")
if st.button("예측 실행"):
    try: 
        res = requests.post(API_URL, json=payload, timeout=10)

        if res.status_code != 200:
            st.error(f"API 오류, status={res.status_code}\n{res.text}")
        else:
            result = res.json()
            proba = float(result["proba"])
            pred = int(result["pred"])

            left, right = st.columns(2)
            with left:
                st.metric("고지출자 확률 P(y=1)", f"{proba:.4f}")
                st.progress(min(max(proba, 0.0), 1.0))
            with right:
                st.metric("예측 라벨(0/1)", pred)
                if pred == 1:
                    st.success("고지출자(1)로 예측됨")
                else:
                    st.warning("일반지출자(0)로 예측됨")
            st.caption("※ 임계값(threshold)을 조절하면 '1로 판단'하는 기준이 바뀝니다.")

            if show_input_vector:
                st.subheader("모델 입력 벡터(원핫 포함) 1행")
                st.dataframe(pd.DataFrame([result["input_vector"]]))

    except requests.exceptions.ConnectionError:
        st.error("FastAPI 서버에 연결할 수 없습니다. (서버가 실행 중인지 확인: uvicorn server:app --reload)")
    except requests.exceptions.Timeout:
        st.error("API 요청 시간이 초과되었습니다. 서버가 멈췄거나 너무 느림")
    except Exception as e:
        st.error(f"예상치 못한 에러 : {e}")

else:
    st.write("왼쪽에서 정보를 입력하고 **예측하기** 버튼을 눌러보세요.")

st.divider()
st.write("###참고##")
st.write("- 화면에서는 직업명을 보여주지만, 내부적으로는 모델이 학습한 JOB 코드로 변환해 예측합니다.")
st.write("- 이 서비스는 학습된 모델이 본 데이터 패턴을 기반으로 예측합니다.")
st.write("- 결과는 원인 분석이 아니라, 모델의 추정 결과입니다.")