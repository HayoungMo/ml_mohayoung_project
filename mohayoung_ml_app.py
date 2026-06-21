# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student-por.csv"
REGRESSION_RESULT_PATH = BASE_DIR / "outputs" / "regression_result.csv"
CLASSIFICATION_RESULT_PATH = BASE_DIR / "outputs" / "classification_result.csv"


def grade_group(score: float) -> str:
    if score <= 9:
        return "위험"
    if score <= 14:
        return "보통"
    return "우수"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["grade_group"] = df["G3"].apply(grade_group)
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    x = df.drop(columns=["G3", "grade_group"])
    y_reg = df["G3"]
    y_clf = df["grade_group"]

    numeric_cols = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = x.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    regression_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    classification_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    regression_model.fit(x, y_reg)
    classification_model.fit(x, y_clf)
    return regression_model, classification_model, x


def default_row(df: pd.DataFrame) -> dict:
    row = {}
    for col in df.drop(columns=["G3", "grade_group"]).columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = int(df[col].median())
        else:
            row[col] = df[col].mode()[0]
    return row


st.set_page_config(page_title="학생 성적 예측", page_icon="ML", layout="centered")

st.title("학생 생활패턴 기반 성적 예측")
st.caption("UCI Student Performance Dataset 기반 머신러닝 개인 프로젝트")

df = load_data()
regression_model, classification_model, x_train = train_models(df)
row = default_row(df)

st.subheader("입력 정보")

col1, col2 = st.columns(2)

with col1:
    row["age"] = st.slider("나이", 15, 22, int(row["age"]))
    row["studytime"] = st.selectbox(
        "주간 공부시간",
        [1, 2, 3, 4],
        index=[1, 2, 3, 4].index(int(row["studytime"])),
        format_func=lambda x: {
            1: "1: 2시간 미만",
            2: "2: 2~5시간",
            3: "3: 5~10시간",
            4: "4: 10시간 이상",
        }[x],
    )
    row["failures"] = st.slider("과거 낙제 횟수", 0, 4, int(row["failures"]))
    row["absences"] = st.slider("결석일수", 0, int(df["absences"].max()), int(row["absences"]))
    row["G1"] = st.slider("G1 1차 성적", 0, 20, int(row["G1"]))
    row["G2"] = st.slider("G2 2차 성적", 0, 20, int(row["G2"]))

with col2:
    row["health"] = st.slider("건강상태 (1: 매우 나쁨, 5: 매우 좋음)", 1, 5, int(row["health"]))
    row["famrel"] = st.slider("가족관계 (1: 매우 나쁨, 5: 매우 좋음)", 1, 5, int(row["famrel"]))
    row["freetime"] = st.slider("자유시간 (1: 매우 적음, 5: 매우 많음)", 1, 5, int(row["freetime"]))
    row["goout"] = st.slider("외출 빈도 (1: 매우 낮음, 5: 매우 높음)", 1, 5, int(row["goout"]))
    row["internet"] = st.selectbox("인터넷 사용 가능 여부", ["yes", "no"], index=0 if row["internet"] == "yes" else 1)
    row["higher"] = st.selectbox("진학 의향", ["yes", "no"], index=0 if row["higher"] == "yes" else 1)

input_df = pd.DataFrame([row], columns=x_train.columns)

if st.button("예측하기"):
    pred_score = regression_model.predict(input_df)[0]
    pred_group = classification_model.predict(input_df)[0]

    st.subheader("예측 결과")
    st.metric("예상 최종 성적 G3", f"{pred_score:.2f}점")
    st.metric("예상 학업 수준", pred_group)

    st.write(
        "모델 B 방식처럼 생활패턴 정보에 G1, G2 성적을 함께 사용하여 "
        "최종 성적과 학업 위험군을 예측했습니다."
    )

st.subheader("모델 성능 요약")

if REGRESSION_RESULT_PATH.exists():
    st.write("회귀 모델 결과")
    st.dataframe(pd.read_csv(REGRESSION_RESULT_PATH), use_container_width=True)

if CLASSIFICATION_RESULT_PATH.exists():
    st.write("분류 모델 결과")
    clf_result = pd.read_csv(CLASSIFICATION_RESULT_PATH).drop_duplicates()
    st.dataframe(clf_result, use_container_width=True)

with st.expander("프로젝트 설명"):
    st.write(
        """
        이 프로젝트는 학생의 생활패턴, 학교생활, 가정환경, 결석 정보와 과거 성적을 활용하여
        최종 성적 G3를 예측하는 머신러닝 프로젝트입니다.

        회귀 모델은 G3 점수 자체를 예측하고, 분류 모델은 G3를 위험/보통/우수 그룹으로 나누어 예측합니다.
        시연 앱에서는 성능이 가장 좋았던 모델 B 관점에 맞춰 G1, G2 성적을 함께 입력받습니다.
        """
    )
