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

st.set_page_config(
    page_title="학생 성적 예측",
    layout="wide"
)

st.title("학생 생활패턴 기반 성적 예측")
st.caption("UCI Student Performance Dataset 기반 머신러닝 개인 프로젝트")

st.subheader("데이터 요약")

summary_cols = st.columns(4)

with summary_cols[0]:
    st.caption("전체 학생 수")
    st.markdown("### 649명")

with summary_cols[1]:
    st.caption("평균 최종 성적")
    st.markdown("### 11.91점")

with summary_cols[2]:
    st.caption("학업 수준 분포")
    st.markdown("**위험** 100명  \n**보통** 418명  \n**우수** 131명")

with summary_cols[3]:
    st.caption("최고 성능 모델")
    st.markdown("회귀: **LinearRegression_B**  \n분류: **LogisticRegression_B**")


df = load_data()
regression_model, classification_model, x_train = train_models(df)
row = default_row(df)

st.subheader("입력 정보")

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("#### 학교생활 정보")
    age = st.number_input("나이", min_value=15, max_value=22, value=17, step=1)
    studytime = st.selectbox(
        "주간 공부시간",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1: 2시간 미만",
            2: "2: 2~5시간",
            3: "3: 5~10시간",
            4: "4: 10시간 이상"
        }[x],
        index=1
    )
    row["failures"] = st.slider("과거 낙제 횟수", 0, 4, int(row["failures"]))
    absences = st.number_input("결석일수", min_value=0, max_value=100, value=2, step=1)
    
    st.markdown("#### 과거 성적 정보")
    st.caption("G1, G2는 각각 1차/2차 성적이며, 데이터셋 기준 0~20점 척도의 성적입니다.")
    grade_col1, grade_col2 = st.columns(2)
    G1 = st.number_input("G1 1차 성적 (0~20점)", min_value=0, max_value=20, value=11, step=1)
    G2 = st.number_input("G2 2차 성적 (0~20점)", min_value=0, max_value=20, value=11, step=1)
    
with right_col:
    st.markdown("#### 생활 정보")
    health = st.selectbox(
        "건강상태",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x}: " + {
            1: "매우 나쁨",
            2: "나쁨",
            3: "보통",
            4: "좋음",
            5: "매우 좋음"
        }[x],
        index=2
    )
    health = st.selectbox(
        "가족관계",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x}: " + {
            1: "매우 나쁨",
            2: "나쁨",
            3: "보통",
            4: "좋음",
            5: "매우 좋음"
        }[x],
        index=2
    )
    health = st.selectbox(
        "자유시간",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x}: " + {
            1: "매우 나쁨",
            2: "나쁨",
            3: "보통",
            4: "좋음",
            5: "매우 좋음"
        }[x],
        index=2
    )
    health = st.selectbox(
        "외출 빈도",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x}: " + {
            1: "매우 나쁨",
            2: "나쁨",
            3: "보통",
            4: "좋음",
            5: "매우 좋음"
        }[x],
        index=2
    )
    internet = st.radio("인터넷 사용 가능 여부", ["yes", "no"], horizontal=True)
    higher = st.radio("진학 의향", ["yes", "no"], horizontal=True)

input_df = pd.DataFrame([row], columns=x_train.columns)

if st.button("예측하기"):
    pred_score = regression_model.predict(input_df)[0]
    pred_group = classification_model.predict(input_df)[0]

    st.subheader("예측 결과")
    st.metric("예상 최종 성적 G3", f"{pred_score:.1f}점 / 20점")
    st.metric("예상 학업 수준", pred_group)
    st.info("모델 B는 생활패턴 정보에 G1, G2 과거 성적을 추가하여 예측합니다.")
    if pred_group == "위험":
        st.warning("예측 결과 학업 위험군으로 분류되었습니다. 결석일수와 과거 성적 변화에 주의가 필요합니다.")
    elif pred_group == "보통":
        st.info("예측 결과 보통 그룹으로 분류되었습니다. 생활패턴과 성적 변화를 함께 확인할 수 있습니다.")
    else:
        st.success("예측 결과 우수 그룹으로 분류되었습니다. 현재 입력값 기준으로 성적 예측이 긍정적으로 나타났습니다.")


st.subheader("모델 성능 요약")

with st.expander("모델 성능 비교 보기", expanded=False):
    if REGRESSION_RESULT_PATH.exists():
        st.write("회귀 모델 결과")
        reg_result = pd.read_csv(REGRESSION_RESULT_PATH)
        st.dataframe(reg_result, use_container_width=True)

if CLASSIFICATION_RESULT_PATH.exists():
    st.write("분류 모델 결과")
    clf_result = pd.read_csv(CLASSIFICATION_RESULT_PATH)
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
