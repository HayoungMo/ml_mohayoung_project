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


def default_regression_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": ["LinearRegression_A", "LinearRegression_B", "KNNRegression_A", "KNNRegression_B"],
            "MAE": [2.156382, 0.765060, 2.275385, 1.556923],
            "MSE": [8.189784, 1.475909, 9.066154, 4.646769],
            "RMSE": [2.861780, 1.214870, 3.011005, 2.155637],
            "R2": [0.160170, 0.848651, 0.070302, 0.523492],
        }
    )


def default_classification_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": [
                "LogisticRegression_A",
                "LogisticRegression_B",
                "KNNClassifier_A",
                "KNNClassifier_B",
                "SVM_A",
                "SVM_B",
            ],
            "Accuracy": [0.669231, 0.869231, 0.584615, 0.715385, 0.661538, 0.823077],
            "Precision": [0.643905, 0.869617, 0.560769, 0.740321, 0.649559, 0.849212],
            "Recall": [0.669231, 0.869231, 0.584615, 0.715385, 0.661538, 0.823077],
            "F1": [0.636895, 0.869091, 0.545359, 0.686088, 0.554978, 0.799388],
        }
    )


def load_result_table(path: Path, fallback: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path).drop_duplicates()
    return fallback


def format_score(score: float) -> str:
    return f"{score:.1f}점 / 20점"


def group_message(group: str) -> tuple[str, str]:
    if group == "위험":
        return (
            "학업 위험군",
            "현재 입력값에서는 위험 그룹으로 분류됩니다. 결석일수와 과거 성적 변화를 함께 확인할 필요가 있습니다.",
        )
    if group == "보통":
        return (
            "보통 그룹",
            "현재 입력값에서는 보통 그룹으로 분류됩니다. 생활패턴과 과거 성적 변화에 따라 결과가 달라질 수 있습니다.",
        )
    return (
        "우수 그룹",
        "현재 입력값에서는 우수 그룹으로 분류됩니다. 과거 성적과 생활패턴 기준에서 긍정적인 예측 결과입니다.",
    )


def select_scale(label: str, options: dict[int, str], default: int = 3) -> int:
    return st.selectbox(
        label,
        options=list(options.keys()),
        format_func=lambda x: f"{x}: {options[x]}",
        index=list(options.keys()).index(default),
    )


st.set_page_config(
    page_title="학생 성적 예측",
    page_icon="ML",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.1rem;
        max-width: 1180px;
    }
    .app-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.2rem;
    }
    .muted {
        color: #64748b;
        font-size: 0.96rem;
    }
    .panel {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .result-main {
        border-left: 5px solid #ef4444;
        background: #fff7ed;
    }
    .result-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    .result-value {
        font-size: 2.0rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.15;
    }
    .small-note {
        font-size: 0.9rem;
        color: #475569;
    }
    .insight-box {
        border-left: 4px solid #2563eb;
        background: #eff6ff;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin-top: 1rem;
        color: #1e3a8a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


df = load_data()
regression_model, classification_model, x_train = train_models(df)
default_student_row = default_row(df)
if "student_row" not in st.session_state:
    st.session_state.student_row = default_student_row.copy()
else:
    for key, value in default_student_row.items():
        st.session_state.student_row.setdefault(key, value)
row = st.session_state.student_row.copy()

grade_counts = df["grade_group"].value_counts().reindex(["위험", "보통", "우수"]).fillna(0).astype(int)
g3_mean = df["G3"].mean()
age_min, age_max = int(df["age"].min()), int(df["age"].max())
failures_min, failures_max = int(df["failures"].min()), int(df["failures"].max())
absences_min, absences_max = int(df["absences"].min()), int(df["absences"].max())


@st.dialog("예측할 학생 정보 입력", width="large")
def open_student_dialog():
    edit_row = st.session_state.student_row.copy()
    st.info(
        "포르투갈 중등학생 Student Performance Dataset 기준입니다. "
        "성적 G1, G2, G3는 한국식 100점 만점이 아니라 0~20점 척도입니다."
    )

    with st.form("student_input_form"):
        left, right = st.columns(2)

        with left:
            st.subheader("학교생활 정보")
            edit_row["age"] = st.number_input(
                f"나이 ({age_min}~{age_max}세)",
                min_value=age_min,
                max_value=age_max,
                value=int(edit_row["age"]),
                step=1,
            )
            edit_row["studytime"] = st.selectbox(
                "주간 공부시간 (1~4단계)",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1: 2시간 미만",
                    2: "2: 2~5시간",
                    3: "3: 5~10시간",
                    4: "4: 10시간 이상",
                }[x],
                index=[1, 2, 3, 4].index(int(edit_row["studytime"])),
            )
            edit_row["failures"] = st.number_input(
                f"과거 낙제 횟수 ({failures_min}~{failures_max}회)",
                min_value=failures_min,
                max_value=failures_max,
                value=int(edit_row["failures"]),
                step=1,
            )
            edit_row["absences"] = st.number_input(
                f"결석일수 ({absences_min}~{absences_max}일)",
                min_value=absences_min,
                max_value=absences_max,
                value=int(edit_row["absences"]),
                step=1,
            )

            st.subheader("과거 성적 정보")
            st.caption("G1, G2는 포르투갈 학생 성적 데이터 기준 0~20점 척도입니다.")
            edit_row["G1"] = st.number_input("G1 1차 성적 (0~20점)", min_value=0, max_value=20, value=int(edit_row["G1"]), step=1)
            edit_row["G2"] = st.number_input("G2 2차 성적 (0~20점)", min_value=0, max_value=20, value=int(edit_row["G2"]), step=1)

        with right:
            st.subheader("생활 정보")
            edit_row["health"] = select_scale(
                "건강상태 (1~5단계, 1: 매우 나쁨 / 5: 매우 좋음)",
                {1: "매우 나쁨", 2: "나쁨", 3: "보통", 4: "좋음", 5: "매우 좋음"},
                default=int(edit_row["health"]),
            )
            edit_row["famrel"] = select_scale(
                "가족관계 (1~5단계, 1: 매우 나쁨 / 5: 매우 좋음)",
                {1: "매우 나쁨", 2: "나쁨", 3: "보통", 4: "좋음", 5: "매우 좋음"},
                default=int(edit_row["famrel"]),
            )
            edit_row["freetime"] = select_scale(
                "자유시간 (1~5단계, 1: 매우 적음 / 5: 매우 많음)",
                {1: "매우 적음", 2: "적음", 3: "보통", 4: "많음", 5: "매우 많음"},
                default=int(edit_row["freetime"]),
            )
            edit_row["goout"] = select_scale(
                "외출 빈도 (1~5단계, 1: 매우 낮음 / 5: 매우 높음)",
                {1: "매우 낮음", 2: "낮음", 3: "보통", 4: "높음", 5: "매우 높음"},
                default=int(edit_row["goout"]),
            )
            edit_row["internet"] = st.radio(
                "인터넷 사용 가능 여부 (yes/no)",
                ["yes", "no"],
                horizontal=True,
                index=["yes", "no"].index(str(edit_row["internet"])),
            )
            edit_row["higher"] = st.radio(
                "진학 의향 (yes/no)",
                ["yes", "no"],
                horizontal=True,
                index=["yes", "no"].index(str(edit_row["higher"])),
            )

        submitted = st.form_submit_button("예측하기", use_container_width=True)
        if submitted:
            st.session_state.student_row = edit_row
            st.rerun()

input_df = pd.DataFrame([row], columns=x_train.columns)
pred_score = float(regression_model.predict(input_df)[0])
pred_score = max(0.0, min(20.0, pred_score))
pred_group = classification_model.predict(input_df)[0]
group_title, group_detail = group_message(pred_group)
score_delta = pred_score - g3_mean

st.markdown('<div class="app-title">포르투갈 학생 생활패턴 기반 성적 예측</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">Student Performance Dataset 기반 머신러닝 개인 프로젝트</div>',
    unsafe_allow_html=True,
)

st.write("")

st.markdown(
    """
    <div class="panel">
        <h3 style="margin-top:0;">프로젝트 설명</h3>
        <p>
        포르투갈 중등학생의 생활패턴, 학교생활, 결석 정보와 과거 성적을 함께 분석하여
        최종 성적 G3와 학업 수준을 예측하는 머신러닝 개인 프로젝트입니다.
        </p>
        <p class="small-note" style="margin-bottom:0;">
        모델 A는 생활패턴 중심 변수만 사용하고, 모델 B는 G1, G2 과거 성적을 추가하여 성능 차이를 비교합니다.
        이 앱은 성능이 가장 좋았던 Model B 관점으로 예측 결과를 보여줍니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
stat_cols = st.columns(3)
stat_cols[0].metric("전체 학생 수", f"{len(df)}명")
stat_cols[1].metric("평균 G3", f"{g3_mean:.2f}점")
stat_cols[2].metric("최다 그룹", "보통")

st.write("")
button_cols = st.columns([1.2, 1, 2.8])
with button_cols[0]:
    if st.button("예측할 학생 정보 입력", type="primary", use_container_width=True):
        open_student_dialog()

st.write("")
st.subheader("입력값 요약")
summary_items = [
    ("나이", f"{row['age']}세"),
    ("공부시간", f"{row['studytime']}단계"),
    ("낙제 횟수", f"{row['failures']}회"),
    ("결석일수", f"{row['absences']}일"),
    ("건강상태", f"{row['health']}단계"),
    ("가족관계", f"{row['famrel']}단계"),
    ("자유시간", f"{row['freetime']}단계"),
    ("외출 빈도", f"{row['goout']}단계"),
    ("인터넷", row["internet"]),
    ("진학 의향", row["higher"]),
    ("G1", f"{row['G1']}점"),
    ("G2", f"{row['G2']}점"),
]
for start in range(0, len(summary_items), 6):
    cols = st.columns(6)
    for col, (label, value) in zip(cols, summary_items[start : start + 6]):
        with col:
            with st.container(border=True):
                st.caption(label)
                st.markdown(f"**{value}**")
st.caption("성적 변수 G1, G2, G3는 한국식 100점 만점이 아니라 원본 데이터셋의 0~20점 척도입니다.")

reg_result = load_result_table(REGRESSION_RESULT_PATH, default_regression_result())
clf_result = load_result_table(CLASSIFICATION_RESULT_PATH, default_classification_result())

st.write("")
tab_result, tab_model, tab_data = st.tabs(["예측 결과", "모델 비교", "데이터 정보"])

with tab_result:
    st.subheader("예측 결과")
    st.markdown(
        f"""
        <div class="panel result-main">
            <div class="result-label">현재 입력값 기준 예측 결과</div>
            <div class="result-value">최종 성적 G3 {format_score(pred_score)}</div>
            <p style="font-size:1.2rem; font-weight:700; margin:0.6rem 0 0.2rem;">{group_title}</p>
            <p class="small-note">{group_detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("예측 G3", f"{pred_score:.1f}", f"{score_delta:+.1f} vs 평균")
    metric_cols[1].metric("학업 수준", pred_group)
    metric_cols[2].metric("사용 모델", "Model B")

    st.markdown(
        f"""
        <div class="insight-box">
        <strong>한 줄 해석:</strong> 현재 조건에서는 G3가 {pred_score:.1f}점으로 예측되며,
        학업 수준은 <strong>{pred_group}</strong> 그룹입니다. G1, G2 과거 성적을 함께 쓰는 Model B 기준 결과입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("모델 성능 요약")
    if reg_result is not None and clf_result is not None:
        best_reg = reg_result.sort_values(["RMSE", "MAE"], ascending=True).iloc[0]
        best_clf = clf_result.sort_values(["F1", "Accuracy"], ascending=False).iloc[0]
        st.markdown(
            f"""
            <div class="panel">
                <p><strong>회귀 최고 모델</strong>: {best_reg['Model']}</p>
                <p class="small-note">RMSE {best_reg['RMSE']:.3f} / MAE {best_reg['MAE']:.3f} / R2 {best_reg['R2']:.3f}</p>
                <hr>
                <p><strong>분류 최고 모델</strong>: {best_clf['Model']}</p>
                <p class="small-note" style="margin-bottom:0;">
                Accuracy {best_clf['Accuracy']:.3f} / Precision {best_clf['Precision']:.3f} /
                Recall {best_clf['Recall']:.3f} / F1 {best_clf['F1']:.3f}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("모델 성능 결과 파일을 찾을 수 없습니다.")

    st.caption("상세 성능 비교표는 '모델 비교' 탭에서 확인할 수 있습니다.")

with tab_model:
    st.subheader("모델 비교 핵심")
    st.info(
        "모델 A는 생활패턴과 학교생활 정보만 사용하고,모델 B는 여기에 G1, G2 과거 성적을 추가합니다.\n\n"
        "결과적으로 Model B가 더 좋은 성능을 보였으며,이는 G1/G2가 G3와 강하게 연결되어 있기 때문입니다."
    )
    

    if reg_result is not None:
        best_reg = reg_result.sort_values(["RMSE", "MAE"], ascending=True).iloc[0]
        st.markdown(f"**회귀 최고 성능:** {best_reg['Model']} - RMSE {best_reg['RMSE']:.3f}, R2 {best_reg['R2']:.3f}")
        st.dataframe(reg_result, use_container_width=True, hide_index=True)
    else:
        st.warning("회귀 결과 파일을 찾을 수 없습니다.")

    if clf_result is not None:
        best_clf = clf_result.sort_values(["F1", "Accuracy"], ascending=False).iloc[0]
        st.markdown(f"**분류 최고 성능:** {best_clf['Model']} - Accuracy {best_clf['Accuracy']:.3f}, F1 {best_clf['F1']:.3f}")
        st.dataframe(clf_result, use_container_width=True, hide_index=True)
    else:
        st.warning("분류 결과 파일을 찾을 수 없습니다.")

with tab_data:
    st.subheader("데이터셋 요약")
    data_cols = st.columns(3)
    data_cols[0].metric("데이터 크기", f"{df.shape[0]}행")
    data_cols[1].metric("컬럼 수", f"{df.shape[1] - 1}개")
    data_cols[2].metric("결측치", f"{int(df.isnull().sum().sum())}개")

    group_df = grade_counts.rename_axis("학업 수준").reset_index(name="학생 수")
    st.write("학업 수준 분포")
    st.dataframe(group_df, use_container_width=True, hide_index=True)

    st.write("주요 변수 설명")
    st.dataframe(
        pd.DataFrame(
            [
                ["G1", "1차 성적, 0~20점"],
                ["G2", "2차 성적, 0~20점"],
                ["G3", "최종 성적, 0~20점"],
                ["studytime", "주간 공부시간"],
                ["absences", "결석일수"],
                ["health", "건강상태"],
                ["goout", "외출 빈도"],
            ],
            columns=["변수", "설명"],
        ),
        use_container_width=True,
        hide_index=True,
    )
