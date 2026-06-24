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
        padding-top: 2.5rem;
        padding-bottom: 1.0rem;
        max-width: 1280px;
    }
    .app-title {
        font-size: 1.72rem;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.1rem;
    }
    .muted {
        color: #64748b;
        font-size: 0.82rem;
    }
    .panel {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .result-main {
        border-left: 5px solid #7FFFD4;
        background: #fff7ed;
    }
    .result-label {
        color: #64748b;
        font-size: 0.8rem;
        margin-bottom: 0.25rem;
    }
    .result-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.15;
    }
    .small-note {
        font-size: 0.82rem;
        color: #475569;
    }
    .insight-box {
        border-left: 4px solid #2563eb;
        background: #eff6ff;
        padding: 0.65rem 0.85rem;
        border-radius: 6px;
        margin-top: 0.75rem;
        color: #1e3a8a;
    }
    div[data-testid="stButton"] button {
        white-space: nowrap;
        min-width: 3.4rem;
        min-height: 2.0rem;
        padding: 0.25rem 0.55rem;
        font-size: 0.82rem;
        border-radius: 8px;
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


def apply_student_value(field: str, value):
    st.session_state.student_row[field] = value
    st.session_state[f"draft_{field}"] = value
    st.session_state.pop("edit_config", None)
    st.rerun()


def get_draft_value(field: str):
    draft_key = f"draft_{field}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = st.session_state.student_row[field]
    return st.session_state[draft_key]


def step_draft_value(field: str, delta: int, min_value: int, max_value: int):
    draft_key = f"draft_{field}"
    current_value = int(get_draft_value(field))
    st.session_state[draft_key] = max(min_value, min(max_value, current_value + delta))
    st.rerun()


def set_draft_value(field: str, value):
    st.session_state[f"draft_{field}"] = value
    st.rerun()


def open_edit_dialog(config: dict):
    field = config["field"]
    st.session_state.edit_config = config
    st.session_state[f"draft_{field}"] = st.session_state.student_row[field]
    st.rerun()


def close_edit_dialog():
    st.session_state.pop("edit_config", None)


@st.dialog("입력값 수정", width="small", on_dismiss=close_edit_dialog)
def edit_value_dialog():
    config = st.session_state.get("edit_config")
    if not config:
        return

    field = config["field"]
    kind = config["kind"]
    unit = config.get("unit", "")

    st.caption(config["range_text"])

    if kind == "number":
        min_value = int(config["min_value"])
        max_value = int(config["max_value"])
        draft_value = int(get_draft_value(field))
        minus_col, value_col, plus_col = st.columns([1, 1.4, 1])
        with minus_col:
            if st.button("-", key=f"minus_{field}", use_container_width=True, disabled=draft_value <= min_value):
                step_draft_value(field, -1, min_value, max_value)
        with value_col:
            st.markdown(
                f"<div style='text-align:center; font-size:1.65rem; font-weight:800; line-height:2.2rem;'>{draft_value}{unit}</div>",
                unsafe_allow_html=True,
            )
        with plus_col:
            if st.button("+", key=f"plus_{field}", use_container_width=True, disabled=draft_value >= max_value):
                step_draft_value(field, 1, min_value, max_value)
        if st.button("반영", key=f"apply_{field}", type="primary", use_container_width=True):
            apply_student_value(field, draft_value)

    elif kind == "select":
        options = config["options"]
        option_keys = list(options.keys())
        draft_value = int(get_draft_value(field))
        draft_index = option_keys.index(draft_value)
        prev_col, value_col, next_col = st.columns([1, 1.7, 1])
        with prev_col:
            if st.button("-", key=f"prev_{field}", use_container_width=True, disabled=draft_index <= 0):
                set_draft_value(field, option_keys[draft_index - 1])
        with value_col:
            st.markdown(
                f"<div style='text-align:center; font-size:1.18rem; font-weight:800; line-height:1.4rem;'>{draft_value}단계</div>"
                f"<div style='text-align:center; color:#64748b; font-size:0.78rem;'>{options[draft_value]}</div>",
                unsafe_allow_html=True,
            )
        with next_col:
            if st.button("+", key=f"next_{field}", use_container_width=True, disabled=draft_index >= len(option_keys) - 1):
                set_draft_value(field, option_keys[draft_index + 1])
        if st.button("반영", key=f"apply_{field}", type="primary", use_container_width=True):
            apply_student_value(field, draft_value)

    elif kind == "yes_no":
        draft_value = str(get_draft_value(field))
        yes_col, no_col = st.columns(2)
        with yes_col:
            if st.button("yes", key=f"yes_{field}", use_container_width=True, disabled=draft_value == "yes"):
                set_draft_value(field, "yes")
        with no_col:
            if st.button("no", key=f"no_{field}", use_container_width=True, disabled=draft_value == "no"):
                set_draft_value(field, "no")
        st.markdown(f"**현재 선택값: {draft_value}**")
        if st.button("반영", key=f"apply_{field}", type="primary", use_container_width=True):
            apply_student_value(field, draft_value)


def edit_number_card(label: str, value: str, field: str, title: str, min_value: int, max_value: int, unit: str = ""):
    with st.container(border=True):
        title_col, edit_col = st.columns([0.85, 1.15], gap="small")
        with title_col:
            st.caption(label)
        with edit_col:
            if st.button("수정", key=f"edit_{field}", use_container_width=True):
                open_edit_dialog(
                    {
                        "kind": "number",
                        "field": field,
                        "range_text": f"{title} ({min_value}~{max_value}{unit})",
                        "min_value": min_value,
                        "max_value": max_value,
                        "unit": unit,
                    }
                )
        st.markdown(f"**{value}**")


def edit_select_card(label: str, value: str, field: str, title: str, options: dict[int, str]):
    with st.container(border=True):
        title_col, edit_col = st.columns([0.85, 1.15], gap="small")
        with title_col:
            st.caption(label)
        with edit_col:
            if st.button("수정", key=f"edit_{field}", use_container_width=True):
                open_edit_dialog(
                    {
                        "kind": "select",
                        "field": field,
                        "range_text": title,
                        "options": options,
                    }
                )
        st.markdown(f"**{value}**")


def edit_yes_no_card(label: str, value: str, field: str, title: str):
    with st.container(border=True):
        title_col, edit_col = st.columns([0.85, 1.15], gap="small")
        with title_col:
            st.caption(label)
        with edit_col:
            if st.button("수정", key=f"edit_{field}", use_container_width=True):
                open_edit_dialog(
                    {
                        "kind": "yes_no",
                        "field": field,
                        "range_text": title,
                    }
                )
        st.markdown(f"**{value}**")


if st.session_state.get("edit_config"):
    edit_value_dialog()


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

studytime_options = {
    1: "2시간 미만",
    2: "2~5시간",
    3: "5~10시간",
    4: "10시간 이상",
}
health_options = {1: "매우 나쁨", 2: "나쁨", 3: "보통", 4: "좋음", 5: "매우 좋음"}
famrel_options = {1: "매우 나쁨", 2: "나쁨", 3: "보통", 4: "좋음", 5: "매우 좋음"}
freetime_options = {1: "매우 적음", 2: "적음", 3: "보통", 4: "많음", 5: "매우 많음"}
goout_options = {1: "매우 낮음", 2: "낮음", 3: "보통", 4: "높음", 5: "매우 높음"}

reg_result = load_result_table(REGRESSION_RESULT_PATH, default_regression_result())
clf_result = load_result_table(CLASSIFICATION_RESULT_PATH, default_classification_result())
best_reg = reg_result.sort_values(["RMSE", "MAE"], ascending=True).iloc[0]
best_clf = clf_result.sort_values(["F1", "Accuracy"], ascending=False).iloc[0]

left_col, right_col = st.columns([0.9, 1.45], gap="large")

with left_col:
    st.markdown(
        """
        <div class="panel">
            <strong>프로젝트 설명</strong>
            <p class="small-note" style="margin:0.45rem 0 0;">
            포르투갈 중등학생의 생활패턴, 결석 정보와 과거 성적을 활용해 최종 성적 G3와 학업 수준을 예측합니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(3)
    stat_cols[0].metric("학생 수", f"{len(df)}명")
    stat_cols[1].metric("평균 G3", f"{g3_mean:.2f}점")
    stat_cols[2].metric("최다 그룹", "보통")

    st.subheader("입력값 요약")
    row1 = st.columns(3)
    with row1[0]:
        edit_number_card("나이", f"{row['age']}세", "age", "나이", age_min, age_max, "세")
    with row1[1]:
        edit_select_card("공부시간", f"{row['studytime']}단계", "studytime", "주간 공부시간 (1~4단계)", studytime_options)
    with row1[2]:
        edit_number_card("낙제", f"{row['failures']}회", "failures", "과거 낙제 횟수", failures_min, failures_max, "회")

    row2 = st.columns(3)
    with row2[0]:
        edit_number_card("결석", f"{row['absences']}일", "absences", "결석일수", absences_min, absences_max, "일")
    with row2[1]:
        edit_select_card("건강", f"{row['health']}단계", "health", "건강상태 (1~5단계)", health_options)
    with row2[2]:
        edit_select_card("가족", f"{row['famrel']}단계", "famrel", "가족관계 (1~5단계)", famrel_options)

    row3 = st.columns(3)
    with row3[0]:
        edit_select_card("자유", f"{row['freetime']}단계", "freetime", "자유시간 (1~5단계)", freetime_options)
    with row3[1]:
        edit_select_card("외출", f"{row['goout']}단계", "goout", "외출 빈도 (1~5단계)", goout_options)
    with row3[2]:
        edit_yes_no_card("인터넷", row["internet"], "internet", "인터넷 사용 가능 여부")

    row4 = st.columns(3)
    with row4[0]:
        edit_yes_no_card("진학", row["higher"], "higher", "진학 의향")
    with row4[1]:
        edit_number_card("G1", f"{row['G1']}점", "G1", "G1 1차 성적", 0, 20, "점")
    with row4[2]:
        edit_number_card("G2", f"{row['G2']}점", "G2", "G2 2차 성적", 0, 20, "점")
    st.caption("G1, G2, G3는 원본 데이터셋 기준 0~20점 척도입니다.")

with right_col:
    tab_result, tab_model, tab_data = st.tabs(["예측 결과", "모델 비교", "데이터 정보"])

    with tab_result:
        st.markdown(
            f"""
            <div class="panel result-main">
                <div class="result-label">현재 입력값 기준 예측 결과</div>
                <div class="result-value">최종 성적 G3 {format_score(pred_score)}</div>
                <p style="font-size:1.05rem; font-weight:700; margin:0.45rem 0 0.15rem;">{group_title}</p>
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
            학업 수준은 <strong>{pred_group}</strong> 그룹입니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("모델 성능 요약")
        perf_cols = st.columns(2)
        with perf_cols[0]:
            with st.container(border=True):
                st.caption("회귀 최고 모델")
                st.markdown(f"**{best_reg['Model']}**")
                st.caption(f"RMSE {best_reg['RMSE']:.3f} / R2 {best_reg['R2']:.3f}")
        with perf_cols[1]:
            with st.container(border=True):
                st.caption("분류 최고 모델")
                st.markdown(f"**{best_clf['Model']}**")
                st.caption(f"Accuracy {best_clf['Accuracy']:.3f} / F1 {best_clf['F1']:.3f}")

    with tab_model:
        st.info(
            "모델 A는 생활패턴과 학교생활 정보만 사용하고, 모델 B는 여기에 G1, G2 과거 성적을 추가합니다. "
            "Model B가 더 좋은 성능을 보였으며, G1/G2가 G3와 강하게 연결되어 있음을 확인했습니다."
        )

        st.markdown(f"**회귀 최고 성능:** {best_reg['Model']} - RMSE {best_reg['RMSE']:.3f}, R2 {best_reg['R2']:.3f}")
        st.dataframe(reg_result, use_container_width=True, hide_index=True)

        st.markdown(f"**분류 최고 성능:** {best_clf['Model']} - Accuracy {best_clf['Accuracy']:.3f}, F1 {best_clf['F1']:.3f}")
        st.dataframe(clf_result, use_container_width=True, hide_index=True)

    with tab_data:
        data_cols = st.columns(3)
        data_cols[0].metric("데이터", f"{df.shape[0]}행")
        data_cols[1].metric("컬럼", f"{df.shape[1] - 1}개")
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
