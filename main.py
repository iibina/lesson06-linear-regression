
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온 데이터를 이용해 회귀 직선을 만들고, "
    "선택한 연도의 예상 평균기온을 확인합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
기준연도 = 2025
최소관측일수 = 300


# -----------------------------
# 1. 데이터 불러오기 및 전처리
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    필요한열 = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    누락열 = [열 for 열 in 필요한열 if 열 not in df.columns]

    if 누락열:
        raise ValueError(
            f"데이터에 필요한 열이 없습니다: {', '.join(누락열)}"
        )

    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        format="%Y%m%d",
        errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 기준 기간 이후의 자료 제외
    df = df[df["연도"] <= 기준연도]

    # 연도별 평균기온과 실제 관측일수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 해 제외
    yearly = yearly[
        yearly["관측일수"] >= 최소관측일수
    ].copy()

    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


try:
    with st.spinner("서울 기온 데이터를 불러오는 중입니다..."):
        yearly = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오거나 처리하지 못했습니다.")
    st.code(str(e))
    st.info(
        "인터넷 연결과 데이터 주소를 확인해 주세요. "
        "CSV의 열 이름과 날짜 형식도 확인해 주세요."
    )
    st.stop()


if len(yearly) < 2:
    st.error(
        "회귀 직선을 만들기 위한 유효한 연도가 부족합니다. "
        "데이터와 관측일 조건을 확인해 주세요."
    )
    st.stop()


# -----------------------------
# 2. 선형회귀 및 상관계수 계산
# -----------------------------
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

기울기, 절편 = np.polyfit(x, y, 1)
상관계수 = np.corrcoef(x, y)[0, 1]

시작연도 = int(yearly["연도"].min())
끝연도 = int(yearly["연도"].max())
사용연도수 = len(yearly)


def predict_temperature(year):
    """회귀 직선으로 예상 연평균기온을 계산합니다."""
    return 기울기 * year + 절편


# -----------------------------
# 3. 분석 결과 요약
# -----------------------------
st.subheader("📊 회귀 분석 결과")

c1, c2, c3, c4 = st.columns(4)

c1.metric("회귀에 사용한 연도 수", f"{사용연도수}년")
c2.metric("시작 연도", f"{시작연도}년")
c3.metric("끝 연도", f"{끝연도}년")
c4.metric("상관계수", f"{상관계수:.3f}")

st.caption(
    f"분석 기준: {기준연도}년까지의 자료 중 "
    f"연간 관측일이 {최소관측일수}일 이상인 연도"
)


# -----------------------------
# 4. 산점도와 회귀 직선
# -----------------------------
st.subheader("서울 연평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측값 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선은 분석 기간에 대해서만 표시
line_x = np.linspace(시작연도, 끝연도, 200)
line_y = predict_temperature(line_x)

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="선형회귀 직선",
        line=dict(width=3, color="red"),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"**회귀식:** 예상 연평균기온 = "
    f"{기울기:.5f} × 연도 + {절편:.3f}"
)

if 상관계수 > 0:
    st.write("상관계수는 양수입니다. 연도가 커질수록 기온이 높아지는 경향이 나타납니다.")
elif 상관계수 < 0:
    st.write("상관계수는 음수입니다. 연도가 커질수록 기온이 낮아지는 경향이 나타납니다.")
else:
    st.write("상관계수는 0입니다. 두 변수 사이의 선형 상관관계가 거의 없습니다.")

st.caption(
    "상관계수는 -1부터 1까지의 값입니다. "
    "절댓값이 1에 가까울수록 선형 관계가 강하고, "
    "0에 가까울수록 선형 관계가 약합니다."
)


# -----------------------------
# 5. 연도 슬라이더와 예상 기온
# -----------------------------
st.divider()
st.subheader("🌡️ 원하는 연도의 예상 기온 확인")

선택연도 = st.slider(
    "예상 기온을 확인할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

예상기온 = predict_temperature(선택연도)

st.metric(
    label=f"{선택연도}년 예상 연평균기온",
    value=f"{예상기온:.2f} ℃"
)

if 선택연도 < 시작연도 or 선택연도 > 끝연도:
    st.warning(
        f"{선택연도}년은 회귀에 사용한 기간 "
        f"{시작연도}~{끝연도}년의 바깥입니다. "
        "이 값은 회귀 직선을 연장해 계산한 추정치이므로 "
        "불확실성이 더 클 수 있습니다."
    )
else:
    st.info(
        "선택한 연도는 회귀 분석에 사용한 기간 안에 있습니다. "
        "표시된 값은 해당 연도의 실제 관측값이 아니라 "
        "회귀 직선으로 계산한 추정치입니다."
    )


# -----------------------------
# 6. 분석에 사용된 연도별 데이터
# -----------------------------
with st.expander("분석에 사용된 연도별 데이터 보기"):
    보여줄표 = yearly.copy()
    보여줄표["연평균기온"] = 보여줄표["연평균기온"].round(2)
    st.dataframe(보여줄표, use_container_width=True, hide_index=True)
