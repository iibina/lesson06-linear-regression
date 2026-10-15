
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    필요한열 = ["날짜", "평균기온"]
    for 열 in 필요한열:
        if 열 not in df.columns:
            raise ValueError(f"필요한 열이 없습니다: {열}")

    # 날짜 형식 자동 처리
    날짜문자열 = (
        df["날짜"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["날짜"] = pd.to_datetime(
        날짜문자열,
        format="%Y%m%d",
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 날짜와 평균기온이 유효한 행만 사용
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 2025년까지의 데이터만 사용
    df = df[df["날짜"].dt.year <= 2025].copy()
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온 및 관측 날짜 수
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일이 300일 이상인 연도만 유지
    yearly = yearly[yearly["관측일수"] >= 300].copy()

    return yearly.sort_values("연도").reset_index(drop=True)


yearly = load_data()

# 원인 확인용 출력
st.write("조건 적용 후 남은 연도 수:", len(yearly))
st.dataframe(yearly, use_container_width=True)

if len(yearly) < 2:
    st.error(
        "유효한 연도가 2개 미만입니다. "
        "날짜 변환 결과와 데이터 주소를 확인해 주세요."
    )
    st.stop()
