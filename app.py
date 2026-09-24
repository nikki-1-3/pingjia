import streamlit as st
st.title("测试")
st.write("Streamlit 版本：", st.__version__)

try:
    import jieba
    st.success("jieba OK")
except Exception as e:
    st.error(f"jieba 报错：{e}")

try:
    import matplotlib.pyplot as plt
    st.success("matplotlib OK")
except Exception as e:
    st.error(f"matplotlib 报错：{e}")

try:
    @st.dialog("测试弹窗")
    def _d():
        st.write("hi")
    st.success("@st.dialog OK")
except Exception as e:
    st.error(f"@st.dialog 报错：{e}")

try:
    df = __import__("pandas").read_csv("大众点评评论数据.csv", encoding="gb18030", nrows=100)
    st.success(f"CSV OK，读到 {len(df)} 行")
    st.write(df.head())
except Exception as e:
    st.error(f"CSV 报错：{e}")
