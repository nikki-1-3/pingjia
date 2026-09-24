import streamlit as st

st.title("测试页面")
st.write("Streamlit 版本：", st.__version__)
st.success("如果看到这行，说明部署正常")

try:
    import jieba
    st.success("jieba 已安装")
except Exception as e:
    st.error(f"jieba 导入失败：{e}")

try:
    import matplotlib.pyplot as plt
    st.success("matplotlib 已安装")
except Exception as e:
    st.error(f"matplotlib 导入失败：{e}")

try:
    @st.dialog("测试")
    def _d():
        st.write("hi")
    st.success("@st.dialog 可用")
except Exception as e:
    st.error(f"@st.dialog 失败：{e}")

try:
    import pandas as pd
    df = pd.read_csv("大众点评评论数据.csv", encoding="gb18030", nrows=100)
    st.success(f"CSV 读取正常，{len(df)} 行")
except Exception as e:
    st.error(f"CSV 读取失败：{e}")
