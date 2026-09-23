import re
import os
import urllib.request
import jieba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import streamlit as st

# ========== 页面配置 ==========
st.set_page_config(page_title="大众点评评论优缺点总结", page_icon="🍽️", layout="wide")

# 暖橙色主题 CSS
st.markdown("""
<style>
    .stApp { background-color: #FFF9F5; }
    h1, h2, h3 { color: #E85D2F !important; }
    .hero {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%);
        padding: 40px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 { color: white !important; margin: 0 0 8px 0; font-size: 2.2rem; }
    .hero p { margin: 0; opacity: 0.95; font-size: 1.05rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(255,107,53,0.08);
        border-left: 4px solid #FF6B35;
        margin-bottom: 12px;
    }
    .metric-card .label { color: #999; font-size: 0.85rem; }
    .metric-card .value { color: #E85D2F; font-size: 1.6rem; font-weight: bold; }
    .aspect-card {
        background: white;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
        border-left: 3px solid #4CAF50;
    }
    .aspect-card.neg { border-left-color: #F44336; }
    .aspect-card .title { font-weight: bold; color: #333; }
    .aspect-card .desc { color: #666; font-size: 0.9rem; margin-top: 4px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 中文字体自动下载
FONT_PATH = "SimHei.ttf"
if not os.path.exists(FONT_PATH):
    try:
        urllib.request.urlretrieve(
            "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf",
            FONT_PATH
        )
    except Exception:
        pass
if os.path.exists(FONT_PATH):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# ========== Hero 区 ==========
st.markdown("""
<div class="hero">
    <h1>🍽️ 大众点评评论优缺点总结系统</h1>
    <p>基于情感分类与属性抽取的餐饮评论分析工具 · 一键看懂好评与差评</p>
</div>
""", unsafe_allow_html=True)

# ========== 1. 加载数据 ==========
@st.cache_data
def load_data(path):
    return pd.read_csv(path, encoding='gb18030', low_memory=False)

FILE_PATH = '大众点评评论数据.csv'
try:
    df_raw = load_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"找不到数据文件：{FILE_PATH}。请确认文件与 app.py 在同一目录下。")
    st.stop()

# ========== 2. 提取列 ==========
df = df_raw[['Content_review', 'Rating']].rename(columns={
    'Content_review': 'review', 'Rating': 'label'
})
df = df.dropna(subset=['review', 'label'])
df['review'] = df['review'].astype(str)
df = df[df['label'] != 3]
df['label'] = df['label'].apply(lambda x: 1 if x >= 4 else 0)

# 侧边栏
with st.sidebar:
    st.markdown("### ⚙️ 分析控制台")
    sample_size = st.slider("采样数量", min_value=1000, max_value=20000, value=5000, step=1000)
    st.caption("样本越大，分析越准，但加载时间更长。")

if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

# 数据概览卡片
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><div class="label">分析样本</div><div class="value">{len(df)}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="label">好评</div><div class="value">{len(df[df["label"]==1])}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="label">差评</div><div class="value">{len(df[df["label"]==0])}</div></div>', unsafe_allow_html=True)

# ========== 3. 预处理 ==========
stopwords = set(['的','了','还','很','也','就','都','和','与','在','是','有','一','个','这','那','不','我','你','他','她','它','们','但','而','且','或','被','把','给','让','从','到','对','为','以','于','之','其','此','该','等','着','过','吗','呢','吧','啊','呀','哦','嗯','这个','那个','什么','怎么','可以','没有','不是'])

def preprocess(text):
    text = re.sub(r'[^\u4e00-\u9fa5]', '', str(text))
    return ' '.join([w for w in jieba.cut(text) if w not in stopwords and len(w) > 1])

with st.spinner("正在分词预处理..."):
    df['clean'] = df['review'].apply(preprocess)

# ========== 4. 情感分类 ==========
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)

# ========== 5. 优缺点抽取 ==========
aspect_dict = {
    '口味': ['口味','味道','好吃','难吃','香','辣','咸','淡','甜','鲜'],
    '环境': ['环境','装修','氛围','干净','卫生','嘈杂','安静','宽敞'],
    '服务': ['服务','态度','服务员','热情','周到','冷漠','耐心'],
    '价格': ['价格','性价比','贵','便宜','划算','实惠','收费'],
    '上菜速度': ['上菜','速度','等','慢','快','排队','催'],
    '分量': ['分量','份量','量','少','足','多','精致'],
}
pos_words = {'多','足','大','好','不错','干净','方便','热情','周到','舒适','满意','棒','优秀','喜欢','漂亮','好吃','香','实惠','快','新鲜','美味','赞'}
neg_words = {'少','小','差','脏','吵','旧','破','坏','失望','糟糕','难','远','贵','慢','难吃','咸','淡','冷','腻','一般','坑'}

def extract_aspects(text):
    results = []
    words = list(jieba.cut(text))
    for aspect, keywords in aspect_dict.items():
        for i, w in enumerate(words):
            if w in keywords:
                for cw in words[i+1:i+4]:
                    if cw in pos_words:
                        results.append((aspect, '正面', cw)); break
                    elif cw in neg_words:
                        results.append((aspect, '负面', cw)); break
                else:
                    for cw in words[max(0, i-2):i]:
                        if cw in pos_words:
                            results.append((aspect, '正面', cw)); break
                        elif cw in neg_words:
                            results.append((aspect, '负面', cw)); break
    return results

with st.spinner("正在抽取优缺点..."):
    pos_aspects, neg_aspects = [], []
    for r in df[df['label']==1]['review']:
        pos_aspects.extend(extract_aspects(r))
    for r in df[df['label']==0]['review']:
        neg_aspects.extend(extract_aspects(r))

pos_detail = defaultdict(Counter)
for a, d, w in pos_aspects:
    if d == '正面': pos_detail[a][w] += 1
neg_detail = defaultdict(Counter)
for a, d, w in neg_aspects:
    if d == '负面': neg_detail[a][w] += 1

# ========== 6. 三个标签页 ==========
tab1, tab2, tab3 = st.tabs(["📊 情感分类", "👍 优缺点总结", "📈 可视化分析"])

with tab1:
    st.subheader("情感分类模型表现")
    st.caption("基于 TF-IDF + 朴素贝叶斯，对好评/差评进行自动分类。")
    st.code(classification_report(y_test, y_pred, target_names=['差评','好评'], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig1, ax1 = plt.subplots(figsize=(3.2, 2.6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['差评','好评'], yticklabels=['差评','好评'], ax=ax1)
    ax1.set_title('混淆矩阵', fontsize=10)
    ax1.set_ylabel('真实标签'); ax1.set_xlabel('预测标签')
    plt.tight_layout()
    st.pyplot(fig1)

with tab2:
    st.subheader("用户最常提到的优点与缺点")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ 优点")
        for aspect, wc in sorted(pos_detail.items(), key=lambda x: -sum(x[1].values())):
            total = sum(wc.values())
            if total == 0: continue
            words = '、'.join([f"{aspect}{w}" for w, _ in wc.most_common(3)])
            st.markdown(f'<div class="aspect-card"><div class="title">{aspect} · 提及 {total} 次</div><div class="desc">{words}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### ❌ 缺点")
        for aspect, wc in sorted(neg_detail.items(), key=lambda x: -sum(x[1].values())):
            total = sum(wc.values())
            if total == 0: continue
            words = '、'.join([f"{aspect}{w}" for w, _ in wc.most_common(3)])
            st.markdown(f'<div class="aspect-card neg"><div class="title">{aspect} · 提及 {total} 次</div><div class="desc">{words}</div></div>', unsafe_allow_html=True)

with tab3:
    st.subheader("各属性在好评 / 差评中的提及对比")
    all_aspects = sorted(set(list(pos_detail.keys()) + list(neg_detail.keys())))
    if all_aspects:
        pos_vals = [sum(pos_detail.get(a, Counter()).values()) for a in all_aspects]
        neg_vals = [sum(neg_detail.get(a, Counter()).values()) for a in all_aspects]

        x = np.arange(len(all_aspects)); width = 0.35
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.bar(x - width/2, pos_vals, width, label='好评', color='#FF6B35')
        ax2.bar(x + width/2, neg_vals, width, label='差评', color='#FFB088')
        ax2.set_xticks(x); ax2.set_xticklabels(all_aspects, rotation=15)
        ax2.set_ylabel('提及次数'); ax2.set_title('属性提及对比')
        ax2.legend(); plt.tight_layout()
        st.pyplot(fig2)

        angles = np.linspace(0, 2*np.pi, len(all_aspects), endpoint=False).tolist()
        pos_r = pos_vals + [pos_vals[0]]; neg_r = neg_vals + [neg_vals[0]]
        angles_r = angles + [angles[0]]
        fig3, ax3 = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
        ax3.plot(angles_r, pos_r, 'o-', linewidth=2, label='好评', color='#FF6B35')
        ax3.fill(angles_r, pos_r, alpha=0.25, color='#FF6B35')
        ax3.plot(angles_r, neg_r, 'o-', linewidth=2, label='差评', color='#FFB088')
        ax3.fill(angles_r, neg_r, alpha=0.25, color='#FFB088')
        ax3.set_xticks(angles); ax3.set_xticklabels(all_aspects)
        ax3.set_title('属性情感雷达图'); ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        st.pyplot(fig3)
    else:
        st.warning("未抽取到任何属性，可能是属性词典与数据不匹配。")

st.markdown("---")
st.caption("🍽️ 大众点评评论优缺点总结系统 · 基于情感分类与属性抽取")
