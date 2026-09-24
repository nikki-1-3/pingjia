import re
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

# 暖橙色主题
st.markdown("""
<style>
    .stApp { background-color: #FFF9F5; }
    h1, h2, h3 { color: #E85D2F !important; }
    section[data-testid="stSidebar"] { background-color: #FFF3EC; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFE8DC;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        color: #E85D2F;
    }
    .stTabs [aria-selected="true"] { background-color: #FF6B35; color: white; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🍽️ 大众点评评论优缺点总结系统")
st.markdown("基于情感分类与属性抽取的餐饮评论分析工具")

# 中文字体（Streamlit Cloud 上 SimHei 可能不存在，做兜底）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8

# ========== 1. 加载数据集（带缓存） ==========
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, encoding='gb18030', low_memory=False)
    return df

FILE_PATH = '大众点评评论数据.csv'

try:
    df_raw = load_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"找不到数据文件：{FILE_PATH}。请确认文件与 app.py 在同一目录下。")
    st.stop()

st.success(f"数据加载成功，共 {len(df_raw)} 条评论")

# ========== 2. 提取评论列和标签列 ==========
df = df_raw[['Content_review', 'Rating']].rename(columns={
    'Content_review': 'review',
    'Rating': 'label'
})
df = df.dropna(subset=['review', 'label'])
df['review'] = df['review'].astype(str)

df = df[df['label'] != 3]
df['label'] = df['label'].apply(lambda x: 1 if x >= 4 else 0)

sample_size = st.sidebar.slider("采样数量", min_value=1000, max_value=20000, value=5000, step=1000)
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

st.write(f"当前使用数据：**{len(df)}** 条")
st.write(f"好评：**{len(df[df['label']==1])}** 条 ｜ 差评：**{len(df[df['label']==0])}** 条")

# ========== 3. 文本预处理 ==========
stopwords = set(['的','了','还','很','也','就','都','和','与','在','是','有','一','个','这','那','不','我','你','他','她','它','们','但','而','且','或','被','把','给','让','从','到','对','为','以','于','之','其','此','该','等','着','过','吗','呢','吧','啊','呀','哦','嗯','这个','那个','什么','怎么','可以','没有','不是'])

def preprocess(text):
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    words = [w for w in jieba.cut(text) if w not in stopwords and len(w) > 1]
    return ' '.join(words)

with st.spinner("正在分词预处理..."):
    df['clean'] = df['review'].apply(preprocess)

with st.expander("查看预处理示例"):
    for i in range(min(3, len(df))):
        st.write(f"**原文**：{df['review'].iloc[i][:80]}...")
        st.write(f"**清洗**：{df['clean'].iloc[i][:80]}...")
        st.write("---")

# ========== 4. 情感分类 ==========
st.header("一、情感分类")

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)

report = classification_report(y_test, y_pred, target_names=['差评','好评'], zero_division=0)

# 文字和图片同行
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ========== 第一块：只讲判断结果（好评/差评各判对多少） ==========
st.subheader("📋 判断结果")

real_pos = int((y_test == 1).sum())
real_neg = int((y_test == 0).sum())
correct_pos = int(((y_test == 1) & (y_pred == 1)).sum())
correct_neg = int(((y_test == 0) & (y_pred == 0)).sum())

# 用两行文字分别说清楚，不堆术语
st.markdown(
    f"一共测了 **{len(y_test)}** 条评论："
    f"真实好评 **{real_pos}** 条，真实差评 **{real_neg}** 条。"
)
st.markdown(
    f"- 好评里，系统判对 **{correct_pos}** 条，判错 **{real_pos - correct_pos}** 条\n"
    f"- 差评里，系统判对 **{correct_neg}** 条，判错 **{real_neg - correct_neg}** 条"
)

# 图：好评、差评各自判对/判错
labels = ['好评', '差评']
correct_counts = [correct_pos, correct_neg]
wrong_counts = [real_pos - correct_pos, real_neg - correct_neg]

x = np.arange(len(labels))
w = 0.38
fig1, ax1 = plt.subplots(figsize=(6, 3.4))
b1 = ax1.bar(x - w/2, correct_counts, w, label='判对', color='#FF6B35')
b2 = ax1.bar(x + w/2, wrong_counts, w, label='判错', color='#FFD9C2')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.set_ylabel('评论条数', fontsize=9)
ax1.set_title('好评 / 差评 各判对多少', fontsize=11)
ax1.legend(fontsize=9)
for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        ax1.annotate(f'{int(h)}', (b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4 if h > 0 else -12),
                     textcoords='offset points',
                     ha='center', fontsize=8,
                     color='#999' if h == 0 else 'black')
ax1.set_ylim(0, max(real_pos, real_neg) * 1.2)
plt.tight_layout()
st.pyplot(fig1)

st.markdown("---")

# ========== 第二块：只讲准确率 ==========
st.subheader("🎯 准确率")

acc = accuracy_score(y_test, y_pred)

st.markdown(
    f"系统整体判断对了 **{acc:.1%}** 的评论。"
)
st.progress(min(acc, 1.0), text=f"{acc:.1%}")

if acc >= 0.9:
    st.success("😄 准确率很高，结果可以放心参考。")
elif acc >= 0.8:
    st.info("🙂 准确率不错，结果基本可靠。")
else:
    st.warning("😐 准确率一般，建议结合原文一起看。")

with st.expander("想看更细的指标（可选）"):
    st.code(report)

# ========== 5. 优缺点抽取 ==========
st.header("二、优缺点抽取")

aspect_dict = {
    '口味': ['口味','味道','好吃','难吃','香','辣','咸','淡','甜','鲜'],
    '环境': ['环境','装修','氛围','干净','卫生','嘈杂','安静','宽敞'],
    '服务': ['服务','态度','服务员','热情','周到','冷漠','耐心'],
    '价格': ['价格','性价比','贵','便宜','划算','实惠','收费'],
    '上菜速度': ['上菜','速度','等','慢','快','排队','催'],
    '分量': ['分量','份量','量','少','足','多','精致'],
}

pos_words = {
    '多': '多', '足': '足', '大': '大', '好': '好', '不错': '不错',
    '干净': '干净', '方便': '方便', '热情': '热情', '周到': '周到',
    '舒适': '舒适', '满意': '满意', '棒': '棒', '优秀': '优秀',
    '喜欢': '喜欢', '漂亮': '漂亮', '好吃': '好吃', '香': '香',
    '实惠': '实惠', '快': '快', '新鲜': '新鲜', '美味': '美味', '赞': '赞'
}

neg_words = {
    '少': '少', '小': '小', '差': '差', '脏': '脏', '吵': '吵',
    '旧': '旧', '破': '破', '坏': '坏', '失望': '失望', '糟糕': '糟糕',
    '难': '难', '远': '远', '贵': '贵', '慢': '慢', '难吃': '难吃',
    '咸': '咸', '淡': '淡', '冷': '冷', '腻': '腻', '一般': '一般', '坑': '坑'
}

def extract_aspects(text):
    results = []
    words = list(jieba.cut(text))
    for aspect, keywords in aspect_dict.items():
        for i, w in enumerate(words):
            if w in keywords:
                context = words[i+1:i+4]
                found = False
                for cw in context:
                    if cw in pos_words:
                        results.append((aspect, '正面', pos_words[cw]))
                        found = True
                        break
                    elif cw in neg_words:
                        results.append((aspect, '负面', neg_words[cw]))
                        found = True
                        break
                if not found:
                    context = words[max(0, i-2):i]
                    for cw in context:
                        if cw in pos_words:
                            results.append((aspect, '正面', pos_words[cw]))
                            break
                        elif cw in neg_words:
                            results.append((aspect, '负面', neg_words[cw]))
                            break
    return results

with st.spinner("正在抽取优缺点..."):
    pos_reviews = df[df['label']==1]['review']
    neg_reviews = df[df['label']==0]['review']

    pos_aspects = []
    for r in pos_reviews:
        pos_aspects.extend(extract_aspects(r))

    neg_aspects = []
    for r in neg_reviews:
        neg_aspects.extend(extract_aspects(r))

pos_detail = defaultdict(Counter)
for aspect, direction, word in pos_aspects:
    if direction == '正面':
        pos_detail[aspect][word] += 1

neg_detail = defaultdict(Counter)
for aspect, direction, word in neg_aspects:
    if direction == '负面':
        neg_detail[aspect][word] += 1

# ========== 6. 总结展示 ==========
col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ 优点")
    for aspect, word_counter in sorted(pos_detail.items(), key=lambda x: -sum(x[1].values())):
        total = sum(word_counter.values())
        if total == 0:
            continue
        phrases = [f"{aspect}{w}" for w, c in word_counter.most_common(3)]
        st.write(f"- **{aspect}**（提及 {total} 次）：{'、'.join(phrases)}")

with col2:
    st.subheader("❌ 缺点")
    for aspect, word_counter in sorted(neg_detail.items(), key=lambda x: -sum(x[1].values())):
        total = sum(word_counter.values())
        if total == 0:
            continue
        phrases = [f"{aspect}{w}" for w, c in word_counter.most_common(3)]
        st.write(f"- **{aspect}**（提及 {total} 次）：{'、'.join(phrases)}")

# ========== 7. 可视化 ==========
st.header("三、可视化分析")

all_aspects = sorted(set(list(pos_detail.keys()) + list(neg_detail.keys())))
if all_aspects:
    pos_vals = [sum(pos_detail.get(a, Counter()).values()) for a in all_aspects]
    neg_vals = [sum(neg_detail.get(a, Counter()).values()) for a in all_aspects]

    x = np.arange(len(all_aspects))
    width = 0.35

    # 柱状图与说明同行
    col_desc1, col_img1 = st.columns([1, 2])
    with col_desc1:
        st.markdown("**各属性提及次数对比**")
        st.markdown("柱状图展示每个属性在好评和差评中被提到的次数。绿色代表好评，红色代表差评，柱子越高说明该属性被讨论得越多。")
    with col_img1:
        fig2, ax2 = plt.subplots(figsize=(6, 3.2))
        ax2.bar(x - width/2, pos_vals, width, label='好评提及', color='#FF6B35')
        ax2.bar(x + width/2, neg_vals, width, label='差评提及', color='#FFB088')
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_aspects, rotation=15, fontsize=8)
        ax2.set_ylabel('提及次数', fontsize=8)
        ax2.set_title('各属性在好评/差评中的提及次数对比', fontsize=9)
        ax2.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)

    # 雷达图与说明同行
    col_desc2, col_img2 = st.columns([1, 2])
    with col_desc2:
        st.markdown("**属性情感雷达图**")
        st.markdown("雷达图从多个维度对比好评与差评的分布。橙色区域越往外，说明该属性在好评中被提及得越多；浅色区域代表差评。")
    with col_img2:
        angles = np.linspace(0, 2*np.pi, len(all_aspects), endpoint=False).tolist()
        pos_vals_r = pos_vals + [pos_vals[0]]
        neg_vals_r = neg_vals + [neg_vals[0]]
        angles_r = angles + [angles[0]]

        fig3, ax3 = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax3.plot(angles_r, pos_vals_r, 'o-', linewidth=2, label='好评', color='#FF6B35')
        ax3.fill(angles_r, pos_vals_r, alpha=0.25, color='#FF6B35')
        ax3.plot(angles_r, neg_vals_r, 'o-', linewidth=2, label='差评', color='#FFB088')
        ax3.fill(angles_r, neg_vals_r, alpha=0.25, color='#FFB088')
        ax3.set_xticks(angles)
        ax3.set_xticklabels(all_aspects, fontsize=8)
        ax3.set_title('属性情感雷达图', fontsize=9)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        st.pyplot(fig3)
else:
    st.warning("未抽取到任何属性，可能是属性词典与数据不匹配。")

st.markdown("---")
st.caption("课程项目 · 基于大众点评评论的优缺点挖掘与可视化")

