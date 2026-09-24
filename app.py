import re
import os
import jieba
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import streamlit as st

# ========== 页面配置 ==========
st.set_page_config(page_title="大众点评评论优缺点总结", page_icon="🍽️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #FFF9F5; }
    h1, h2, h3 { color: #E85D2F !important; }
    section[data-testid="stSidebar"] { background-color: #FFF3EC; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🍽️ 大众点评评论优缺点总结系统")
st.markdown("基于情感分类与属性抽取的餐饮评论分析工具")

# ========== 中文字体（只读本地，不联网） ==========
FONT_CANDIDATES = [
    "SimHei.ttf",
    "main/SimHei.ttf",
    "/mount/src/pingjia/SimHei.ttf",
    "/mount/src/pingjia/main/SimHei.ttf",
    "/tmp/SimHei.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)

if FONT_PATH:
    try:
        fm.fontManager.addfont(FONT_PATH)
        _name = fm.FontProperties(fname=FONT_PATH).get_name()
        plt.rcParams['font.sans-serif'] = [_name, 'DejaVu Sans']
    except Exception:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8

# ========== 通用工具：放大查看 ==========
@st.dialog("🔍 放大查看", width="large")
def show_big(fig):
    st.pyplot(fig)

def show_chart_with_zoom(fig, key):
    st.pyplot(fig)
    if st.button("🔍 放大查看", key=key, use_container_width=True):
        show_big(fig)

# ========== 加载数据 ==========
@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path, encoding='gb18030', low_memory=False)

FILE_PATH = '大众点评评论数据.csv'
try:
    df_raw = load_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"找不到数据文件：{FILE_PATH}")
    st.stop()

st.success(f"数据加载成功，共 {len(df_raw)} 条评论")

# ========== 清洗标签 ==========
df = df_raw[['Content_review', 'Rating']].rename(
    columns={'Content_review': 'review', 'Rating': 'label'}
)
df = df.dropna(subset=['review', 'label'])
df['review'] = df['review'].astype(str)
df = df[df['label'] != 3].reset_index(drop=True)
df['label'] = df['label'].apply(lambda x: 1 if x >= 4 else 0)

# 关键：限制最大数据量，避免 Cloud 内存爆掉
MAX_ROWS = 30000
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

# ========== 采样 ==========
sample_size = st.sidebar.slider("采样数量", min_value=1000, max_value=20000, value=5000, step=1000)
if len(df) > sample_size:
    df_used = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
else:
    df_used = df.reset_index(drop=True)

st.write(f"当前使用数据：**{len(df_used)}** 条")
st.write(f"好评：**{len(df_used[df_used['label']==1])}** 条 ｜ 差评：**{len(df_used[df_used['label']==0])}** 条")

# ========== 分词（只对采样数据做） ==========
stopwords = set(['的','了','还','很','也','就','都','和','与','在','是','有','一','个','这','那','不','我','你','他','她','它','们','但','而','且','或','被','把','给','让','从','到','对','为','以','于','之','其','此','该','等','着','过','吗','呢','吧','啊','呀','哦','嗯','这个','那个','什么','怎么','可以','没有','不是'])

def _tokenize(text):
    text = re.sub(r'[^\u4e00-\u9fa5]', '', str(text))
    return [w for w in jieba.cut(text) if w not in stopwords and len(w) > 1]

with st.spinner("正在分词..."):
    df_used['tokens'] = df_used['review'].apply(_tokenize)
    df_used['clean'] = df_used['tokens'].apply(lambda ws: ' '.join(ws))

with st.expander("查看预处理示例"):
    for i in range(min(3, len(df_used))):
        st.write(f"**原文**：{df_used['review'].iloc[i][:80]}...")
        st.write(f"**清洗**：{df_used['clean'].iloc[i][:80]}...")
        st.write("---")

# ========== 情感分类 ==========
st.header("一、情感分类")

X_train, X_test, y_train, y_test = train_test_split(
    df_used['clean'], df_used['label'], test_size=0.3, random_state=42,
    stratify=df_used['label']
)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(class_weight='balanced', max_iter=300, solver='liblinear')
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)
report = classification_report(y_test, y_pred, target_names=['差评', '好评'], zero_division=0)

# ---- 板块 A ----
st.subheader("A. 评价好坏：数据构成")
real_pos = int((df_used['label'] == 1).sum())
real_neg = int((df_used['label'] == 0).sum())

col_text1, col_img1 = st.columns([1, 2])
with col_text1:
    st.markdown("**这一块看的是：评论本身是好还是差**")
    st.markdown(
        f"当前使用的**全部数据**共 **{len(df_used)}** 条评论：\n\n"
        f"- 好评：**{real_pos}** 条\n"
        f"- 差评：**{real_neg}** 条"
    )
with col_img1:
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    bars = ax1.bar(['好评', '差评'], [real_pos, real_neg],
                   color=['#FF6B35', '#FFB088'], width=0.5)
    ax1.set_ylabel('评论条数', fontsize=9)
    ax1.set_title(f'评价好坏：当前 {len(df_used)} 条中各有多少', fontsize=11)
    for b in bars:
        h = b.get_height()
        ax1.annotate(f'{int(h)}', (b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords='offset points',
                     ha='center', fontsize=9)
    ax1.set_ylim(0, max(real_pos, real_neg) * 1.2)
    plt.tight_layout()
    show_chart_with_zoom(fig1, key="zoom_A")

st.markdown("---")

# ---- 板块 B ----
st.subheader("B. 判断准确度：整体表现")
correct_total = int((y_test == y_pred).sum())
wrong_total = len(y_test) - correct_total
acc = accuracy_score(y_test, y_pred)

col_text2, col_img2 = st.columns([1, 2])
with col_text2:
    st.markdown("**这一块看的是：系统整体判得准不准**")
    st.markdown(
        f"从当前数据中抽出 **{len(y_test)}** 条（30%）做检验：\n\n"
        f"- 判对：**{correct_total}** 条\n"
        f"- 判错：**{wrong_total}** 条\n"
        f"- 准确度：**{acc:.1%}**"
    )
    if acc >= 0.9:
        st.success("😄 准确度很高。")
    elif acc >= 0.8:
        st.info("🙂 准确度不错。")
    else:
        st.warning("😐 准确度一般。")
with col_img2:
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    bars2 = ax2.bar(['判对', '判错'], [correct_total, wrong_total],
                    color=['#E85D2F', '#FFD9C2'], width=0.5)
    ax2.set_ylabel('评论条数', fontsize=9)
    ax2.set_title(f'判断准确度：{len(y_test)} 条测试中的对错', fontsize=11)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f'{int(h)}', (b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords='offset points',
                     ha='center', fontsize=9)
    ax2.set_ylim(0, len(y_test) * 1.15)
    plt.tight_layout()
    show_chart_with_zoom(fig2, key="zoom_B")

st.progress(min(acc, 1.0), text=f"准确度 {acc:.1%}")

# ---- 详细指标 ----
with st.expander("📊 详细指标可视化（精确率 / 召回率 / F1）"):
    prec_neg = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_neg  = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_neg   = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    prec_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_pos  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_pos   = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    col_desc, col_chart = st.columns([1, 2])
    with col_desc:
        st.markdown("**三个指标分别是什么意思**")
        st.markdown(
            "- **精确率**：系统说是某一类的，有多少是真的\n"
            "- **召回率**：真实的某一类，有多少被找出来\n"
            "- **F1**：精确率和召回率的综合分"
        )
        st.markdown(
            f"**好评**：精确率 {prec_pos:.1%}，召回率 {rec_pos:.1%}，F1 {f1_pos:.1%}\n\n"
            f"**差评**：精确率 {prec_neg:.1%}，召回率 {rec_neg:.1%}，F1 {f1_neg:.1%}"
        )
    with col_chart:
        metrics = ['精确率', '召回率', 'F1']
        pos_scores = [prec_pos, rec_pos, f1_pos]
        neg_scores = [prec_neg, rec_neg, f1_neg]
        x = np.arange(len(metrics))
        w = 0.38
        fig3, ax3 = plt.subplots(figsize=(5.5, 3.2))
        b1 = ax3.bar(x - w/2, pos_scores, w, label='好评', color='#FF6B35')
        b2 = ax3.bar(x + w/2, neg_scores, w, label='差评', color='#FFB088')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics, fontsize=10)
        ax3.set_ylim(0, 1.1)
        ax3.set_ylabel('得分', fontsize=9)
        ax3.set_title('好评 / 差评 三项指标对比', fontsize=11)
        ax3.legend(fontsize=9)
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                ax3.annotate(f'{h:.2f}', (b.get_x() + b.get_width()/2, h),
                             xytext=(0, 3), textcoords='offset points',
                             ha='center', fontsize=8)
        plt.tight_layout()
        show_chart_with_zoom(fig3, key="zoom_C")

    with st.expander("查看原始分类报告文本"):
        st.code(report)

# ========== 优缺点抽取 ==========
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
    '多':'多','足':'足','大':'大','好':'好','不错':'不错','干净':'干净','方便':'方便',
    '热情':'热情','周到':'周到','舒适':'舒适','满意':'满意','棒':'棒','优秀':'优秀',
    '喜欢':'喜欢','漂亮':'漂亮','好吃':'好吃','香':'香','实惠':'实惠','快':'快',
    '新鲜':'新鲜','美味':'美味','赞':'赞'
}
neg_words = {
    '少':'少','小':'小','差':'差','脏':'脏','吵':'吵','旧':'旧','破':'破','坏':'坏',
    '失望':'失望','糟糕':'糟糕','难':'难','远':'远','贵':'贵','慢':'慢','难吃':'难吃',
    '咸':'咸','淡':'淡','冷':'冷','腻':'腻','一般':'一般','坑':'坑'
}

def extract_aspects_from_tokens(words):
    results = []
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

pos_aspects = []
for tokens in df_used[df_used['label']==1]['tokens']:
    pos_aspects.extend(extract_aspects_from_tokens(tokens))
neg_aspects = []
for tokens in df_used[df_used['label']==0]['tokens']:
    neg_aspects.extend(extract_aspects_from_tokens(tokens))

pos_detail = defaultdict(Counter)
for aspect, direction, word in pos_aspects:
    if direction == '正面':
        pos_detail[aspect][word] += 1

neg_detail = defaultdict(Counter)
for aspect, direction, word in neg_aspects:
    if direction == '负面':
        neg_detail[aspect][word] += 1

col1, col2 = st.columns(2)
with col1:
    st.subheader("✅ 优点")
    for aspect, wc in sorted(pos_detail.items(), key=lambda x: -sum(x[1].values())):
        total = sum(wc.values())
        if total == 0: continue
        phrases = [f"{aspect}{w}" for w, c in wc.most_common(3)]
        st.write(f"- **{aspect}**（提及 {total} 次）：{'、'.join(phrases)}")
with col2:
    st.subheader("❌ 缺点")
    for aspect, wc in sorted(neg_detail.items(), key=lambda x: -sum(x[1].values())):
        total = sum(wc.values())
        if total == 0: continue
        phrases = [f"{aspect}{w}" for w, c in wc.most_common(3)]
        st.write(f"- **{aspect}**（提及 {total} 次）：{'、'.join(phrases)}")

# ========== 可视化 ==========
st.header("三、可视化分析")
all_aspects = sorted(set(list(pos_detail.keys()) + list(neg_detail.keys())))
if all_aspects:
    pos_vals = [sum(pos_detail.get(a, Counter()).values()) for a in all_aspects]
    neg_vals = [sum(neg_detail.get(a, Counter()).values()) for a in all_aspects]
    x = np.arange(len(all_aspects))
    width = 0.35

    col_desc1, col_img1b = st.columns([1, 2])
    with col_desc1:
        st.markdown("**各属性提及次数对比**")
        st.markdown("柱状图展示每个属性在好评和差评中被提到的次数。")
    with col_img1b:
        fig4, ax4 = plt.subplots(figsize=(6, 3.2))
        ax4.bar(x - width/2, pos_vals, width, label='好评提及', color='#FF6B35')
        ax4.bar(x + width/2, neg_vals, width, label='差评提及', color='#FFB088')
        ax4.set_xticks(x)
        ax4.set_xticklabels(all_aspects, rotation=15, fontsize=8)
        ax4.set_ylabel('提及次数', fontsize=8)
        ax4.set_title('各属性在好评/差评中的提及次数对比', fontsize=9)
        ax4.legend(fontsize=8)
        plt.tight_layout()
        show_chart_with_zoom(fig4, key="zoom_bar")

    col_desc2, col_img2b = st.columns([1, 2])
    with col_desc2:
        st.markdown("**属性情感雷达图**")
        st.markdown("雷达图从多个维度对比好评与差评。")
    with col_img2b:
        angles = np.linspace(0, 2*np.pi, len(all_aspects), endpoint=False).tolist()
        pos_vals_r = pos_vals + [pos_vals[0]]
        neg_vals_r = neg_vals + [neg_vals[0]]
        angles_r = angles + [angles[0]]
        fig5, ax5 = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax5.plot(angles_r, pos_vals_r, 'o-', linewidth=2, label='好评', color='#FF6B35')
        ax5.fill(angles_r, pos_vals_r, alpha=0.25, color='#FF6B35')
        ax5.plot(angles_r, neg_vals_r, 'o-', linewidth=2, label='差评', color='#FFB088')
        ax5.fill(angles_r, neg_vals_r, alpha=0.25, color='#FFB088')
        ax5.set_xticks(angles)
        ax5.set_xticklabels(all_aspects, fontsize=8)
        ax5.set_title('属性情感雷达图', fontsize=9)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        show_chart_with_zoom(fig5, key="zoom_radar")
else:
    st.warning("未抽取到任何属性。")

st.markdown("---")
st.caption("课程项目 · 基于大众点评评论的优缺点挖掘与可视化")
