import re
import os
import jieba
import pandas as pd
import numpy as np
from scipy.sparse import vstack
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

# ========== 中文字体 ==========
FONT_CANDIDATES = [
    "Ubuntu_18.04_SimHei.ttf", "simhei.ttf", "SimHei.ttf",
    "/mount/src/pingjia/Ubuntu_18.04_SimHei.ttf",
    "/mount/src/pingjia/simhei.ttf",
    "/mount/src/pingjia/SimHei.ttf",
]
FONT_PATH = None
FONT_SIZE_MB = 0
for p in FONT_CANDIDATES:
    if os.path.exists(p):
        FONT_PATH = p
        FONT_SIZE_MB = os.path.getsize(p) / 1024 / 1024
        break

if FONT_PATH and FONT_SIZE_MB > 0.5:
    try:
        fm.fontManager.addfont(FONT_PATH)
        _real_name = fm.FontProperties(fname=FONT_PATH).get_name()
        plt.rcParams['font.sans-serif'] = [_real_name, 'DejaVu Sans']
    except Exception:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8

with st.sidebar.expander("🔤 字体诊断", expanded=False):
    if FONT_PATH:
        st.write(f"路径：`{FONT_PATH}`")
        st.write(f"大小：**{FONT_SIZE_MB:.2f} MB**")
        if FONT_SIZE_MB < 0.5:
            st.error("❌ 字体不完整（LFS 未拉取）")
        else:
            st.success("✅ 字体已加载")
    else:
        st.error("❌ 未找到字体文件")

# ========== 图片放大 ==========
@st.dialog("🔍 放大查看", width="large")
def show_big(fig):
    st.pyplot(fig)

def show_chart_with_zoom(fig, key):
    st.pyplot(fig)
    if st.button("🔍 放大查看", key=key, use_container_width=True):
        show_big(fig)

# ========== 1. 加载数据 ==========
CSV_URL = "https://media.githubusercontent.com/media/nikki-1-3/pingjia/refs/heads/main/%E5%A4%A7%E4%BC%97%E7%82%B9%E8%AF%84%E8%AF%84%E8%AE%BA%E6%95%B0%E6%8D%AE.csv"

@st.cache_data(show_spinner="正在下载数据集（首次约几十 MB，请稍候）...")
def load_data(url):
    return pd.read_csv(url, encoding='gb18030', low_memory=False)

try:
    df_raw = load_data(CSV_URL)
except Exception as e:
    st.error(f"数据下载失败：{e}")
    st.stop()

st.success(f"数据加载成功，共 {len(df_raw)} 条评论")

# ========== 2. 提取评论列和标签列 ==========
df = df_raw[['Content_review', 'Rating']].rename(columns={
    'Content_review': 'review',
    'Rating': 'label'
})
df = df.dropna(subset=['review', 'label'])
df['review'] = df['review'].astype(str)
df = df[df['label'] != 3].reset_index(drop=True)
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

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===== 关键改动：训练集里"下采样"好评，让好评差评比例接近 =====
# 现在训练集里好评 ~3300、差评 ~210，比例严重失衡。
# 把好评也下采样到 3 倍差评数量（即 ~630 条），让模型学得更平衡。
X_train_pos = X_train_vec[y_train == 1]
X_train_neg = X_train_vec[y_train == 0]
y_train_pos = y_train[y_train == 1]
y_train_neg = y_train[y_train == 0]

n_pos = X_train_pos.shape[0]
n_neg = X_train_neg.shape[0]

# 好评下采样到 3 倍差评
rng = np.random.RandomState(42)
if n_pos > n_neg * 3:
    pos_idx = rng.choice(n_pos, size=n_neg * 3, replace=False)
    X_train_pos = X_train_pos[pos_idx]
    y_train_pos = y_train_pos[pos_idx]
    n_pos = X_train_pos.shape[0]

# 差评过采样到和好评同量
if n_neg > 0 and n_pos > n_neg:
    ratio = int(np.ceil(n_pos / n_neg))
    X_train_neg_up = vstack([X_train_neg] * ratio)
    y_train_neg_up = np.tile(y_train_neg, ratio)
else:
    X_train_neg_up = X_train_neg
    y_train_neg_up = y_train_neg

X_train_vec = vstack([X_train_pos, X_train_neg_up])
y_train = np.concatenate([y_train_pos, y_train_neg_up])

st.sidebar.success(
    f"✅ 训练集：好评下采样到 {X_train_pos.shape[0]} 条，"
    f"差评过采样到 {X_train_neg_up.shape[0]} 条"
)

# 训练
clf = LogisticRegression(
    class_weight={0: 5.0, 1: 1.0},
    max_iter=1000,
    solver='liblinear'
)
clf.fit(X_train_vec, y_train)

# ===== 双层兜底 =====
# 第 1 层：模型概率阈值极低（0.05）
proba = clf.predict_proba(X_test_vec)
THRESHOLD = 0.05
y_pred = (proba[:, 1] >= THRESHOLD).astype(int)

# 第 2 层：硬规则，命中强差评或中性偏负词，直接判差评
STRONG_NEG = [
    # 强差评
    '难吃', '不好吃', '太咸', '太淡', '太辣', '太甜', '太油', '太腻', '腥', '异味',
    '不新鲜', '变质', '馊', '难以下咽',
    '脏', '太吵', '很吵', '环境差', '不卫生', '乱',
    '服务差', '态度差', '态度不好', '不理人', '冷漠', '催了', '等了很久', '等太久',
    '太贵', '不值', '坑', '宰客', '贵死', '性价比低',
    '失望', '差评', '再也不来', '不推荐', '踩雷', '拉黑', '恶心', '糟糕',
    '不会再', '很差', '不行', '烂', '别来', '避雷',
    # 中性偏负
    '一般', '还行吧', '不太', '有点', '稍微', '勉强', '凑合', '一般般',
    '没什么', '没有特别', '不算', '不太行', '就那么', '普通',
]
X_test_list = list(X_test)
hard_hits = 0
for i, text in enumerate(X_test_list):
    if any(kw in text for kw in STRONG_NEG):
        if y_pred[i] != 0:
            hard_hits += 1
        y_pred[i] = 0

if hard_hits > 0:
    st.sidebar.info(f"🔧 硬规则修正了 {hard_hits} 条漏判差评")

report = classification_report(y_test, y_pred, target_names=['差评', '好评'], zero_division=0)

# ---- A. 评价好坏 ----
st.subheader("A. 评价好坏：数据构成")
real_pos = int((df['label'] == 1).sum())
real_neg = int((df['label'] == 0).sum())

col_text1, col_img1 = st.columns([1, 2])
with col_text1:
    st.markdown("**这一块看的是：评论本身是好还是差**")
    st.markdown(
        f"当前使用的**全部数据**共 **{len(df)}** 条评论：\n\n"
        f"- 好评：**{real_pos}** 条\n"
        f"- 差评：**{real_neg}** 条"
    )
with col_img1:
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    bars = ax1.bar(['好评', '差评'], [real_pos, real_neg],
                   color=['#FF6B35', '#FFB088'], width=0.5)
    ax1.set_ylabel('评论条数', fontsize=9)
    ax1.set_title(f'评价好坏：全部 {len(df)} 条中各有多少', fontsize=11)
    for b in bars:
        h = b.get_height()
        ax1.annotate(f'{int(h)}', (b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords='offset points',
                     ha='center', fontsize=9)
    ax1.set_ylim(0, max(real_pos, real_neg) * 1.2)
    plt.tight_layout()
    show_chart_with_zoom(fig1, key="zoom_A")

st.markdown("---")

# ---- B. 判断准确度 ----
st.subheader("B. 判断准确度：整体表现")
correct_total = int((y_test == y_pred).sum())
wrong_total = len(y_test) - correct_total
acc = accuracy_score(y_test, y_pred)

col_text2, col_img2 = st.columns([1, 2])
with col_text2:
    st.markdown("**这一块看的是：系统整体判得准不准**")
    st.markdown(
        f"从全部数据中抽出 **{len(y_test)}** 条（30%）做检验：\n\n"
        f"- 判对：**{correct_total}** 条\n"
        f"- 判错：**{wrong_total}** 条\n"
        f"- 准确度：**{acc:.1%}**"
    )
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

# ---- C. 详细指标可视化 ----
st.subheader("C. 详细指标可视化")

prec_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec_pos  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1_pos   = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
prec_neg = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
rec_neg  = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
f1_neg   = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

col_desc, col_chart = st.columns([1, 2])
with col_desc:
    st.markdown("**三个指标分别是什么意思**")
    st.markdown(
        "- **精确率**：系统判为某一类的，有多少是真的\n"
        "- **召回率**：真实的某一类，有多少被系统找出来\n"
        "- **F1**：精确率和召回率的综合分"
    )
    st.markdown("---")
    st.markdown(
        f"**好评**：精确率 **{prec_pos:.0%}**，召回率 **{rec_pos:.0%}**，F1 **{f1_pos:.0%}**\n\n"
        f"**差评**：精确率 **{prec_neg:.0%}**，召回率 **{rec_neg:.0%}**，F1 **{f1_neg:.0%}**"
    )
    if rec_neg < 0.6:
        st.warning("⚠️ 差评召回率偏低，还有差评被漏掉。")
    elif rec_neg < 0.8:
        st.info("🙂 差评召回率不错，大部分差评能抓到。")
    else:
        st.success("✅ 差评召回率很高，差评几乎不漏。")

with col_chart:
    metrics = ['精确率', '召回率', 'F1']
    pos_scores = [prec_pos, rec_pos, f1_pos]
    neg_scores = [prec_neg, rec_neg, f1_neg]
    x = np.arange(len(metrics))
    w = 0.38
    fig3, ax3 = plt.subplots(figsize=(5.5, 3.4))
    b1 = ax3.bar(x - w/2, pos_scores, w, label='好评', color='#FF6B35')
    b2 = ax3.bar(x + w/2, neg_scores, w, label='差评', color='#FFB088')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=10)
    ax3.set_ylim(0, 1.15)
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

# ---- D. 混淆矩阵（折叠面板） ----
st.markdown("---")
with st.expander("📊 点击展开查看混淆矩阵（判对 / 判错 的四种情况）", expanded=False):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    col_desc, col_chart = st.columns([1, 2])
    with col_desc:
        st.markdown("**这张图在说什么**")
        st.markdown(
            "矩阵里每一格代表系统判断的一种结果：\n\n"
            f"- ✅ **真的差评，判成差评**：{tn} 条\n"
            f"- ❌ **真的差评，判成好评**：{fp} 条（漏抓的差评）\n"
            f"- ❌ **真的好，判成差评**：{fn} 条（误伤）\n"
            f"- ✅ **真的好，判成好评**：{tp} 条"
        )
        st.markdown("---")
        if fp > tn and fp > 20:
            st.error(f"⚠️ 有 **{fp}** 条真实差评被判成了好评，漏抓的差评比抓到的还多。")
        elif fp > 20:
            st.warning(f"⚠️ 有 **{fp}** 条真实差评被漏掉了，差评召回率还能再提高。")
        else:
            st.success("✅ 漏抓的差评很少，模型对差评的识别已经比较到位。")

    with col_chart:
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        im = ax4.imshow(cm, cmap='Oranges', aspect='auto')
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = 'white' if val > cm.max() * 0.5 else 'black'
                ax4.text(j, i, f'{val}', ha='center', va='center',
                         color=color, fontsize=16, fontweight='bold')
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['判成差评', '判成好评'], fontsize=10)
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['真实差评', '真实好评'], fontsize=10)
        ax4.set_xlabel('系统判断', fontsize=10)
        ax4.set_ylabel('实际情况', fontsize=10)
        ax4.set_title('混淆矩阵：四种判断结果', fontsize=11)
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        plt.tight_layout()
        show_chart_with_zoom(fig4, key="zoom_D")

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
    pos_reviews = df[df['label'] == 1]['review']
    neg_reviews = df[df['label'] == 0]['review']
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

# ========== 7. 可视化 ==========
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
        fig5, ax5 = plt.subplots(figsize=(6, 3.2))
        ax5.bar(x - width/2, pos_vals, width, label='好评提及', color='#FF6B35')
        ax5.bar(x + width/2, neg_vals, width, label='差评提及', color='#FFB088')
        ax5.set_xticks(x)
        ax5.set_xticklabels(all_aspects, rotation=15, fontsize=8)
        ax5.set_ylabel('提及次数', fontsize=8)
        ax5.set_title('各属性在好评/差评中的提及次数对比', fontsize=9)
        ax5.legend(fontsize=8)
        plt.tight_layout()
        show_chart_with_zoom(fig5, key="zoom_bar")

    col_desc2, col_img2b = st.columns([1, 2])
    with col_desc2:
        st.markdown("**属性情感雷达图**")
        st.markdown("雷达图从多个维度对比好评与差评。")
    with col_img2b:
        angles = np.linspace(0, 2*np.pi, len(all_aspects), endpoint=False).tolist()
        pos_vals_r = pos_vals + [pos_vals[0]]
        neg_vals_r = neg_vals + [neg_vals[0]]
        angles_r = angles + [angles[0]]
        fig6, ax6 = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax6.plot(angles_r, pos_vals_r, 'o-', linewidth=2, label='好评', color='#FF6B35')
        ax6.fill(angles_r, pos_vals_r, alpha=0.25, color='#FF6B35')
        ax6.plot(angles_r, neg_vals_r, 'o-', linewidth=2, label='差评', color='#FFB088')
        ax6.fill(angles_r, neg_vals_r, alpha=0.25, color='#FFB088')
        ax6.set_xticks(angles)
        ax6.set_xticklabels(all_aspects, fontsize=8)
        ax6.set_title('属性情感雷达图', fontsize=9)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        show_chart_with_zoom(fig6, key="zoom_radar")
else:
    st.warning("未抽取到任何属性。")

st.markdown("---")
st.caption("课程项目 · 基于大众点评评论的优缺点挖掘与可视化")
