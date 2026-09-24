# ===== 过采样：差评复制到好评的 5 倍 =====
X_train_pos = X_train_vec[y_train == 1]
X_train_neg = X_train_vec[y_train == 0]
y_train_pos = y_train[y_train == 1]
y_train_neg = y_train[y_train == 0]

n_pos = X_train_pos.shape[0]
n_neg = X_train_neg.shape[0]

if n_neg > 0 and n_pos > n_neg:
    ratio = int(np.ceil(n_pos / n_neg * 5.0))
    X_train_neg_up = vstack([X_train_neg] * ratio)
    y_train_neg_up = np.tile(y_train_neg, ratio)
else:
    X_train_neg_up = X_train_neg
    y_train_neg_up = y_train_neg

X_train_vec = vstack([X_train_pos, X_train_neg_up])
y_train = np.concatenate([y_train_pos, y_train_neg_up])

st.sidebar.success(
    f"✅ 训练集：好评 {X_train_pos.shape[0]} 条，"
    f"差评过采样到 {X_train_neg_up.shape[0]} 条（{ratio}倍）"
)

# 训练
clf = LogisticRegression(
    class_weight={0: 5.0, 1: 1.0},
    max_iter=1000,
    solver='liblinear'
)
clf.fit(X_train_vec, y_train)

# ===== 双层兜底 =====
proba = clf.predict_proba(X_test_vec)
THRESHOLD = 0.07
y_pred = (proba[:, 1] >= THRESHOLD).astype(int)

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
    # 新增隐含差评
    '没什么味道', '不太新鲜', '不太干净', '不太热情', '一般般吧',
    '不如', '比不上', '没有以前', '也就那样', '只能算',
    '有点失望', '略贵', '小贵', '偏贵', '偏咸', '偏淡',
    '不太推荐', '不会再点', '不怎么样', '不太满意',
    '等了半小时', '等了一小时', '等了很久',
    '环境一般', '服务一般', '味道一般', '分量少', '分量小',
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
