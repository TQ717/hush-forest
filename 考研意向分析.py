
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# 读取数据
DATA_PATH = "市调数据.xlsx"
df = pd.read_excel(DATA_PATH)

# 字段映射
col_map = {
    # 考研相关特征（X）
    "12、以下是关于“读研内容与未来就业关系”的一些说法，请根据您的认同程度打分。（1-5分，1=很不同意，5=很同意）—我认为硕士阶段学习的内容与我理想的就业岗位关系密切。": "RelationJob",
    "12、我认为通过考研可以提高自身学历和能力。": "EnhanceAbility",
    "12、我希望通过读研来提升未来进入更好城市 / 行业 / 平台的竞争力。": "PromotePlatform",
    "12、我考虑读研的一个原因是“暂时回避就业压力”，延迟进入职场。": "AvoidPressure",
    "12、我的家庭普遍认为读到硕士及以上更体面，并希望我至少读研。": "FamilyExpect",
    "12、我身边准备考研或读研的同学比例很高，对我产生较大影响。": "PeerInfluence",
    "14、在考虑是否考研时，我会认真思考成本、风险和不确定性，而不是只关注能否考上。（1-5分，1=很不同意，5=很同意）": "RationalTrade",

    # 因变量（Y）
    "高低考研意向（1-3分为低意向，3.1-5分为高意向）": "y"
}

use_cols = list(col_map.keys())
missing_cols = [c for c in use_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"你的数据缺少这些列，请检查表头是否一致：{missing_cols}")

data = df[use_cols].rename(columns=col_map).copy()

def to_binary_label(v):
    s = str(v).strip().lower()
    if s in {"high", "1", "高", "高意向", "high意向"}:
        return 1
    if s in {"low", "0", "低", "低意向", "low意向"}:
        return 0
    try:
        x = float(s)
        return 1 if x > 3 else 0
    except:
        raise ValueError(f"无法识别的y取值：{v}")

data["y"] = data["y"].apply(to_binary_label).astype(int)

# 处理缺失值
data = data.dropna(axis=0).reset_index(drop=True)

X = data[
    ["RelationJob", "EnhanceAbility", "PromotePlatform", "AvoidPressure",
     "FamilyExpect", "PeerInfluence", "RationalTrade"]
].astype(float)

y = data["y"].astype(int)

# 切分训练/测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y
)

# 建模：随机森林
rf = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=5,
    bootstrap=True,
    max_features="sqrt",   # sklearn中auto已弃用；分类默认sqrt，等价于SPSSPRO的auto直觉
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# 训练集 / 测试集评估
def eval_binary(model, Xp, yp, name: str):
    proba = model.predict_proba(Xp)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(yp, pred)
    rec = recall_score(yp, pred)
    pre = precision_score(yp, pred)
    f1 = f1_score(yp, pred)
    auc = roc_auc_score(yp, proba)

    print(f"\n[{name}]")
    print(f"Accuracy={acc:.3f}  Recall={rec:.3f}  Precision={pre:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    print("Confusion matrix:\n", confusion_matrix(yp, pred))
    return acc, rec, pre, f1, auc

train_metrics = eval_binary(rf, X_train, y_train, "Train")
test_metrics  = eval_binary(rf, X_test,  y_test,  "Test")

# 5折交叉验证（输出均值）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "roc_auc": "roc_auc"
}
cv_res = cross_validate(rf, X, y, cv=cv, scoring=scoring, n_jobs=-1)

print("\n[Cross Validation (5-fold)]")
for k in scoring.keys():
    print(f"{k}: mean={cv_res['test_' + k].mean():.3f}  std={cv_res['test_' + k].std():.3f}")

# 特征重要性
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n[Feature Importance]")
print((imp * 100).round(2).astype(str) + "%")

# ROC曲线
RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title("ROC Curve - Graduate Intention (RF)")
plt.show()