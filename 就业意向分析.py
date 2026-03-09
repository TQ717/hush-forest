
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# 读取数据
DATA_PATH = "2.13市调-就业有实习等经历版.xlsx"  # TODO: 修改为你的实际文件名
df = pd.read_excel(DATA_PATH)

# 字段映射（中文列名 -> 英文特征名）
col_map = {
    # 就业相关特征（X）
    "18、关于你最近一次的实习 / 校内外实践 / 预录用机会，请根据你的真实感受对以下方面打分。（如果有多次，请选择你认为最具有代表性的一次经历）（1-5分，1=很不同意，5=很同意）—感觉实习质量高。": "InternQuality",
    "18、岗位目标明确度高。": "JobClarity",
    "18、求职准备度高。": "JobPrep",
    "18、自我效能感高。": "SelfEfficacy",
    "18、薪酬 / 补贴水平基本符合或接近我的心理预期。。": "SalaryMatch",
    "18、想尽快经济独立。": "EconIndep",

    # 因变量（Y）
    "高低就业意向(0-3为低意向)": "y"
}

use_cols = list(col_map.keys())
missing_cols = [c for c in use_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"你的数据缺少这些列，请检查表头是否一致：{missing_cols}")

data = df[use_cols].rename(columns=col_map).copy()

# 处理Y：转为二分类(0/1)
def to_binary_label(v):
    s = str(v).strip().lower()
    if s in {"high", "1", "高", "高意向", "high意向"}:
        return 1
    if s in {"low", "0", "低", "低意向", "low意向"}:
        return 0
    try:
        x = float(s)
        return 0 if x <= 3 else 1
    except:
        raise ValueError(f"无法识别的y取值：{v}")

data["y"] = data["y"].apply(to_binary_label).astype(int)

# 缺失处理：按行删除
data = data.dropna(axis=0).reset_index(drop=True)

X = data[
    ["InternQuality", "JobClarity", "JobPrep", "SelfEfficacy", "SalaryMatch", "EconIndep"]
].astype(float)

y = data["y"].astype(int)

# 切分训练/测试（0.7, shuffle=True）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y
)

# 建模：随机森林
rf = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=5,
    bootstrap=True,
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 训练/测试评估
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

# 5折交叉验证
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
plt.title("ROC Curve - Employment Intention (RF)")
plt.show()