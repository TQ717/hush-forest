# hush-forest
考研/就业分析模型
# 考研还是就业：问卷数据建模与随机森林分类分析

本项目基于问卷调查数据，围绕“本科毕业生选择考研还是就业”的主题，构建两套二分类模型，用于识别影响 **考研意向** 与 **就业意向** 的关键因素，并输出可解释的特征重要性与常用分类评估指标（Accuracy/Recall/Precision/F1/AUC）。

> 说明：本仓库主要用于研究复现与方法展示，代码以可读性与可复用性为主。

---

## 研究任务概览

本项目包含两条建模主线：

### 1）考研意向模型（高/低）
- 目标：预测受访者的 **考研意向水平**（高 / 低）
- 输入特征（7项，见下方英文变量映射）
- 模型：随机森林分类（Random Forest Classifier）
- 输出：评估指标 + ROC曲线 + 特征重要性

### 2）就业意向模型（高/低）
- 目标：预测受访者的 **就业意向水平**（高 / 低）
- 输入特征（6项，见下方英文变量映射）
- 模型：随机森林分类（Random Forest Classifier）
- 输出：评估指标 + ROC曲线 + 特征重要性

---

## 特征字段英文名称

为便于代码复现与论文/报告写作，本项目统一使用如下英文变量名。

### 考研相关特征（7项）
| 英文名 | 含义 |
|---|---|
| RelationJob | 学习内容与就业关系 |
| EnhanceAbility | 提高学历能力 |
| PromotePlatform | 提升平台竞争力 |
| AvoidPressure | 回避就业压力 |
| FamilyExpect | 家庭期望 |
| PeerInfluence | 同伴影响 |
| RationalTrade | 理性权衡 |

### 就业相关特征（6项）
| 英文名 | 含义 |
|---|---|
| InternQuality | 感觉实习质量高 |
| JobClarity | 岗位目标明确度高 |
| JobPrep | 求职准备度高 |
| SelfEfficacy | 自我效能感高 |
| SalaryMatch | 薪酬符合心理预期 |
| EconIndep | 想尽快经济独立 |

---

## 方法与评估指标

### 模型
- 随机森林分类（Random Forest Classifier）
- 训练/测试划分：70% / 30%
- 交叉验证：5折（StratifiedKFold）

### 评估指标
- **Accuracy**：整体预测正确率  
- **Recall**：正类召回率（更关注“漏判”时尤其重要）  
- **Precision**：预测为正类中实际为正类的比例（更关注“误判”时尤其重要）  
- **F1-Score**：Precision 与 Recall 的综合平衡指标  
- **AUC**：ROC 曲线下面积，用于衡量整体区分能力

---

## 目录结构（建议）

你可以按如下方式组织仓库：
python                    3.12
scikit-learn              1.8.0
pandas                    2.3.3
