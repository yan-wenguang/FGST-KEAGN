# 二分类评估约定

约定版本：**1.0**。入口：`fgst_keagn_support.utils.metrics.compute_binary_metrics`。

## 有效输入

- 真实标签必须为非空、一维、有限数值数组，也接受布尔标签。
- 标签必须全部属于 `{0, 1}` 或 `{-1, +1}` 中的一种编码；带符号的标签会被明确归一化。拒绝混合编码、不支持的值、缺失标签、字符串和复数。
- 预测概率必须为长度相同的一维数组，包含 `[0, 1]` 范围内的有限实数。拒绝布尔型或字符串型概率数组。
- 阈值必须为 `[0, 1]` 范围内的有限实数标量。`probability >= threshold` 时预测为正类。
- 调用方必须提供归一化后正类的概率。对于 0/1 类别，常规示例使用 `predict_proba(X)[:, 1]`。

不会静默修复标签或失败的预测。在此辅助函数中，`decision_function` 的分数不能替代概率，尽管部分排序接口接受范围不受限制的决策分数。

## 指标值与有效性状态

每个结果包含七个指标值、`status` 状态映射、`support` 样本计数，以及包含 TP、TN、FP、FN 的 `confusion_matrix`。未定义的指标以 Python `None` / JSON `null` 表示，不能将其作为普通数值参与平均。

| 指标 | 定义 | 未定义条件 |
|---|---|---|
| `roc_auc` | scikit-learn 二分类 ROC-AUC | 真实标签缺少任一类别 |
| `average_precision` | 以召回率增量为权重，对精确率进行非插值加权求和 | 真实标签无正类 |
| `accuracy` | (TP + TN) / n | 通过非空输入校验后无未定义情况 |
| `precision` | TP / (TP + FP) | 无预测正类 |
| `recall` | TP / (TP + FN) | 真实标签无正类 |
| `f1` | 2TP / (2TP + FP + FN) | 分母为零 |
| `mcc` | (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | 分母为零 |

真实标签全部为正类时，平均精确率可以为 1，但 ROC-AUC 仍未定义。对于包含两个真实类别的数据，如果没有预测正类，精确率和 MCC 未定义，而召回率和 F1 定义为零。有效性按各指标分别判断，不使用统一的单类别回退值。

**平均精确率不等同于精确率—召回率曲线下的梯形积分面积。** 原来的 `pr_auc` 名称已替换为 `average_precision`，`auc` 已替换为 `roc_auc`。这是有意的 API 变更，不是对历史研究结果的重新解释。

状态原因值包括 `defined`、`requires_both_true_classes`、`no_positive_truth`、`no_predicted_positives`、`zero_f1_denominator` 和 `zero_mcc_denominator`。

## 数据与字段安全

`infer_schema` 拒绝缺失标签、缺失上下文引用、重复列名，以及将标签作为上下文。CSV 表头会在 pandas 自动改写重复列名之前检查。示例的基线输入列表必须非空且无重复，并且是已声明的静态非标签特征的子集；上下文被明确排除。

这些边界检查防止公开示例中此前发现的“将标签作为特征”配置错误，不能替代对真实研究数据中语义泄漏或依赖关系的审查。

## 序列化与相关工具

`save_json` 在打开输出文件之前，通过 `allow_nan=False` 验证标准 JSON 序列化，避免无效数值内容截断现有目标文件。这不保证原子写入或故障恢复。内容必须使用兼容标准 JSON 的 Python 值；调用方应明确转换 NumPy 专用值。

`stable_softmax` 接受非空、有限实数、一维数值数组，将整数输入转换为浮点数，减去最大值后计算指数并归一化。极小的权重可能下溢为零。该函数不是多维张量 softmax 接口。

## 验证证据与范围

测试覆盖已知的非完美排序、无效输入拒绝、各指标未定义条件、混淆矩阵计数和合成数据基线。这验证的是所选公开辅助函数的行为，不是私有 FGST-KEAGN 评估器或论文结果。

参考：[平均精确率](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)、[ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)。
