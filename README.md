# FGST-KEAGN

**中文** | [English](README.en.md)

两阶段森林火灾风险预测框架的核心算法实现：**FGAT-FS** 特征选择与 **LLM-KEAGN** 知识增强自适应门控。

仓库维护者：[yan-wenguang](https://github.com/yan-wenguang)。

## 概述

森林火灾风险预测是防灾减灾决策的重要前提。然而，多数预测方法在处理高维多源特征时存在严重的相关性冗余，且特征权重完全来自统计拟合，与火灾发生的物理机制缺乏一致性，导致其在动态风险条件下的可解释性不足。针对这些挑战，本文提出一种融合结构化特征选择与大语言模型驱动知识增强的可解释森林火灾风险预测框架。

## 算法实现

- `src/graph/`：互信息、相关性和分类协同矩阵，特征节点统计，以及考虑冗余的特征选择。
- `src/models/fgat_fs.py`：特征图注意力、可学习的关系融合、特征评分和软选择。
- `src/models/llm_keagn.py`：数据分支与知识分支、上下文编码、自适应门控和预测头。
- `src/losses/`：分类、稀疏性、分支对齐和上下文平滑损失。
- `src/knowledge/`：特征卡片构建与本地 JSON 先验提供器。
- `src/trainers/`：两个阶段的训练、基于验证集的检查点选择，以及结果输出。
- `src/pipelines/`：各阶段的独立运行入口与完整两阶段运行入口。

当前知识提供器读取本地 JSON 先验。缺失的先验通过特征卡片和统计信息回退生成。所提供的实现不会在线请求大语言模型。`LLM-KEAGN` 沿用论文和模块的命名，并不表示运行期间调用了大语言模型服务。

## 安装

使用 Python 3.10–3.12，在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install -e .
python -m pip check
```

可安装的 `fgst_keagn_support` 包包含辅助工具。核心算法模块使用 `src.*` 导入路径，需要从仓库根目录运行。

## 运行核心算法

```bash
python -m src.pipelines.run_full_pipeline --config configs/core_demo.yaml
```

公开配置使用 `examples/synthetic.csv`，两个阶段各训练三个轮次。这是实际模型的基本运行示例，不是论文基准实验。所选特征、指标、权重和预测结果保存在 `artifacts/core_demo/`；Git 会忽略这些生成文件。

如需使用自己的数值型 CSV，请复制 `configs/core_demo.yaml`，设置 `data.csv_path`、`data.label_column`、上下文列、训练参数和知识文件路径。标签必须为 0/1 或 -1/+1。特征选择的上下界应与可用特征数量匹配，并至少保留一个数值型上下文列。详见 [CORE_ALGORITHM.md](docs/CORE_ALGORITHM.md)。

## 数据

`examples/synthetic.csv` 包含 240 行独立生成的数据，具有三个输入特征、一个上下文字段和一个二分类目标。[DATA.md](docs/DATA.md) 说明其来源和结构。五份处理后的研究 CSV 已放在 [`data/processed/`](data/processed/)，包括 1,215 行主数据和四份外部数据变体。[RESEARCH_DATA.md](docs/RESEARCH_DATA.md) 说明字段、校验码、标签数量与来源限制。本次发布不包含原始输入、研究专用先验文件、训练检查点和完整实验归档。

使用主数据运行：

```bash
python -m src.pipelines.run_full_pipeline --config configs/processed_primary.yaml
```

公开配置采用较短的训练轮次用于验证运行，不是论文最终实验配置。

## 测试与辅助示例

```bash
python -m unittest discover -s tests -v
python -m examples.generate
python -m examples.run --config configs/synthetic_example.json
```

逻辑回归示例与两阶段核心算法分别运行。[EVALUATION.md](docs/EVALUATION.md) 说明辅助评估器；原始核心评估器采用自身的指标约定，详见 [CORE_ALGORITHM.md](docs/CORE_ALGORITHM.md)。[VALIDATION.md](docs/VALIDATION.md) 区分历史辅助包检查与核心算法运行检查。

## 引用与权利

尚未提供最终论文标题、发表场所和 DOI。[CITATION.md](docs/CITATION.md) 给出仓库引用指引，不虚构发表信息。本版本未授予通用开源许可；依赖包仍适用各自的许可。

仓库组织方式参考：[GAT](https://github.com/PetarV-/GAT) 与 [GraphSAGE](https://github.com/williamleif/GraphSAGE)。未复制其代码或数据。
