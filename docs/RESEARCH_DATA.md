# 处理后的研究数据

五份 CSV 从所提供的 `new_data.zip` 逐字节复制，属于真实的处理后研究输入，与 `examples/synthetic.csv` 合成数据分别管理。

| 文件 | 行数 | 列数 | 标签数量 |
|---|---:|---:|---|
| [fire_preprocessed_data.csv](../data/processed/fire_preprocessed_data.csv) | 1215 | 24 | {'1': 631, '-1': 584} |
| [algeria_wildfire_enriched_for_pipeline.csv](../data/processed/algeria_wildfire_enriched_for_pipeline.csv) | 517 | 25 | {'0': 247, '1': 270} |
| [algeria_wildfire_for_pipeline.csv](../data/processed/algeria_wildfire_for_pipeline.csv) | 517 | 15 | {'0': 247, '1': 270} |
| [america_wildfire_for_pipeline.csv](../data/processed/america_wildfire_for_pipeline.csv) | 64196 | 22 | {'1': 32098, '0': 32098} |
| [america_wildfire_horizon_for_pipeline.csv](../data/processed/america_wildfire_horizon_for_pipeline.csv) | 64196 | 24 | {'1': 32098, '0': 32098} |

目标字段为 `fire_occurred`。完整字段清单、标签分布和 SHA-256 校验码见 [manifest.json](../data/processed/manifest.json)。五份文件均已扫描空白值、非数值和非有限值，结果记录在该清单中。这验证的是数据序列化，不是科学正确性或标签质量。

`america_*` 文件包含纬度和经度，`algeria_*` 文件包含 X/Y 字段。未进行去标识化或坐标脱敏。文件名按原样保留；未根据文件名核实地理来源、发布者、许可、标签构建方式或预处理来源。不能仅凭文件名判断数据集身份或复用权限。

本次数据发布不包含原始来源文件、位置增强的来源 CSV、训练权重、逐行预测或实验产物。两份 `algeria_*` 文件及两份 `america_*` 文件属于不同处理变体，不能自动视为独立数据集拼接。

## 使用主数据运行

```bash
python -m src.pipelines.run_full_pipeline --config configs/processed_primary.yaml
```

该配置将主数据 CSV 接入已发布的两阶段实现，采用三个训练轮次进行基本运行检查，并使用回退知识先验。这不是论文最终的私有配置，不能据此认定已复现发表的指标。如需开展另行明确的实验，请增加训练轮次并提供有记录依据的知识信息。
