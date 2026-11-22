# 学术代码仓库组织方式

[GAT](https://github.com/PetarV-/GAT) 提供模型代码、辅助工具、最小示例、经过测试的依赖和论文引用。[GraphSAGE](https://github.com/williamleif/GraphSAGE) 将小型示例数据与完整数据下载分别组织，并说明输入格式和运行方式。

本仓库以 models/graph/losses/trainers/pipelines 组织核心算法，以 configs 保存可执行示例配置，以 examples 保存独立生成的数据，以 tests 保存测试，以 docs 保存接口和数据约定。未复制参考仓库的代码或数据。

当前版本在早期仅包含辅助工具的基础上，增加了原始核心实现。五份处理后的研究 CSV 已公开；原始输入和实验归档仍未纳入仓库。
