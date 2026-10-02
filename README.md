# SmartInspect

233-token 词法图 → 共享 ProNE → 同类全部 seed 配对 → 较小完整函数 AST 作为卷积核的 ExactType GPU 匹配。同一对同时超过两级阈值才命中。代码独立于原工程。

Python 3.10+，需 NVIDIA GPU 和支持 CUDA 的 PyTorch。在本目录运行：

```bash
python3 -m pip install torch==2.11.0 numpy==2.2.6 scipy==1.15.3 scikit-learn==1.7.2 tree-sitter==0.25.2 tree-sitter-solidity==1.2.13
python3 smartinspect.py
```

内置一个 seed 和一个变量重命名后的目标函数。实测 `status: ok`、`graph_score: 1.0`、AST `score: 1.0`、`detected: true`。

自有函数：`python3 smartinspect.py --seed seed.sol --target target.sol --op A`。每个文件放一个完整函数，`--seed`、`--target` 可重复。A/B/C 的严格阈值分别为 `(0.3, 0.2)`、`(0.7, 0.2)`、`(0.3, 0.8)`。多类别可导入 `detect(seeds, targets)`，其中 `seeds` 是类别到函数源码列表的字典。示例只验证方法运行，不包含实验 seed 库或数据集。
