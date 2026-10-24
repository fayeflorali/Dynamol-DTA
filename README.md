# DynaMol-DTA

Dynamic Pocket-guided Multiscale Molecular Learning for Drug–Target Affinity Prediction

以 DMFF-DP1 的动态口袋与在线联合图机制为骨架，合并 ESM-2 逐残基表示和 MolUNet 多尺度原子编码。
工程独立运行，不会在运行时导入三个同学项目中的文件。来源见 [PROVENANCE.md](PROVENANCE.md)。

## 当前模型

```text
蛋白质窗口 → 冻结 ESM-2 缓存 [L,640] → 可训练投影 [L,256]
                                             ├→ masked mean → MolUNet query
药物 SMILES → RDKit 原子/键图 → MolUNet → 多尺度原子表示 [A,256]
                                             ↓
                          ESM 残基 + MolUNet 原子 → DP1 动态口袋门控
                                             ↓
                     门控残基 + 药物原子 + 全局节点 → 联合图 GIN
                                             ↓
SMILES BiLSTM + ESM 残基 → 序列交互注意力 → 与图表示融合 → 亲和力
```

- ESM-2 本体冻结并离线提取；投影层、MolUNet、选择器、GIN、预测头在线训练。
- 药物图仅有 RDKit 原子和化学键；联合图有残基、原子、全局节点。两套边索引分别管理。
- MolUNet 解池化输出与 RDKit 原子顺序一致，避免通过 SMILES token 推测图节点编号。
- 保留 DMFF 风格双向 SMILES LSTM 和三路序列注意力；融合头是新模型的实现，不能视为旧模型权重直接拼接。
- 口袋默认采用 DP1 的 straight-through 二值门控和 0.5 残差信号下限，并不删除节点/边。
- `pocket_ratio=0.15` 是软分数初始化先验，不是严格选择 15% 残基的 top-k。
- 当前口袋在 CSV 已给定的蛋白质窗口内建模，不是全长蛋白质口袋搜索。

## 环境

```sh
pip install -r requirements.txt
```

CUDA 版 PyTorch/PyG 应按对应 CUDA 版本安装。在线提取 ESM-2 另需 `pip install fair-esm`；
已有缓存时，训练过程不需要加载 fair-esm 或下载 ESM 模型。

本机已使用 `C:/Users/11392/miniconda3/envs/dmff-cpu/python.exe` 完成 CPU 验证。

## 先验证代码

从本目录执行：

```sh
python -m unittest tests.test_integration -v
python smoke_test.py
```

`smoke_test.py` 创建 10 条临时用途的合成数据并保存到独立运行目录，完成一折训练、验证、测试及预测重放。
**合成 ESM 特征不是实际 ESM-2 提取结果，测试分数不能用于汇报模型效果。**
已有一次验证产物位于 `runs/smoke_verified/`，其中包括 `fold0.pt`、`metrics.json` 和输入数据。

## 准备真实数据

CSV 必需列：

```text
compound_iso_smiles,target_key,target_sequence,affinity,
target_sequence_start,target_sequence_end
```

Davis 还使用 `uniprot` 列作为缓存文件名；没有该列时使用 `target_key`（例如 KIBA）。
窗口采用零起点、左闭右开 `[start,end)`，必须满足 `end-start == len(target_sequence)`。
affinity 按 CSV 的现有标度读入，代码不额外转换为 pKd。

- `esm-dir/<uniprot>.pt`：窗口对应的 `[L,640]` float tensor，兼容同学版本的缓存；
  新提取器额外保存原序列与模型名称，可检测同长度但不同序列的错配。
- `contact-dir/<uniprot>.npy`：完整蛋白质方形接触图，程序按窗口切片，默认阈值 `>0.8`。
- 同一个缓存 ID 不能对应两个不同窗口，遇到冲突直接报错。
- 不截断 SMILES 或蛋白质窗口；batch 内动态 padding。超长窗口需调小 batch size。
- 当前 MolUNet 路径要求每个药物至少包含一个化学键，无键分子会明确报错。
- 不自动复制大数据、下载模型或静默跳过不合格样本。

可用新提取器生成带序列校验信息的缓存：

```sh
python extract_esm2_features.py --dataset davis --csv /path/davis_processed.csv --out-dir /path/esm2_embedding_davis --device cuda:0
```

如缓存已存在，默认跳过；需要按当前 CSV 重新提取时加 `--overwrite`。

## 训练与独立评估

将示例路径替换为实际路径。相对 CLI 路径按当前工作目录解析，默认词表/配置按脚本目录解析。

```sh
python train.py --csv /path/davis_processed.csv --esm-dir /path/esm2_embedding_davis --contact-dir /path/target_contact_map_davis --device cuda:0 --epochs 100 --batch-size 32
```

默认五折；加 `--fold 0` 只运行第一折。`--workers 2` 可启用数据加载子进程。
初次调试可用 CPU、小批次和一轮训练；实际数据仍需完整且各划分非空。
`--output` 可指定运行目录，默认创建带时间戳的新目录。

数据划分直接复用 DP1 的固定 pair 划分算法：默认五折对应约 70%/10%/20%。
划分固定 seed=0，`--seed` 只影响训练随机性。
可以通过 `--split-dir` 指向已有 DP1 划分目录，但 CSV 内容、文件 stem、行数和折数必须匹配。
PNA 度分布仅从训练折药物图计算。每折从新模型开始，不跨折传递权重。

```sh
python evaluate.py --checkpoint /path/run/fold0.pt --csv /path/davis_processed.csv --esm-dir /path/esm2_embedding_davis --contact-dir /path/target_contact_map_davis --device cuda:0
```

checkpoint 包含 config、PNA 度分布、词表尺寸、模型参数、样本划分和数据文件指纹。
独立评估核对 CSV、词表、ESM 缓存、接触图及口袋标签后，只评估保存的测试索引。
早期模型 checkpoint 不直接兼容；当前默认仅 ESM-2 使用外部预训练表示，其余融合模块从头训练。

## 动态口袋与监督配置

编辑 `config.json`：

| 配置 | 含义 |
|---|---|
| `pocket_mode=dynamic_pocket` | 药物条件残基门控 |
| `pocket_mode=static_sparse` | 不使用当前药物条件的残基门控 |
| `pocket_mode=full_online` | 残基全部通过，联合图仍在线构建 |
| `pocket_loss_weight=0` | 默认只训练亲和力损失 |

可通过 `--pocket-labels pocket_labels_davis.npz` 加载 DP1 的窗口标签，文件以 target_key 为键，
值为与窗口等长的二值向量。将 `pocket_loss_weight` 设为正值启用辅助监督。
缺少注释的样本不参与口袋损失；DP1 原有“辅助监督不更新外部编码器”的梯度隔离逻辑被保留。
独立评估时须提供训练时使用的同一标签文件。

当前已提供三种口袋模式；尚未实现关闭 ESM-2/MolUNet 的完整消融基线。
比较三个旧项目与新模型时，还需要统一数据窗口、固定划分、训练预算和预训练来源。

## 代码导航

- `dynamol/model.py`：完整融合前向计算。
- `dynamol/data.py`：RDKit 构图、窗口/特征校验、双图 batch。
- `dynamol/dynamic_pocket.py`：DP1 选择器。
- `dynamol/molunet/`：MolUNet 编码器及依赖。
- `dynamol/dmff_layers.py`：DP1 的联合图 GIN 和注意力。
- `dynamol/engine.py`：训练、指标、保存/恢复。
- `train.py`、`evaluate.py`：有主入口保护的命令行入口。

这是经过小规模功能验证的初版合并实现，尚未在真实 Davis/KIBA 数据上完成训练或效果比较。
