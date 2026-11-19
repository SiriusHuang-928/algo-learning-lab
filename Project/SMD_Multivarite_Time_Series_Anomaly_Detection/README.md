## 多变量时序无监督异常检测（SMD 数据集）

### 项目简介

本项目基于 Server Machine Dataset（SMD）服务器多指标时序数据集，实现**无监督时序异常检测**。
核心思路：只用正常样本训练重构模型，通过模型对时序序列的重构误差判断是否为异常点，适合服务器运维指标、传感器时序监控场景。
一共实现 2 个对比模型：

1. Conv1D-AE：一维卷积自编码器，基于卷积提取局部时序特征，重构时序
2. TemporalTransformer：时序 Transformer，依靠自注意力捕捉长距离时序依赖

> 
> 任务类型：无监督异常检测
> 评估指标：ROC-AUC、PR-AUC、PA-F1（时序异常检测常用指标）

### 目录结构
.
├── dataset.py               # SMD数据集加载、滑窗采样
├── train.py                 # 训练、评估主脚本，支持模型切换
├── utils/
│   ├── __init__.py
│   └── metrics.py           # 指标计算：roc_auc、pr_auc、pa_f1
├── models/
│   ├── __init__.py
│   ├── conv1d_ae.py         # Conv1D自编码器
│   └── temporal_transformer.py # TemporalTransformer模型
├── ServerMachineDataset/    # SMD原始数据集（自行下载放入，git忽略）
├── logs/                    # 训练日志、指标csv（git忽略）
└── checkpoints/             # 保存模型权重（git忽略）

### 环境依赖

Python >=3.9，推荐 CUDA GPU 环境加速训练

### 数据集准备

1. 下载 SMD（ServerMachineDataset），数据集原仓库：[https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)
2. 下载完成后，将数据集文件夹放到项目根目录，文件夹命名为 `ServerMachineDataset`
3. 数据集包含服务器多指标时序 txt 文件：训练集全部为正常样本，测试集包含正常、异常样本以及对应标签

> 
> 本仓库不包含原始数据集，请自行下载

### 训练命令
# 训练 Conv1D-AE
python train.py --model conv1d_ae

# 训练 TemporalTransformer
python train.py --model temporal_transformer

### 实验结果说明

- 训练方式：无监督，仅使用正常时序样本训练
- 推理逻辑：滑窗采样，计算重构误差作为异常分数
- 指标说明
  - ROC-AUC：区分正常 / 异常的整体判别能力
  - PR-AUC：正负样本不均衡场景下更可靠的指标
  - PA-F1：时序点调整后的 F1 分数，是工业时序异常检测主流指标

### 补充说明

- 窗口大小、学习率、epoch 等超参可在 train.py 内修改
- 训练完成后权重自动保存至 checkpoints，指标自动写入 logs
- 可继续拓展 OmniAnomaly 等对比模型
