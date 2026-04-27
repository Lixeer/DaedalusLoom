# 数据集获取
- [静态手势数据包含[fist,paper,scissor]](https://baidu.com)


# 基准测试报告
## 2. 实验结果与曲线分析

我们将实验结果按照学习率和批大小的不同组合进行分组对比。

### 2.1 高学习率配置 (LR=1e-5, BS=64)
在此配置下，模型能够较快地捕捉特征，但也更考验架构对权重的更新稳定性。

| 模型架构 | 训练曲线图 |
| :--- | :--- |
| **CNN** | ![](/doc/img/batchsize=64_lr=1e-5_model=CNN.png) |
| **MLP** | ![](/doc/img/batchsize=64_lr=1e-5_model=MLP.png) |

**分析：** 比较两图可以发现，在较高学习率下，CNN通常展现出更好的特征提取能力和收敛稳定性，而MLP可能在损失下降速度上有所差异。

### 2.2 低学习率配置 (LR=1e-7)
在极低学习率下，我们观察不同 Batch Size 对训练平滑度的影响。

#### 2.2.1 小批次 (BS=64)
| 模型架构 | 训练曲线图 |
| :--- | :--- |
| **CNN** | ![](/doc/img/batchsize=64_lr=1e-7_model=CNN.png) |
| **MLP** | ![](/doc/img/batchsize=64_lr=1e-7_model=MLP.png) |

#### 2.2.2 大批次 (BS=512)
| 模型架构 | 训练曲线图 |
| :--- | :--- |
| **CNN** | ![](/doc/img/batchsize=512_lr=1e-7_model=CNN.png) |
| **MLP** | ![](/doc/img/batchsize=512_lr=1e-7_model=MLP.png) |