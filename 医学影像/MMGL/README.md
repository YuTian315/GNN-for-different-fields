
### 论文名称 - Multi-modal Graph Learning for Disease Prediction

#### 环境配置

1. Python 3.6
2. Pyotrch 1.1 以上
3. munkres

#### 运行

1. 按照Data preprocessing先处理数据
2. ABIDE 和 Tadpole 两个数据集运行参数不一样，具体参考.sh文件
3. 运行完代码可使用attn_vis.py 进行注意力的可视化

#### 模型概述

1. 使用REF(递归特征消除)进行特征选择
2. 功能链接特征，人口学特征，解剖学特征，等特征先通过自注意力融合
3. 融合之后产生共享性特征（多模态通过注意力矩阵加权之后的特征）和特异性特征（注意力矩阵拉平的特征向量）
4. 通过一个损失函数对特异性特征进行监督
5. 特异性特征和共享性特征进行拼接之后用于构造自适应图
6. 通过cos相似性（代码用的内积）来构图，通过一个图损失函数来限制的图的稀疏性和平滑性（具体看文章部分）
7. 构造的图和特征通过2层图卷积进行节点分类

![overchart.png](./overchart.png)


#### 文章引用

**该文章发在 TMI 2022**

Zheng, S., Zhu, Z., Liu, Z., Guo, Z., Liu, Y., Yang, Y., & Zhao, Y. (2022). Multi-modal Graph Learning for Disease Prediction. IEEE Transactions on Medical Imaging.


**转投**

#### 其他

1. 文章特征选择部分存在一定争议，具体看开源代码（论文有链接）的issue
2. 关于Tadpole数据的结果没有复现出来
