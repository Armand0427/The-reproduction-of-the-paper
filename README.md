# NGAFID Maintenance Event Binary Classification

复现论文《A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID》中的维护事件二分类实验。

> 这是一次不成熟的尝试，结果与论文有差异

## 任务

判断一个航班是维修前（标签0）还是维修后（标签1）。

- 输入：23个传感器 × 4096时间步的多变量时间序列
- 输出：0（维修前）或 1（维修后）

## 数据

- 来源：[NGAFID Dataset](https://doi.org/10.5281/zenodo.6624956)
- 样本数：22,431个航班（维修前12,140，维修后10,291）
- 预处理：截断/填充到4096时间步，NaN前向+后向填充，不缩放

## 方法

- 特征提取：MiniRocketMultivariate (num_kernels=10000)
- 分类器：RidgeClassifierCV (alphas=logspace(-3,3,10))
- 验证：5折分层交叉验证 (StratifiedKFold, random_state=42)

## 结果

| 指标 | 均值 ± 标准差 |
|------|---------------|
| Accuracy | 0.6909 ± 0.0045 |
| F1 Score | 0.6530 ± 0.0035 |
| AUC | 0.7498 ± 0.0033 |

详细结果见 [results/](results/)。

> 注：论文Table 2中MiniRocket准确率为59.8%，本实验使用了更多数据（全部维修前后航班而非论文子集），因此结果不可直接对比。

## 如何运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
# 下载NGAFID数据集到 all_flights/ 目录

# 3. 生成npy文件
python src/ngafid_loader.py --save

# 4. 训练
python src/train.py
