# 数据说明

## 数据来源

本实验使用的数据集来自论文：

> Yang, H., & Desell, T. (2022). A Large-Scale Annotated Multivariate Time Series 
> Aviation Maintenance Dataset from the NGAFID. arXiv:2210.07317.

**论文链接**：https://arxiv.org/abs/2210.07317

**数据集下载**：
- Zenodo: https://doi.org/10.5281/zenodo.6624956
- Kaggle: https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid

## 数据内容

下载后包含：
all_flights/
├── flight_header.csv # 航班元数据（before/after/same、日期差、维修类型）
└── one_parq/ # 每个航班的传感器数据
├── 1.parquet
├── 2.parquet
└── ...
- **航班数**：28,935个
- **传感器**：23个，每秒记录一次
- **时间步长**：原始长度不一，处理后统一到4096

## 数据预处理

按照论文方法处理（详见 `src/ngafid_loader.py`）：

1. 排除 `before_after == 'same'` 的航班（维修当天）
2. `before` 标为 0，`after` 标为 1
3. 截断/填充到4096时间步（取最后4096秒）
4. NaN用前向填充 + 后向填充 + 0填充
5. 不缩放（MiniRocket不需要）

处理后得到：
- `ngafid_X.npy`：形状 (22431, 4096, 23)
- `ngafid_y.npy`：形状 (22431,)

## 为什么仓库里没有数据文件

- `ngafid_X.npy` 约8.5GB，超出GitHub限制
- `all_flights/` 原始数据约4.3GB，是公开数据集，可自行下载

## 如何获取数据并复现

```bash
# 1. 下载数据集到 all_flights/
# 从上面任一链接下载，解压到项目根目录的 all_flights/

# 2. 生成npy文件
python src/ngafid_loader.py --save

# 3. 训练
python src/train.py
