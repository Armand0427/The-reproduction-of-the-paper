"""
NGAFID Data Loader
按照论文方法(2210.07317v1)处理NGAFID数据

功能：
1. 读取all_flights目录下的flight_header.csv
2. 排除'same'航班，'before'标0，'after'标1
3. 读取/all_flights/one_parq里每个航班的parquet文件
4. 截断/填充到4096时间步（取最后4096秒）
5. 用前向填充+后向填充处理NaN
6. 不缩放（因为用MiniRocket）
7. 返回X (n_flights, 4096, 23) 和 y (n_flights,)
"""
import pandas as pd
import numpy as np
import os
import glob
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class NGAFIDDataLoader:
    """加载和处理NGAFID飞行数据"""

    def __init__(self, data_dir: str = "all_flights"):
        """
        初始化数据加载器

        Parameters
        ----------
        data_dir : str
            数据目录路径，包含flight_header.csv和one_parq子目录
        """
        self.data_dir = data_dir
        self.header_path = os.path.join(data_dir, "flight_header.csv")
        self.parquet_dir = os.path.join(data_dir, "one_parq")

        # 特征列（排除timestep和cluster）共23个特征
        self.feature_cols = [
            'volt1', 'volt2', 'amp1', 'amp2', 'FQtyL', 'FQtyR', 'E1 FFlow',
            'E1 OilT', 'E1 OilP', 'E1 RPM', 'E1 CHT1', 'E1 CHT2', 'E1 CHT3',
            'E1 CHT4', 'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4', 'OAT',
            'IAS', 'VSpd', 'NormAc', 'AltMSL'
        ]

        # 目标序列长度：4096个时间步（秒）
        self.seq_len = 4096

    def load_flight_metadata(self) -> pd.DataFrame:
        """
        加载航班元数据并过滤

        Returns
        -------
        pd.DataFrame
            过滤后的航班元数据，包含Master Index和标签(0=before, 1=after)
        """
        print(f"Loading flight metadata from {self.header_path}")
        metadata = pd.read_csv(self.header_path)

        # 排除'same'航班
        metadata = metadata[metadata['before_after'] != 'same'].copy()

        # 创建标签：before=0, after=1
        metadata['y'] = metadata['before_after'].map({'before': 0, 'after': 1})

        print(f"Loaded {len(metadata)} flights (excluding 'same')")
        print(f"Class distribution: {metadata['y'].value_counts().to_dict()}")

        return metadata

    def process_single_flight(self, flight_data: pd.DataFrame) -> np.ndarray:
        """
        处理单个航班数据：截断/填充到4096时间步，处理NaN

        按照论文方法：
        1. 取最后4096秒的数据（基于timestep列）
        2. 如果不足4096秒，用NaN填充前部
        3. 对NaN进行前向填充+后向填充
        4. 不进行特征缩放

        Parameters
        ----------
        flight_data : pd.DataFrame
            单个航班的DataFrame，包含所有特征列和timestep列

        Returns
        -------
        np.ndarray
            处理后的序列，形状为(seq_len, n_features)
        """
        # 确保按timestep排序
        if 'timestep' in flight_data.columns:
            flight_data = flight_data.sort_values('timestep')
        else:
            # 如果没有timestep列，假设数据已按时间顺序
            pass

        # 提取特征列
        features = flight_data[self.feature_cols].copy()

        # 获取时间步数
        n_timesteps = len(features)

        if n_timesteps >= self.seq_len:
            # 截断：取最后seq_len个时间步（最后4096秒）
            features = features.iloc[-self.seq_len:]
        else:
            # 填充：用NaN填充前部，以便后续插值
            padded = pd.DataFrame(
                np.full((self.seq_len, len(self.feature_cols)), np.nan),
                columns=self.feature_cols
            )
            padded.iloc[-n_timesteps:] = features.values
            features = padded

        # 处理NaN：前向填充 + 后向填充（论文方法）
        features = features.ffill().bfill()
        # 如果仍有NaN（例如所有值都是NaN），用0填充
        features = features.fillna(0)

        return features.values.astype(np.float32)

    def load_all_flights(self, max_files: int = None, batch_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载并处理所有航班数据（主方法）

        Parameters
        ----------
        max_files : int, optional
            最大处理的parquet文件数（用于测试）
        batch_size : int
            批次大小，控制内存使用

        Returns
        -------
        X : np.ndarray
            特征数组，形状为(n_flights, seq_len, n_features)
        y : np.ndarray
            标签数组，形状为(n_flights,)
        """
        # 加载元数据
        metadata = self.load_flight_metadata()

        # 创建Master Index到标签的映射
        idx_to_label = dict(zip(metadata['Master Index'], metadata['y']))
        target_indices = set(idx_to_label.keys())

        # 获取所有parquet文件
        parquet_files = glob.glob(os.path.join(self.parquet_dir, "*.parquet"))
        if max_files is not None:
            parquet_files = parquet_files[:max_files]
            print(f"Limiting to {max_files} parquet files for testing")

        print(f"Found {len(parquet_files)} parquet files to process")

        # 初始化存储
        all_X = []
        all_y = []
        found_indices = set()

        # 分批处理文件
        total_batches = (len(parquet_files) + batch_size - 1) // batch_size
        for batch_idx in range(0, len(parquet_files), batch_size):
            batch_files = parquet_files[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            print(f"\nProcessing batch {batch_num}/{total_batches} "
                  f"({len(batch_files)} files)")

            for file_path in batch_files:
                try:
                    # 读取parquet文件（仅需要的列以减少内存）
                    df = pd.read_parquet(file_path, columns=self.feature_cols + ['timestep'])

                    # 获取文件中唯一的Master Index
                    file_indices = set(df.index.unique())

                    # 找出目标索引的交集
                    common_indices = target_indices.intersection(file_indices)
                    # 过滤掉已处理的索引
                    common_indices = [idx for idx in common_indices if idx not in found_indices]

                    if not common_indices:
                        continue

                    # 为每个索引提取数据
                    for master_idx in common_indices:
                        flight_data = df.loc[master_idx]
                        if isinstance(flight_data, pd.Series):
                            flight_data = pd.DataFrame([flight_data])

                        # 处理航班数据
                        processed = self.process_single_flight(flight_data)

                        all_X.append(processed)
                        all_y.append(idx_to_label[master_idx])
                        found_indices.add(master_idx)

                        # 进度报告
                        if len(found_indices) % 100 == 0:
                            print(f"  Processed {len(found_indices)} flights")

                except Exception as e:
                    print(f"Error processing {os.path.basename(file_path)}: {e}")
                    continue

            # 每批结束后报告进度
            print(f"  Batch {batch_num} complete. Total flights processed: {len(found_indices)}")

        # 转换为numpy数组
        if not all_X:
            print("Warning: No flights found!")
            return np.array([]), np.array([])

        X = np.stack(all_X)
        y = np.array(all_y)

        print(f"\n{'='*60}")
        print("Data loading completed!")
        print(f"{'='*60}")
        print(f"Total flights processed: {len(found_indices)}")
        print(f"X shape: {X.shape} (n_flights, {self.seq_len}, {len(self.feature_cols)})")
        print(f"y shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"X dtype: {X.dtype}, min: {X.min():.3f}, max: {X.max():.3f}, mean: {X.mean():.3f}")
        print(f"NaN count in X: {np.isnan(X).sum()}")

        return X, y

    def save_data(self, X: np.ndarray, y: np.ndarray, prefix: str = "ngafid"):
        """
        保存处理后的数据到文件

        Parameters
        ----------
        X : np.ndarray
            特征数组
        y : np.ndarray
            标签数组
        prefix : str
            文件名前缀
        """
        np.save(f"{prefix}_X.npy", X)
        np.save(f"{prefix}_y.npy", y)
        print(f"Data saved to {prefix}_X.npy and {prefix}_y.npy")


def main():
    """主函数：运行数据加载器"""
    import argparse

    parser = argparse.ArgumentParser(description='NGAFID数据加载器')
    parser.add_argument('--data-dir', default='all_flights',
                       help='数据目录路径')
    parser.add_argument('--max-files', type=int, default=None,
                       help='限制处理的parquet文件数量（用于测试）')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='批次大小（控制内存使用）')
    parser.add_argument('--save', action='store_true',
                       help='保存数据到文件')
    parser.add_argument('--save-prefix', default='ngafid',
                       help='保存文件的前缀')

    args = parser.parse_args()

    print("="*60)
    print("NGAFID Data Loader")
    print("="*60)

    # 初始化加载器
    loader = NGAFIDDataLoader(data_dir=args.data_dir)

    # 加载数据
    X, y = loader.load_all_flights(
        max_files=args.max_files,
        batch_size=args.batch_size
    )

    # 保存数据
    if args.save and len(X) > 0:
        loader.save_data(X, y, prefix=args.save_prefix)

    print("\nDone!")


if __name__ == "__main__":
    main()