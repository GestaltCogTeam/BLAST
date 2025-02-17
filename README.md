# <div align="center"> BLAST: Balanced Sampling Time Series Corpus for UniversalForecasting Models </div>

<img src="assets/BLAST.png" alt="TheTable" style="zoom:42%;" />


> The advent of universal time series forecasting models has revolutionized zero-shot forecasting across diverse domains, yet the critical role of data diversity in training these models remains underexplored. Existing large-scale time series datasets often suffer from inherent biases and imbalanced distributions, leading to suboptimal model performance and generalization. To address this gap, we introduce BLAST, a novel pre-training corpus designed to enhance data diversity through a balanced sampling strategy. First, BLAST incorporates 321 billion observations from publicly available datasets and employs a comprehensive suite of statistical metrics to characterize time series patterns. Then, to facilitate pattern-oriented sampling, the data is implicitly clustered using grid-based partitioning. Furthermore, by integrating grid sampling and grid mixup techniques, BLAST ensures a balanced and representative coverage of diverse patterns. Experimental results demonstrate that models pre-trained on BLAST achieve state-of-the-art performance with a fraction of the computational resources and training tokens required by existing methods. Our findings highlight the pivotal role of data diversity in improving both training efficiency and model performance for the universal forecasting task.

---
这个仓库包括了产生BLAST语料库的代码。
其中，`raw_data_construction`, `metrics_calculation`, `feature_construction`, `dimension_reduction`, and `sampling`文件夹包含了上图中对应的部分。BLAST的原始数据包含了321B个观测值，大小约为3.4TB。采样后的BLAST包含3M条长度最大为4096的时间序列，大小约2.7G。我们将会在审稿结束之后，在HuggingFace上开源上述数据。

## 💿 Requirements

The code is built based on Python 3.11. The required packages can be installed using the following command:

```bash
pip install -r requirements.txt
```

## 📂 Prepare Raw Data
Download Data from HuggingFace, and placed them in `datasets/raw_datasets/` folder.

## BLAST
### 1. Raw Data Construction
> Read raw data from different sources, and save them in an unified format
```bash
python raw_data_construction/Chronos_read_and_save.py
python raw_data_construction/LOTSA_read_and_save.py
python raw_data_construction/Monash_read_and_save.py
python raw_data_construction/UCR_read_and_save.py
python raw_data_construction/UAD_read_and_save.py
```

The processed data will be saved in `datasets/processed_datasets/` folder.

### 2. Metrics Calculation
```bash
python metrics_calculation/main.py
```

The results will be saved in `metrics_calculation/output/` folder.

### 3. Feature Construction
```bash
python feature_construction/main.py
```

The results will be saved in `feature_construction/output/` folder.

### 4. Dimension Reduction

Use jupyter notebook to run `dimension_reduction/dimension_reduction.ipynb`

### 5. Sampling
```bash
python sampling/data_sampler.py
python sampling/mixup.py
```

## TODO
- [ ] Clean the code, and add comments
- [ ] Add the scripts of training TimeMoE on BLAST