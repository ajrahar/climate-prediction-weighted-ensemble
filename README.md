# 🌡️ Multi-City Daily Temperature Prediction on Java Island

### Using Weighted Ensemble of BigLSTM, BigGRU, and Transformer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: NASA POWER](https://img.shields.io/badge/Data-NASA%20POWER-blue)](https://power.larc.nasa.gov/)
[![Journal: JOSCEX](https://img.shields.io/badge/Journal-JOSCEX-orange)](https://shmpublisher.com/index.php/joscex)

> **Paper:** *Multi-City Daily Temperature Prediction on Java Island Using Weighted Ensemble of BigLSTM, BigGRU, and Transformer*
> **Authors:** Miftahul Fazi Raharja, Mulia Sulistiyono
> **Affiliation:** Department of Informatics, AMIKOM Yogyakarta University, Indonesia

-----

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Study Area](#-study-area)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

-----

## 🔍 Overview

This repository contains the full implementation of a **weighted deep learning ensemble** for daily temperature prediction across **6 climatologically diverse cities on Java Island, Indonesia**. Three architectures — BigLSTM, BigGRU, and WeatherTransformer — were trained on 26 years of NASA POWER meteorological data and combined via **inverse validation-loss weighting**.

### Key Contributions

1. Weighted ensemble of BigLSTM, BigGRU, and WeatherTransformer trained on 26 years of NASA POWER data for six Java cities representing distinct climate zones
1. Systematic evaluation across **5 chronological split configurations** (90/10 to 50/50) per city, yielding **90 total trained models**
1. Empirical analysis demonstrating that the **optimal split configuration is climate-zone-dependent**

-----

## 📊 Key Results

|Metric                     |Value (Best: 80/20 Split)|
|---------------------------|-------------------------|
|**RMSE**                   |0.5021°C                 |
|**MAE**                    |0.3981°C                 |
|**R²**                     |0.756                    |
|**MAPE**                   |1.58%                    |
|**Improvement vs Baseline**|13.6%                    |

**Per-city best RMSE:**

|City      |Best Split|RMSE (°C)|MAE (°C)|R²    |Impr. %|
|----------|----------|---------|--------|------|-------|
|Jakarta   |80/20     |0.3781   |0.2985  |0.7492|14.36% |
|Bandung   |60/40     |0.5779   |0.4590  |0.6288|15.04% |
|Semarang  |80/20     |0.4989   |0.3981  |0.7876|15.61% |
|Yogyakarta|80/20     |0.4536   |0.3612  |0.7294|12.27% |
|Surabaya  |90/10     |0.5490   |0.4365  |0.8978|11.96% |
|Malang    |80/20     |0.5429   |0.4293  |0.7890|12.35% |

-----

## 🗺️ Study Area

Six cities representing distinct climate zones across Java Island:

|City      |Elevation (m)|Climate Zone   |Coordinates     |
|----------|-------------|---------------|----------------|
|Jakarta   |8            |North Coastal  |6.21°S, 106.85°E|
|Bandung   |768          |Highland       |6.92°S, 107.62°E|
|Semarang  |5            |North Coastal  |6.99°S, 110.42°E|
|Yogyakarta|114          |Central Plateau|7.80°S, 110.37°E|
|Surabaya  |6            |North Coastal  |7.26°S, 112.75°E|
|Malang    |445          |Highland       |7.98°S, 112.63°E|

-----

## 🏗️ Architecture

### Ensemble Components

```
Input (14-day window × 12 features)
        │
        ├──── BigLSTM ────────► prediction_lstm
        │     (341K params)
        │
        ├──── BigGRU ─────────► prediction_gru
        │     (257K params)
        │
        └──── WeatherTransformer ► prediction_transformer
              (103K params)
                    │
                    ▼
        Inverse Validation-Loss Weighting
                    │
                    ▼
          Final Ensemble Prediction
```

|Model                 |Architecture                                 |Parameters|
|----------------------|---------------------------------------------|----------|
|**BigLSTM**           |3-layer LSTM, hidden=128, dropout=0.3        |~341,000  |
|**BigGRU**            |3-layer GRU, hidden=128, dropout=0.3         |~257,000  |
|**WeatherTransformer**|2 encoder layers, d_model=64, 4 heads, ff=256|~103,000  |

### Ensemble Weighting

```
wᵢ = (1 / val_lossᵢ) / Σ(1 / val_lossⱼ)
```

Weights are persisted to JSON checkpoint files for reproducibility.

-----

## 📦 Dataset

Data was retrieved from the **NASA POWER Daily Meteorology API**:

- **Period:** January 10, 2000 – May 22, 2026 (~9,630 observations/city)
- **Variables:** T2M (°C), PRECTOTCORR (mm/day), RH2M (%), WS10M (m/s)
- **API:** <https://power.larc.nasa.gov/>

### Feature Engineering (12 Features)

|Feature                              |Description                             |
|-------------------------------------|----------------------------------------|
|`TEMPERATURE`                        |Near-surface temperature T2M (°C)       |
|`PRECIPITATION`                      |Bias-corrected precipitation (mm/day)   |
|`HUMIDITY`                           |Relative humidity RH2M (%)              |
|`WIND_SPEED`                         |Wind speed at 10m WS10M (m/s)           |
|`month_sin`, `month_cos`             |Sinusoidal cyclical month encoding      |
|`day_sin`, `day_cos`                 |Sinusoidal day-of-year encoding         |
|`TEMP_LAG1`, `TEMP_LAG3`, `TEMP_LAG7`|Lagged temperature at 1, 3, 7 days prior|
|`TEMP_ROLL7`                         |7-day rolling mean temperature          |

-----

## 📁 Project Structure

```
climate-prediction-weighted-ensemble/
│
├── 📂 data/
│   ├── raw/                    # Raw NASA POWER data (per city)
│   └── processed/              # Preprocessed & feature-engineered data
│
├── 📂 models/
│   ├── big_lstm.py             # BigLSTM architecture
│   ├── big_gru.py              # BigGRU architecture
│   ├── weather_transformer.py  # WeatherTransformer architecture
│   └── ensemble.py             # Weighted ensemble module
│
├── 📂 notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training.ipynb       # Model training pipeline
│   ├── 04_evaluation.ipynb     # Results & visualizations
│   └── jawa_weather_multikota.ipynb  # Full multi-city pipeline
│
├── 📂 checkpoints/
│   └── weights/                # Saved ensemble weights (JSON)
│
├── 📂 results/
│   ├── metrics/                # RMSE, MAE, R², MAPE per city/split
│   └── figures/                # Heatmaps, residual plots, predictions
│
├── 📂 src/
│   ├── data_loader.py          # NASA POWER API fetcher
│   ├── preprocessing.py        # Feature engineering pipeline
│   ├── trainer.py              # Training loop & early stopping
│   └── evaluator.py            # Metrics & visualization
│
├── requirements.txt
├── config.yaml                 # Hyperparameters & settings
└── README.md
```

-----

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/ajrahar/climate-prediction-weighted-ensemble.git
cd climate-prediction-weighted-ensemble

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0
scipy>=1.11.0
jupyter>=1.0.0
```

-----

## 🚀 Usage

### 1. Fetch Data from NASA POWER API

```python
from src.data_loader import fetch_nasa_power

cities = {
    "Jakarta":    (-6.21, 106.85),
    "Bandung":    (-6.92, 107.62),
    "Semarang":   (-6.99, 110.42),
    "Yogyakarta": (-7.80, 110.37),
    "Surabaya":   (-7.26, 112.75),
    "Malang":     (-7.98, 112.63),
}

df = fetch_nasa_power(lat=-7.80, lon=110.37, start="20000110", end="20260522")
```

### 2. Train Models

```python
from src.trainer import train_ensemble

results = train_ensemble(
    city="Yogyakarta",
    split_ratio=0.8,        # 80/20 split
    window_size=14,
    epochs=500,
    patience=80,
    seed=42
)
```

### 3. Run Full Multi-City Pipeline

```bash
jupyter notebook notebooks/jawa_weather_multikota.ipynb
```

### 4. Evaluate & Visualize

```python
from src.evaluator import plot_heatmap, plot_residuals

plot_heatmap(results_dict)          # RMSE heatmap all cities × splits
plot_residuals(preds, actuals)      # Residual analysis
```

-----

## 📈 Results

### Average Ensemble Performance per Split

|Split    |RMSE (°C) |MAE (°C)  |R²        |MAPE     |Impr. %  |
|---------|----------|----------|----------|---------|---------|
|90/10    |0.5094    |0.4062    |0.7710    |1.59%    |12.2%    |
|**80/20**|**0.5021**|**0.3981**|**0.7556**|**1.58%**|**13.6%**|
|70/30    |0.5114    |0.4047    |0.7755    |1.60%    |12.9%    |
|60/40    |0.5197    |0.4111    |0.7538    |1.63%    |12.3%    |
|50/50    |0.5180    |0.4091    |0.7655    |1.62%    |12.4%    |

### Mean Ensemble Weight Distribution

|City      |W_LSTM|W_GRU|W_Transformer|Dominant   |
|----------|------|-----|-------------|-----------|
|Jakarta   |32.6% |32.0%|**35.4%**    |Transformer|
|Bandung   |33.5% |32.7%|**33.8%**    |Transformer|
|Semarang  |32.8% |33.1%|**34.1%**    |Transformer|
|Yogyakarta|32.9% |32.9%|**34.2%**    |Transformer|
|Surabaya  |32.6% |33.1%|**34.3%**    |Transformer|
|Malang    |32.8% |32.8%|**34.4%**    |Transformer|


> **Finding:** WeatherTransformer consistently receives the highest weight across all cities, suggesting its global attention mechanism captures complementary patterns not fully addressed by sequential recurrent models.

-----

## 📄 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{raharja2026multicity,
  title     = {Multi-City Daily Temperature Prediction on Java Island Using
               Weighted Ensemble of BigLSTM, BigGRU, and Transformer},
  author    = {Raharja, Miftahul Fazi and Sulistiyono, Mulia},
  journal   = {Journal of Soft Computing Exploration},
  year      = {2026},
  publisher = {SHM Publisher},
  url       = {https://shmpublisher.com/index.php/joscex}
}
```

-----

## 🙏 Acknowledgements

- **NASA POWER** — for providing open-access daily meteorological data (<https://power.larc.nasa.gov/>)
- **AMIKOM Yogyakarta University** — Department of Informatics
- Supervisor: **Mulia Sulistiyono, M.Kom.**

-----

## 📜 License

This project is licensed under the **MIT License** — see the <LICENSE> file for details.

-----

<p align="center">
  Made with ❤️ by <a href="https://github.com/ajrahar">Miftahul Fazi Raharja</a> · AMIKOM Yogyakarta University · 2026
</p>
