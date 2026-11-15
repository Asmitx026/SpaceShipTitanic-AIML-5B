# SpaceShip Titanic Passenger Transportation Prediction
A college coursework Machine Learning project that demonstrates the usage of various traditional ML models to predict the sruvival of passengers in a spaceship. This project explores passenger data, engineers meaningful features, and compares traditional machine learning models with a feed-forward neural network to predict whether a passenger was transported to another dimension.

The analysis and experiments are documented in [`spaceship_titanic.ipynb`](spaceship_titanic.ipynb) notebook file.

## Project Collaborators

This project is a group work prepared by 4 members:
- [Asmit Das Choudhury](https://github.com/Asmitx026)
- [Shuvradeep Patra](https://github.com/ShuvradeepPatra)
- [Barsha Saha](https://github.com/lilo006)
- [Sk Masudur Afridi](https://github.com/sk-afridi)

## Learning Outcomes

This project is divided into the following steps:

- Exploratory Data Analysis
- Data Cleaning & Preprocessing
- Feature Engineering
- Categorical (String Data) Encoding
- Feature Scaling (Numeric Data)
- Machine Learning pipelines
- Model Training
- Model Evaluation
- Metric-Based Model Comparison

## Project Highlights

- Cleaned and prepared the Spaceship Titanic training and test datasets.
- Extracted and Combined meaningful datas and discarding unneeded information (Feature Engineering)
  - Extracted `FamSize` from `PassengerId`.
  - Extracted `Cabin_Deck` and `Cabin_Side` from `Cabin`.
  - Combined all of the fve onboard expense columns into a single `Spendings` feature.
- Handled missing values, categorical variables, and numeric scaling.
- Compared Logistic Regression, K-Nearest Neighbours, Random Forest, and a feed-forward neural network.
- Evaluated models using accuracy, precision, recall, and F1 score.

## Dataset

The dataset is from the [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic).

The repository includes the following datasets:

- `datasets/train.csv` - Original labelled training data
- `datasets/test.csv` - Competition test data
- `datasets/clean-train.csv` - Cleaned training data used during analysis

Important source features include:

- `HomePlanet`
- `Destination`
- `CryoSleep`
- `VIP`
- `Age`
- `Cabin`
- `RoomService`
- `FoodCourt`
- `ShoppingMall`
- `Spa`
- `VRDeck`
- `PassengerId`

The prediction target is `Transported`, a binary value indicating whether the passenger was transported.

## Feature Engineering

The notebook transforms the raw data into model-ready features:

| Engineered Feature | Description |
| --- | --- |
| `FamSize` | Group or family size extracted from `PassengerId` |
| `Cabin_Deck` | Deck letter extracted from `Cabin` |
| `Cabin_Side` | Port or starboard side extracted from `Cabin` |
| `Spendings` | Sum of the five onboard spending columns; set to zero for cryosleeping passengers |

## Preprocessing Pipeline

The notebook uses a stratified 80/20 train-validation split with `random_state=42`.

- Boolean features are passed through unchanged.
- Categorical features use most-frequent imputation (`SimplewImputer`) followed by one-hot encoding.
- Numeric features use `KNNImputer` followed by `StandardScaler`.
- The same preprocessing pipeline is reused by the scikit-learn models to reduce inconsistent transformations and data leakage risk.

## Model Results

The following results were obtained on the notebook's validation set of 1,739 passengers:

| Model | Accuracy | Precision | Recall | F1 Score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 72.28% | 77.59% | 63.24% | 69.69% |
| K-Nearest Neighbours (`k=3`) | 68.37% | 70.27% | 64.50% | 67.26% |
| Random Forest (`50 trees`) | 70.96% | 73.22% | **66.78%** | 69.85% |
| Feed-Forward Neural Network | **74.35%** | **81.80%** | 63.13% | **71.26%** |

- The feed-forward neural network achieved the highest accuracy, precision and F1 score in this experiment.
- The Logistic Regression model is the most efficient model among all, taking less time for training and prediction along with quite high eval values.

## Project Structure

```text
.
├── datasets/
│   ├── clean-train.csv
│   ├── test.csv
│   └── train.csv
├── requirements.txt
├── spaceship_titanic.ipynb
└── README.md
```

## Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow/Keras
- KaggleHub

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Asmitx026/SpaceShipTitanic-AIML-5B.git
cd SpaceShipTitanic-AIML-5B
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

Open `spaceship_titanic.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab:

```bash
jupyter notebook spaceship_titanic.ipynb
```

Run the notebook cells in order to reproduce the analysis and model evaluation.

The notebook downloads the Kaggle competition data through `kagglehub`. Configure Kaggle credentials before running the download cell, or use the CSV files already stored in the `datasets/` directory.
