# kaggle-house-prices-regression1
price prediction using regression.  Utilizes the Kaggle House Prices dataset.
# House Price Prediction - Data Exploration and Visualization

This project explores the "House Prices - Advanced Regression Techniques" dataset from Kaggle, focusing on data loading, initial exploration, and visualization.  The primary goal is to understand the dataset's characteristics and identify potential features for predicting house prices.

## Project Overview

This notebook (`house_price_exploration.ipynb` or similar) performs the following key tasks:

* Loads the training dataset using Pandas.
* Displays the first few rows of the data.
* Examines the columns and their data types.
* Calculates and displays descriptive statistics for the target variable (`SalePrice`).
* Visualizes the distribution of the target variable using Seaborn's `distplot`.
* Calculates and prints the skewness of the target variable.

## Dataset

The dataset used in this project is the "House Prices - Advanced Regression Techniques" dataset available on Kaggle.  You can download it from the competition page: [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

**Note:** Ensure that the `train.csv` file is placed in the correct directory (specified by the file path in the notebook) before running the code.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn

## Code Highlights

```python
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
import numpy as np

from sklearn.preprocessing import StandardScaler
from scipy import stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import joblib

import warnings

warnings.filterwarnings('ignore')
%matplotlib inline

train = pd.read_csv(r"D:\data science\project kaggle\home-data-for-ml-course\train.csv")
train.head()  # Display the first few rows

train.columns  # Show the columns in the dataset

train['SalePrice'].describe()  # Descriptive statistics for SalePrice

sns.distplot(train['SalePrice']) # Distribution plot of SalePrice

print (f"skewness: {train['SalePrice'].skew()}") # Skewness of SalePrice
