# feature selection and feature engineering

## feature importance
check the features you have and whether they are good for your model

### for categorical features
1. compare group mean and total mean
2. risk
$$risk = negativeOutcomeRateGroup / negativeOutcomeRateTotal $$
3. mutual information
```python
from sklearn.metrics import mutual_info_score

def calculate_mi(series):
    return mutual_info_score(series, df.y)

df_mi = df[categorical].apply(calculate_mi)
```
4. entropy
```python
from sklearn.metrics import log_loss
```
### for numerical features
1. correlation coefficient
```python
df[numerical].corrwith(y)
```

## feature engineering
### converting categorical to numerical features
1. one hot encoding
