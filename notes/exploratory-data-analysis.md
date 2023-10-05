# Exploratory data analysis

in EDA you should look at:

- The distribution of the target variable
- The features in this dataset and their type
- The quality of the data
- The number of missing values
- The distribution of values in these features
- relation of each feature with the target variable

## imports

```python
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
```

## check for data types

```python
df.columns
df.info()
df.dtypes
df.head().T
df.describe()
```

## change data types
```python
pd.to_numeric(df.TotalCharges, errors='coerce')
df.churn = (df.churn == 'yes').astype(int)
```

## normalizing column names and string values

```python
# normalizing column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# normalizing string values
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
```

## target variable analysis

```python
sns.histplot(...)
log_y = np.log1p(df.y.values)
```

## Checking for missing values

```python
df.isnull().sum()
df.finllna(value)
```