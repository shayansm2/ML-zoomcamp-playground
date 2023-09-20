# Exploratory data analysis

in EDA you should look at:

- The distribution of the target variable
- The features in this dataset
- The distribution of values in these features
- The quality of the data
- The number of missing values

## imports

```python
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
```

## normalizing column names and string values

```python
# normalizing column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

#normalizing string values
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
```

## target variable analysis
```python
sns.histplot(...)
log_y = np.log1p(df.y)
```