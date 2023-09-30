## train, validate and test

```python
n = len(df)

n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

np.random.seed(2)
idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train + n_val].copy()
df_test = df_shuffled.iloc[n_train + n_val:].copy()
```

or 


```python
from sklearn.model_selection import train_test_split

seed = 123
df_train_valid, df_test = train_test_split(df, test_size=0.2, random_state=seed)
df_train, df_valid = train_test_split(df_train_valid, test_size=0.25, random_state=seed)
```

## separating inputs and outputs
```python
y_train = df_train.y
y_valid = df_valid.y
y_test = df_test.y

df_train.drop(columns=['y'], inplace = True)
df_valid.drop(columns=['y'], inplace = True)
df_test.drop(columns=['y'], inplace = True)
```