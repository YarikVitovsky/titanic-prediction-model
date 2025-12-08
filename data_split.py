import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/train_cleaned.csv')

X = data.drop(columns=['Survived'])
y = data['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv('data/X_train.csv', index=False)
X_val.to_csv('data/X_val.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_val.to_csv('data/y_val.csv', index=False)