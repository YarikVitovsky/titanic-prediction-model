import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

X_train = pd.read_csv('data/X_train.csv')
X_val = pd.read_csv('data/X_val.csv')
y_train = pd.read_csv('data/y_train.csv')
y_val = pd.read_csv('data/y_val.csv')

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train.values.ravel())

joblib.dump(model, 'model/logistic_regression_model.pkl')

y_pred = model.predict(X_val)

accuracy_score = accuracy_score(y_val, y_pred)
print(f"validation accuracy: {accuracy_score:.2f}")