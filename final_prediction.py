import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('data/test.csv')
scaler = StandardScaler()

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch'])
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

missing_values = data.isnull().sum()
print(missing_values)

# data.to_csv('data/test_cleaned.csv', index=False)

scaled_data = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled_data)

model = joblib.load('model/logistic_regression_model.pkl')

predictions = model.predict(scaled_data)

test = pd.read_csv('data/test.csv')
test['Survived'] = predictions
test.to_csv('data/final_test.csv')