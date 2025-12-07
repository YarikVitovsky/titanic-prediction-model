import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/train.csv')
scaler = StandardScaler()

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch'])
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Age'] = data['Age'].fillna(data['Age'].mean())

missing_values = data.isnull().sum()
print(missing_values)

scaled_data = scaler.fit_transform(data) #NumPy Object
data = pd.DataFrame(scaled_data, columns=data.columns) #converting back togit data frame

data.to_csv('data/train_cleaned.csv', index=False)