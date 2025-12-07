import pandas as pd

data = pd.read_csv('data/train.csv')

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch'])
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Age'] = data['Age'].fillna(data['Age'].mean())

missing_values = data_cleaned.isnull().sum()
print(missing_values)


data_cleaned.to_csv('data/train_cleaned.csv', index=False)