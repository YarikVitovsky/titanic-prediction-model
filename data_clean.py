import pandas as pd
from sklearn.preprocessing import StandardScaler 

df = pd.read_csv('data/train.csv')

# Drop columns with too many missing values or not useful
df.drop(['Cabin'], axis=1, inplace=True)

# Fill numeric columns with mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if df[col].isnull().any():
        df[col].fillna(round(df[col].mean()), inplace=True)

# Fill text columns with mode (most common value)
text_cols = df.select_dtypes(include=['object']).columns
for col in text_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# print(df['Embarked'].unique())

# Create FamilySize and IsAlone features before scaling
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Select numeric columns for scaling, excluding 'Survived'
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()  # Convert to a list of column names
numeric_cols = [col for col in numeric_cols if col not in ['Survived', 'FamilySize', 'IsAlone']]

# Scale numeric values
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Extract Title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

# Group rare titles into a single category
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# Encode Title as numeric
title_mapping = {title: idx for idx, title in enumerate(df['Title'].unique())}
df['Title'] = df['Title'].map(title_mapping)

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)









# Save the final cleaned and scaled data
df.to_csv('data/train_cleaned.csv', index=False)