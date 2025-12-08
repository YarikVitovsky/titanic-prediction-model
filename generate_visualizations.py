import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Titanic dataset
df = pd.read_csv('data/train.csv')  # Replace with your cleaned dataset file

# 1. Survival Rates by Fare
def plot_fare_survival(df):
    bins = [0, 10, 25, 50, 100, 600]
    labels = ["0 - 10£", "11 - 25£", "26 - 50£", "51 - 100£", "101£ +"]

    df_copy = df.copy()
    df_copy['FareGroup'] = pd.cut(df_copy['Fare'], bins=bins, labels=labels, right=False)

    fare_survival_rate = df_copy.groupby('FareGroup')['Survived'].mean()

    plt.figure(figsize=(10,6))
    sns.barplot(x=fare_survival_rate.index, y=fare_survival_rate.values, palette='coolwarm')

    plt.title('Survival Rate By Fare Group', fontsize = 16)
    plt.xlabel('Fare Groups', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.ylim(0, 1)

    for i, v in enumerate(fare_survival_rate.values):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('images/fare_survival_rate.png')
    plt.close()

# 2. Survival Rates by Gender
def plot_gender_survival(df):
    df_copy = df.copy()
    gender_survival_rate = df_copy.groupby('Sex')['Survived'].mean()

    plt.figure(figsize=(8,6))
    sns.barplot(x=gender_survival_rate.index, y=gender_survival_rate.values, palette='coolwarm')

    plt.title('Survival Rate By Gender')
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.ylim(0, 1)

    for i, v in enumerate(gender_survival_rate.values):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('images/gender_survival_rate.png')
    plt.close()

# 3. Survival Rates by Family Size
def plot_family_size_survival(df):
    bins = [1, 2, 4, 6, 8, 12]
    labels = ["1", "2-3", "4-5", "6-7", "8+"]

    df_copy = df.copy()
    df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
    df_copy['FamilySizeGroups'] = pd.cut(df_copy['FamilySize'], bins=bins, labels=labels, right=False)

    family_size_survival_rate = df_copy.groupby('FamilySizeGroups')['Survived'].mean()

    plt.figure(figsize=(10,6))
    sns.barplot(x=family_size_survival_rate.index, y=family_size_survival_rate.values, palette='coolwarm')

    plt.title('Survival Rate By Family Size', fontsize = 16)
    plt.xlabel('Family Size', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.ylim(0, 1)

    for i, v in enumerate(family_size_survival_rate.values):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('images/family_size_survival_rate.png')
    plt.close()

# 4. Survival Rates by Passanger Class
def plot_passanger_class_survival(df):
    bins = [1, 2, 3, 4]
    labels = ["First Class", "Second Class", "Third Class"]

    df_copy = df.copy()
    df_copy['ClassGroups'] = pd.cut(df_copy['Pclass'], bins=bins, labels=labels, right=False)

    passanger_class_survival_rate = df_copy.groupby('ClassGroups')['Survived'].mean()

    plt.figure(figsize=(10,6))
    sns.barplot(x=passanger_class_survival_rate.index, y=passanger_class_survival_rate.values, palette='coolwarm')

    plt.title('Survival Rate By passanger_class_survival_rate', fontsize = 16)
    plt.xlabel('Passanger Class', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.ylim(0, 1)

    for i, v in enumerate(passanger_class_survival_rate.values):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('images/passanger_class_survival_rate.png')
    plt.close()

# 5. Survival Rates by Age
def plot_age_survival(df):
    bins = [0, 16, 26, 36, 46, 56, 100]
    labels = ["0-15", "16-25", "26-35", "36-45", "46-55", "56+"]

    df_copy = df.copy()
    df_copy['AgeGroups'] = pd.cut(df_copy['Age'], bins=bins, labels=labels, right=False)

    age_survival_rate = df_copy.groupby('AgeGroups')['Survived'].mean()

    plt.figure(figsize=(10,6))
    sns.barplot(x=age_survival_rate.index, y=age_survival_rate.values, palette='coolwarm')

    plt.title('Survival Rate By age_survival_rate', fontsize = 16)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.ylim(0, 1)

    for i, v in enumerate(age_survival_rate.values):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('images/age_survival_rate.png')
    plt.close()


# Generate all visualizations
plot_fare_survival(df)
plot_gender_survival(df)
plot_family_size_survival(df)
plot_passanger_class_survival(df)
plot_age_survival(df)