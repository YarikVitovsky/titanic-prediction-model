import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv('data/train_cleaned.csv')  # Replace with your cleaned dataset file

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Survival Rates by Gender
def plot_gender_survival(df):
    # Create gender labels based on normalized values
    df_copy = df.copy()
    df_copy['Gender'] = df_copy['Sex'].apply(lambda x: 'Female' if x > 0 else 'Male')
    
    gender_survival = df_copy.groupby('Gender')['Survived'].mean()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=gender_survival.index, y=gender_survival.values, palette=['lightblue', 'pink'])
    plt.title('Survival Rates by Gender', fontsize=16)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.xlabel('Gender', fontsize=12)
    plt.ylim(0, 1)
    for i, v in enumerate(gender_survival.values):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('images/gender_survival.png')
    plt.close()

# 2. Survival Rates by Class
def plot_class_survival(df):
    # Create class labels based on normalized values
    df_copy = df.copy()
    def map_class(x):
        if x < -1:
            return '1st Class'
        elif x < 0.5:
            return '3rd Class' 
        else:
            return '2nd Class'
    
    df_copy['Class'] = df_copy['Pclass'].apply(map_class)
    
    class_survival = df_copy.groupby('Class')['Survived'].mean()
    # Reorder to logical sequence
    class_order = ['1st Class', '2nd Class', '3rd Class']
    class_survival = class_survival.reindex([c for c in class_order if c in class_survival.index])
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_survival.index, y=class_survival.values, palette=['gold', 'silver', 'brown'])
    plt.title('Survival Rates by Passenger Class', fontsize=16)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.xlabel('Passenger Class', fontsize=12)
    plt.ylim(0, 1)
    for i, v in enumerate(class_survival.values):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('images/class_survival.png')
    plt.close()

# 3. Age Distribution and Survival
def plot_age_survival(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30, palette='coolwarm', alpha=0.7)
    plt.title('Age Distribution and Survival Rates', fontsize=16)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig('images/age_survival.png')
    plt.close()

# 4. Family Size and Survival
def plot_family_survival(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    family_survival = df.groupby('FamilySize')['Survived'].mean()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=family_survival.index, y=family_survival.values, marker='o', color='green')
    plt.title('Survival Rates by Family Size', fontsize=16, pad=20)  # Add padding to the title
    plt.ylabel('Survival Rate', fontsize=12)
    plt.xlabel('Family Size', fontsize=12)
    plt.ylim(0, 1)
    for i, v in enumerate(family_survival.values):
        plt.text(family_survival.index[i], v + 0.02, f'{v:.1%}', ha='center', fontsize=10)
    plt.tight_layout()  # Ensure proper spacing
    plt.savefig('images/family_survival.png')
    plt.close()

# Generate all visualizations
plot_gender_survival(df)
plot_class_survival(df)
plot_age_survival(df)
plot_family_survival(df)

print("Visualizations generated and saved in the 'images' folder.")