import seaborn as sns
import pandas as pd

df = sns.load_dataset('titanic')

print(df.info())
print("_____________")
print(df.describe())
print("_____________")
print(df.isnull().sum())

df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0]) # لما تكون نسبة القيم الناقصة قليلة جداً

df['deck'] = df['deck'].astype(str)
df['deck'] = df['deck'].fillna('Unknown') # لما تكون نسبة القيم الناقصة كثيرة جداً
df['age'] = df['age'].fillna(df['age'].mean()) 

print(df[['embarked', 'deck', 'age']].isnull().sum())

print("_____________")

print(df['embarked'].nunique())
print(df['deck'].nunique())

print("_____________")

df = pd.get_dummies(df, columns=['deck'], drop_first=True)
print(df)

df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

target_column = 'survived'  

X = df.drop(columns=[target_column]) 

y = df[target_column]

print("_____________")
print(X.head())
print(y.head())

df.to_csv('cleaned_data.csv', index=False)
