from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier         
from sklearn.neighbors import KNeighborsClassifier      
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("cleaned_data.csv")
print(df.head())

X = df[['pclass', 'age', 'sibsp', 'parch', 'fare']] 
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

model2 = KNeighborsClassifier()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

scaler = StandardScaler()

Xtrains = scaler.fit_transform(X_train)
Xtests = scaler.transform(X_test)

model_knn_scaled = KNeighborsClassifier()
model_knn_scaled.fit(Xtrains, y_train)

ypreds = model_knn_scaled.predict(Xtests)
accuracy_scaled = accuracy_score(y_test, ypreds)

print(f"KNN Accuracy after Scaling: {accuracy_scaled}")