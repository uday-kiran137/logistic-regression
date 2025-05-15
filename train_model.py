import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load and clean data
df = pd.read_csv("Titanic_train.csv")
df = df[['Pclass', 'Sex', 'Age', 'Survived']].dropna()
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # male=1, female=0

# Train
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
