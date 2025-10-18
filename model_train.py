import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv("dataset.csv")
features = ["Ease_of_Ordering", "Good_Taste", "Time_Saving", "Offers", "Age"]
target = "Will_Purchase"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Model trained successfully! Accuracy: {acc*100:.2f}%")

with open("food_dss_model.pkl", "wb") as f:
    pickle.dump(model, f)
