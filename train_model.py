import pandas as pd
import os

data = []
gestures = []

base_dir = 'gesture_data'
for gesture_name in os.listdir(base_dir):
    file_path = os.path.join(base_dir, gesture_name, f"{gesture_name}.csv")
    df = pd.read_csv(file_path)
    data.append(df)
    gestures.append(gesture_name)

full_data = pd.concat(data)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = full_data.drop('label', axis=1)
y = full_data['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import joblib

joblib.dump(model, 'gesture_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
