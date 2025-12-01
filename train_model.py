import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("artifacts", exist_ok=True)
df_ref = pd.read_csv("reference.csv")
X_train = df_ref.drop('target', axis=1)
y_train = df_ref['target']
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "artifacts/model.joblib")
print("Modèle entraîné et sauvegardé dans artifacts/model.joblib")