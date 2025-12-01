import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset

# 1. Charger données Iris
iris = load_iris(as_frame=True)
df = iris.frame
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

# 2. Split en référence (train) et courant (test)
df_ref, df_curr = train_test_split(df, test_size=0.5, random_state=42)

# 3. Sauvegarder CSV
df_ref.to_csv("reference.csv", index=False)
df_curr.to_csv("current.csv", index=False)

# 4. Entraîner modèle RandomForest sur les données de référence
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

X_train = df_ref[feature_names]
y_train = df_ref['target']

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Sauvegarder modèle
joblib.dump(model, "model.joblib")

# 6. Charger données courantes et faire prédictions
current_data = pd.read_csv("current.csv")
current_data['prediction'] = model.predict(current_data[feature_names])

reference_data = pd.read_csv("reference.csv")
# Ajouter prédictions dans current_data
current_data['prediction'] = model.predict(current_data[feature_names])

# Ajouter aussi prédictions dans reference_data
reference_data['prediction'] = model.predict(reference_data[feature_names])


# 7. Créer dossier reports
os.makedirs("reports", exist_ok=True)

# 8. Générer rapport Evidently
model_performance_report = Report(metrics=[
    ClassificationPreset(),
])

model_performance_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=None
)

model_performance_report.save_html("reports/model_performance_report.html")
