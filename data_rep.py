import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Charger le jeu de données Iris
iris = load_iris(as_frame=True)
df = iris.frame
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

# Séparer les données en référence (entraînement) et production (courantes)
df_ref, df_curr = train_test_split(df, test_size=0.5, random_state=42)

# Sauvegarder les fichiers
df_ref.to_csv("reference.csv", index=False)
df_curr.to_csv("current.csv", index=False)

print("Fichiers reference.csv et current.csv créés.")