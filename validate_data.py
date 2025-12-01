import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset

os.makedirs("reports", exist_ok=True)

# Charger les données CSV
reference_data = pd.read_csv("reference.csv")
current_data = pd.read_csv("current.csv")

# Rapport de Qualité
data_quality_report = Report(metrics=[DataQualityPreset()])
data_quality_report.run(reference_data=reference_data, current_data=current_data, column_mapping=None)
data_quality_report.save_html("reports/data_quality_report.html")

# Test Suite pour CI
data_stability_test = TestSuite(tests=[DataStabilityTestPreset()])
data_stability_test.run(reference_data=reference_data, current_data=current_data, column_mapping=None)
data_stability_test.save_html("reports/data_stability_test_suite.html")

# Logique d'échec du pipeline
if not data_stability_test.as_dict()['summary']['all_passed']:
    print("\n[ERREUR CI] La Test Suite de stabilité des données a échoué.")
    exit(1)
else:
    print("\n[CI OK] La Test Suite de stabilité des données a réussi.")
    exit(0)
