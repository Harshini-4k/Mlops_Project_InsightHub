import subprocess
import sys
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent


def run_step(title, script):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"\nRunning: python {script}\n")

    result = subprocess.run(
        [sys.executable, str(BASE_PATH / script)],
        cwd=BASE_PATH
    )

    if result.returncode != 0:
        print(f"\n❌ {script} failed!")
        sys.exit(result.returncode)


print("\n" + "=" * 60)
print("          INSIGHTHUB MLOPS PIPELINE")
print("=" * 60)

# 1. Data Validation
run_step(
    "STEP 1 — DATA VALIDATION",
    "src/data/validate_data.py"
)

# 2. Preprocessing
run_step(
    "STEP 2 — DATA PREPROCESSING",
    "src/data/preprocess_data.py"
)

# 3. Feature Engineering
run_step(
    "STEP 3 — FEATURE ENGINEERING",
    "src/features/feature_engineering.py"
)

# 4. Model Training + MLflow
run_step(
    "STEP 4 — K-MEANS TRAINING + MLFLOW",
    "src/models/train_kmeans.py"
)

# 5. Evaluation
run_step(
    "STEP 5 — CLUSTER EVALUATION",
    "src/models/evaluate_clusters.py"
)

# 6. Monitoring
run_step(
    "STEP 6 — DATA MONITORING",
    "src/monitoring/data_drift.py"
)

print("\n" + "=" * 60)
print("       INSIGHTHUB PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nMLOps Components:")
print("✓ Data Validation")
print("✓ Data Preprocessing")
print("✓ Feature Engineering")
print("✓ K-Means Model")
print("✓ MLflow Experiment Tracking")
print("✓ DVC Pipeline")
print("✓ Data Monitoring")
print("✓ GitHub Version Control")
print()