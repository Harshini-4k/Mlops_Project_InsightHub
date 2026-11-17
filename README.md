# InsightHub
## Customer Segmentation Pipeline Using MLOps Concepts

InsightHub groups customers using K-Means based on age, annual income, and spending score. The dashboard displays customer groups, charts, model results, and MLflow/DVC project information.

### Run in PyCharm (Windows)

1. Extract this ZIP to a folder, for example `Documents\InsightHub`.
2. Open the extracted folder in PyCharm.
3. Open the PyCharm Terminal at the project root.
4. Create a virtual environment if needed:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
5. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Run the dashboard:
   ```powershell
   streamlit run app.py
   ```

If the dashboard says the customer results are missing, run:
```powershell
python run_pipeline.py
```
Then run `streamlit run app.py` again.

### MLflow
The training script logs K-Means parameters and the silhouette score to the local `mlflow.db` database. From the project root, you can view the MLflow UI with:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open the local URL shown in the terminal.

### DVC
The project includes `dvc.yaml` and `dvc.lock`. To run the stages, use:
```powershell
dvc repro
```
Only claim the pipeline is reproducible after `dvc repro` completes successfully in your environment.

### Notes
- The silhouette score is a clustering quality measure, not model accuracy.
- The dashboard uses saved customer segmentation output. Run the pipeline to refresh the results.
- Customer group names and suggested actions are simple descriptions based on group averages.
