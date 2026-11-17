
import os
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlflow.tracking import MlflowClient

BASE = Path(__file__).resolve().parent
RAW = BASE / "data/raw/customer_data.csv"
SEG = BASE / "data/processed/customer_segments.csv"
FEAT = BASE / "data/processed/segmentation_data.csv"
MODEL = BASE / "models/kmeans_model.pkl"
DB = BASE / "mlflow.db"

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
PALETTE = ["#9B75FF", "#EC65C5", "#43C7E8", "#F7B955", "#51D6AD"]

st.set_page_config(
    page_title="InsightHub | Customer Analytics",
    page_icon="◈",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:#0d1019;color:#eef2ff}
[data-testid="stHeader"]{background:rgba(13,16,25,.95)}
[data-testid="stSidebar"]{background:#171a27;border-right:1px solid #292d40}
[data-testid="stSidebar"] *{color:#d8dced}
.block-container{padding-top:1.5rem;max-width:1600px}
.hero h1{font-size:34px;font-weight:800;margin:0;color:#f6f5ff;letter-spacing:-.7px}
.hero p,.muted{color:#98a2bb;font-size:13px}
.card{background:linear-gradient(145deg,#191d2b,#151824);border:1px solid #292d40;border-radius:16px;padding:17px 19px;min-height:112px}
.klabel{font-size:11px;color:#9da8c1;font-weight:700;letter-spacing:.6px}
.kvalue{font-size:29px;color:#f5f3ff;font-weight:800;margin-top:8px}
.knote{font-size:11px;color:#8f9ab4;margin-top:7px}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid #9b75ff;background:linear-gradient(100deg,#8567f7,#d957bc);color:white;font-weight:700}
[data-testid="stDataFrame"]{border:1px solid #2a3044;border-radius:12px;overflow:hidden}
hr{border-color:#2b3044}
</style>
""", unsafe_allow_html=True)


def metric(label, value, note):
    st.markdown(
        f"<div class='card'><div class='klabel'>{label}</div>"
        f"<div class='kvalue'>{value}</div>"
        f"<div class='knote'>{note}</div></div>",
        unsafe_allow_html=True
    )


def theme(fig, height=330):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=45, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c8cee0", size=11),
        colorway=PALETTE,
        legend=dict(orientation="h", y=1.03, x=0)
    )
    fig.update_xaxes(gridcolor="#2b3044")
    fig.update_yaxes(gridcolor="#2b3044")
    return fig


def get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def get_mlflow_client():
    """
    Streamlit Cloud: connect to DagsHub using Streamlit Secrets.
    Local computer: use the existing mlflow.db as a fallback.
    """

    tracking_uri = get_secret("MLFLOW_TRACKING_URI", "")
    username = get_secret("MLFLOW_TRACKING_USERNAME", "")
    password = get_secret("MLFLOW_TRACKING_PASSWORD", "")

    if tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

        if username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username

        if password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        client = MlflowClient(tracking_uri=tracking_uri)
        return client, "DagsHub"

    if DB.exists():
        local_uri = "sqlite:///" + DB.as_posix()
        client = MlflowClient(tracking_uri=local_uri)
        return client, "Local MLflow database"

    return None, None


def get_mlflow_runs():
    client, connection_name = get_mlflow_client()

    if client is None:
        return None, None, "No MLflow connection is configured."

    experiments = client.search_experiments()

    if not experiments:
        return pd.DataFrame(), connection_name, None

    experiment_names = {
        exp.experiment_id: exp.name
        for exp in experiments
    }

    experiment_ids = list(experiment_names.keys())

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        order_by=["attributes.start_time DESC"],
        max_results=10
    )

    rows = []

    for run in runs:
        start_time = run.info.start_time

        if start_time:
            started = pd.to_datetime(
                start_time,
                unit="ms"
            ).strftime("%d-%m-%Y %I:%M %p")
        else:
            started = "N/A"

        run_name = run.data.tags.get(
            "mlflow.runName",
            "Unnamed run"
        )

        metrics_text = ", ".join(
            f"{name}: {value:.4f}"
            for name, value in run.data.metrics.items()
        )

        rows.append({
            "Run name": run_name,
            "Experiment": experiment_names.get(
                run.info.experiment_id,
                "Unknown"
            ),
            "Status": run.info.status,
            "Started": started,
            "Run ID": run.info.run_id,
            "Metrics": metrics_text if metrics_text else "No metrics"
        })

    return pd.DataFrame(rows), connection_name, None


@st.cache_data
def read_data():
    raw = pd.read_csv(RAW) if RAW.exists() else pd.DataFrame()
    seg = pd.read_csv(SEG) if SEG.exists() else pd.DataFrame()

    if not seg.empty and all(c in seg.columns for c in FEATURES + ["Cluster"]):
        if len(raw) == len(seg) and "CustomerID" in raw:
            seg = seg.copy()
            seg["CustomerID"] = raw["CustomerID"].values
            if "Gender" in raw:
                seg["Gender"] = raw["Gender"].values

    return raw, seg


raw, df = read_data()

if not df.empty and all(c in df.columns for c in FEATURES + ["Cluster"]):
    counts = df["Cluster"].value_counts().sort_index()
    profiles = df.groupby("Cluster")[FEATURES].mean().round(1)

    try:
        score = (
            silhouette_score(df[FEATURES], df["Cluster"])
            if df["Cluster"].nunique() > 1
            else None
        )
    except Exception:
        score = None
else:
    counts = pd.Series(dtype=int)
    profiles = pd.DataFrame()
    score = None


with st.sidebar:
    st.markdown(
        "<h2 style='margin-bottom:0'>✦ "
        "<span style='color:#b69aff'>InsightHub</span></h2>"
        "<div class='muted'>CUSTOMER ANALYTICS</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Customer Explorer",
            "Upload Customer Data",
            "Segment Intelligence",
            "Model Results",
            "MLOps Tracking"
        ]
    )

    st.markdown("---")
    st.caption("Customer Segmentation Pipeline")
    st.caption("Python · K-Means · MLflow · DVC")


# --------------------------------------------------
# 1. OVERVIEW
# --------------------------------------------------

if page == "Overview":

    if df.empty:
        st.warning(
            "Saved customer results are not available. "
            "Please run the project pipeline or use Upload Customer Data."
        )
        st.stop()

    st.markdown(
        "<div class='hero'><h1>Customer intelligence</h1>"
        "<p>Understand customer behaviour across different groups.</p></div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric("TOTAL CUSTOMERS", f"{len(df):,}", "Customers in the dataset")

    with c2:
        metric("CUSTOMER GROUPS", str(df.Cluster.nunique()), "Created using K-Means")

    with c3:
        metric(
            "SILHOUETTE SCORE",
            f"{score:.4f}" if score is not None else "N/A",
            "Group separation score"
        )

    with c4:
        metric(
            "MISSING VALUES",
            str(int(raw.isna().sum().sum())) if not raw.empty else "N/A",
            "Checked in raw data"
        )

    st.write("")

    left, right = st.columns([1.65, 1])

    with left:
        st.subheader("Income vs spending")
        st.caption("Each dot represents one customer. Colours represent groups.")

        fig = px.scatter(
            df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color=df["Cluster"].astype(str),
            color_discrete_sequence=PALETTE,
            hover_data=[
                c for c in ["CustomerID", "Age", "Gender"] if c in df
            ],
            labels={"color": "Group"}
        )

        st.plotly_chart(theme(fig, 380), use_container_width=True)

    with right:
        st.subheader("Group distribution")

        fig = px.pie(
            values=counts.values,
            names=[f"Group {i}" for i in counts.index],
            hole=.66,
            color_discrete_sequence=PALETTE
        )

        fig.update_traces(
            textinfo="percent",
            marker=dict(line=dict(color="#171b29", width=3))
        )

        st.plotly_chart(theme(fig, 380), use_container_width=True)

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Customers in each group")

        bdf = counts.rename("Customers").reset_index()
        bdf.columns = ["Cluster", "Customers"]
        bdf["Group"] = bdf.Cluster.map(lambda x: f"Group {x}")

        fig = px.bar(
            bdf,
            x="Group",
            y="Customers",
            color="Group",
            text="Customers",
            color_discrete_sequence=PALETTE
        )

        fig.update_traces(textposition="outside")
        st.plotly_chart(theme(fig, 280), use_container_width=True)

    with right:
        st.subheader("Quick insights")

        st.markdown(
            f"**✦ Group {int(counts.idxmax())}** "
            f"has the most customers ({int(counts.max())})."
        )

        st.markdown(
            f"**↗ Group {int(profiles['Spending Score (1-100)'].idxmax())}** "
            "has the highest average spending score."
        )

        st.markdown(
            f"**◇ Group {int(profiles['Annual Income (k$)'].idxmax())}** "
            "has the highest average income."
        )

        st.caption("These points are calculated from the saved results.")

    st.download_button(
        "⬇ Export customer results",
        df.to_csv(index=False).encode("utf-8"),
        "customer_segments.csv",
        "text/csv"
    )


# --------------------------------------------------
# 2. CUSTOMER EXPLORER
# --------------------------------------------------

elif page == "Customer Explorer":

    if df.empty:
        st.warning("Saved customer results are not available. Use Upload Customer Data.")
        st.stop()

    st.markdown(
        "<div class='hero'><h1>Customer explorer</h1>"
        "<p>Search customers and view their assigned group.</p></div>",
        unsafe_allow_html=True
    )

    a, b, c = st.columns([1, 1, 1.4])

    with a:
        search = st.text_input("Search customer ID", placeholder="e.g. 12")

    with b:
        group = st.selectbox(
            "Customer group",
            ["All groups"] + [
                f"Group {int(x)}" for x in sorted(df.Cluster.unique())
            ]
        )

    with c:
        spending = st.slider("Spending score", 0, 100, (0, 100))

    view = df.copy()

    if search.strip() and "CustomerID" in view:
        view = view[
            view.CustomerID.astype(str).str.contains(
                search.strip(), case=False, na=False
            )
        ]

    if group != "All groups":
        view = view[view.Cluster == int(group.split()[-1])]

    view = view[
        view["Spending Score (1-100)"].between(*spending)
    ]

    st.caption(f"{len(view)} customer(s) match your filters")
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button(
        "Download filtered data",
        view.to_csv(index=False).encode("utf-8"),
        "filtered_customers.csv",
        "text/csv"
    )

    if not view.empty and "CustomerID" in view:

        st.subheader("Customer details")

        cid = st.selectbox(
            "Choose customer ID",
            view.CustomerID.astype(str).tolist()
        )

        row = view[view.CustomerID.astype(str) == cid].iloc[0]

        x, y, z = st.columns(3)

        with x:
            metric("AGE", str(int(row.Age)), "Years")

        with y:
            metric(
                "ANNUAL INCOME",
                f"{row['Annual Income (k$)']:.0f}k",
                "Dataset income unit"
            )

        with z:
            metric(
                "SPENDING SCORE",
                f"{row['Spending Score (1-100)']:.0f}/100",
                f"Group {int(row.Cluster)}"
            )


# --------------------------------------------------
# 3. UPLOAD CUSTOMER DATA
# --------------------------------------------------

elif page == "Upload Customer Data":

    st.markdown(
        "<div class='hero'><h1>Upload customer data</h1>"
        "<p>Upload your own CSV file and discover customer groups using K-Means.</p></div>",
        unsafe_allow_html=True
    )

    st.subheader("1. Upload your CSV file")

    st.info(
        "Your CSV must contain these three columns: "
        "**Age**, **Annual Income (k$)** and "
        "**Spending Score (1-100)**."
    )

    uploaded_file = st.file_uploader(
        "Choose your customer CSV file",
        type=["csv"],
        help="Upload a CSV file containing customer information."
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.success(f"Uploaded successfully: {uploaded_file.name}")

            st.subheader("2. Preview your data")
            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True,
                hide_index=True
            )

            missing_columns = [
                col for col in FEATURES if col not in uploaded_df.columns
            ]

            if missing_columns:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            elif len(uploaded_df) < 3:
                st.error(
                    "Please upload a dataset containing at least 3 customers."
                )

            else:

                for col in FEATURES:
                    uploaded_df[col] = pd.to_numeric(
                        uploaded_df[col],
                        errors="coerce"
                    )

                if uploaded_df[FEATURES].isna().any().any():
                    st.error(
                        "Your required columns contain missing or "
                        "non-numeric values. Please clean your CSV and upload again."
                    )

                elif np.isinf(uploaded_df[FEATURES].to_numpy()).any():
                    st.error(
                        "Your data contains infinite values. "
                        "Please correct them and upload again."
                    )

                else:

                    st.subheader("3. Choose the number of customer groups")

                    max_clusters = min(10, len(uploaded_df) - 1)

                    n_clusters = st.selectbox(
                        "Number of groups (K)",
                        options=list(range(2, max_clusters + 1)),
                        index=min(3, max_clusters - 2),
                        key="upload_cluster_count"
                    )

                    if st.button(
                        "Run Customer Segmentation",
                        type="primary"
                    ):

                        try:
                            model = KMeans(
                                n_clusters=n_clusters,
                                random_state=42,
                                n_init=10
                            )

                            result_df = uploaded_df.copy()

                            result_df["Cluster"] = model.fit_predict(
                                result_df[FEATURES]
                            )

                            if result_df["Cluster"].nunique() < 2:
                                st.error(
                                    "K-Means could not create separate groups. "
                                    "Try different data or another K value."
                                )

                            else:

                                result_score = silhouette_score(
                                    result_df[FEATURES],
                                    result_df["Cluster"]
                                )

                                st.session_state["upload_result"] = result_df
                                st.session_state["upload_score"] = result_score
                                st.session_state["upload_filename"] = uploaded_file.name
                                st.session_state["upload_k"] = n_clusters

                        except Exception as e:
                            st.error(f"Segmentation failed: {e}")

                    if (
                        "upload_result" in st.session_state
                        and st.session_state.get("upload_filename") == uploaded_file.name
                    ):

                        result_df = st.session_state["upload_result"]
                        result_score = st.session_state["upload_score"]

                        st.success("Customer segmentation completed!")

                        st.subheader("4. Segmentation results")

                        upload_counts = (
                            result_df["Cluster"].value_counts().sort_index()
                        )

                        c1, c2, c3 = st.columns(3)

                        with c1:
                            metric(
                                "TOTAL CUSTOMERS",
                                str(len(result_df)),
                                "Uploaded records"
                            )

                        with c2:
                            metric(
                                "CUSTOMER GROUPS",
                                str(result_df["Cluster"].nunique()),
                                "K-Means groups"
                            )

                        with c3:
                            metric(
                                "SILHOUETTE SCORE",
                                f"{result_score:.4f}",
                                "Group separation score"
                            )

                        st.subheader("Customer group distribution")

                        fig = px.bar(
                            x=[f"Group {i}" for i in upload_counts.index],
                            y=upload_counts.values,
                            color=[str(i) for i in upload_counts.index],
                            color_discrete_sequence=PALETTE,
                            labels={
                                "x": "Customer group",
                                "y": "Number of customers",
                                "color": "Group"
                            }
                        )

                        st.plotly_chart(
                            theme(fig),
                            use_container_width=True
                        )

                        st.subheader("Income vs spending")

                        hover_columns = [
                            c for c in ["CustomerID", "Age", "Gender"]
                            if c in result_df.columns
                        ]

                        fig = px.scatter(
                            result_df,
                            x="Annual Income (k$)",
                            y="Spending Score (1-100)",
                            color=result_df["Cluster"].astype(str),
                            color_discrete_sequence=PALETTE,
                            hover_data=hover_columns,
                            labels={"color": "Customer group"}
                        )

                        st.plotly_chart(
                            theme(fig, 380),
                            use_container_width=True
                        )

                        st.subheader("5. Final customer segments")

                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "⬇ Download segmented customer data",
                            result_df.to_csv(index=False).encode("utf-8"),
                            file_name="uploaded_customer_segments.csv",
                            mime="text/csv"
                        )

        except Exception as e:
            st.error(f"Unable to process this CSV file: {e}")


# --------------------------------------------------
# 4. SEGMENT INTELLIGENCE
# --------------------------------------------------

elif page == "Segment Intelligence":

    if df.empty:
        st.warning("Saved customer results are not available. Use Upload Customer Data.")
        st.stop()

    st.markdown(
        "<div class='hero'><h1>Segment intelligence</h1>"
        "<p>Compare the average age, income and spending score for each group.</p></div>",
        unsafe_allow_html=True
    )

    table = profiles.copy()
    table["Customers"] = counts
    table.index = [f"Group {int(i)}" for i in table.index]

    st.dataframe(
        table.reset_index(names="Customer group"),
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            table.reset_index(names="Group"),
            x="Group",
            y="Annual Income (k$)",
            color="Group",
            color_discrete_sequence=PALETTE,
            title="Average annual income"
        )

        st.plotly_chart(theme(fig), use_container_width=True)

    with right:
        fig = px.bar(
            table.reset_index(names="Group"),
            x="Group",
            y="Spending Score (1-100)",
            color="Group",
            color_discrete_sequence=PALETTE,
            title="Average spending score"
        )

        st.plotly_chart(theme(fig), use_container_width=True)

    st.subheader("Possible actions")

    for idx, row in profiles.iterrows():

        st.markdown(
            f"**Group {int(idx)}** · {int(counts.loc[idx])} customers"
        )

        if (
            row["Spending Score (1-100)"]
            >= profiles["Spending Score (1-100)"].median()
        ):
            st.write("Possible idea: test loyalty rewards or personal offers.")
        else:
            st.write("Possible idea: test value offers or product bundles.")

    st.caption(
        "Group names and actions are suggestions based on averages, "
        "not guaranteed business outcomes."
    )


# --------------------------------------------------
# 5. MODEL RESULTS
# --------------------------------------------------

elif page == "Model Results":

    if df.empty:
        st.warning("Saved customer results are not available. Use Upload Customer Data.")
        st.stop()

    st.markdown(
        "<div class='hero'><h1>Model results</h1>"
        "<p>Check the K-Means results and compare possible group counts.</p></div>",
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        metric("ALGORITHM", "K-Means", "Unsupervised learning")

    with b:
        metric("GROUPS", str(df.Cluster.nunique()), "Current saved results")

    with c:
        metric(
            "SILHOUETTE",
            f"{score:.4f}" if score is not None else "N/A",
            "Not model accuracy"
        )

    if FEAT.exists():

        features = pd.read_csv(FEAT)

        if all(x in features for x in FEATURES):

            if len(features) >= 3:

                ks = list(range(2, min(9, len(features) - 1) + 1))

                inertias = [
                    KMeans(
                        n_clusters=k,
                        random_state=42,
                        n_init=10
                    ).fit(features[FEATURES]).inertia_
                    for k in ks
                ]

                fig = px.line(
                    x=ks,
                    y=inertias,
                    markers=True,
                    labels={
                        "x": "Number of groups (K)",
                        "y": "Inertia"
                    },
                    title="Elbow method"
                )

                fig.update_traces(
                    line=dict(color="#9B75FF", width=3),
                    marker=dict(size=9)
                )

                st.plotly_chart(theme(fig), use_container_width=True)

    st.info(
        "The silhouette score tells us how well-separated the groups are. "
        "It is not accuracy."
    )

    st.write(
        "Saved model:",
        "Available" if MODEL.exists() else "Not found"
    )


# --------------------------------------------------
# 6. MLOPS TRACKING
# --------------------------------------------------

elif page == "MLOps Tracking":

    st.markdown(
        "<div class='hero'><h1>MLOps tracking</h1>"
        "<p>See the steps and files used in this project.</p></div>",
        unsafe_allow_html=True
    )

    steps = [
        ("01", "Data validation", "Check missing values, duplicates and data types."),
        ("02", "Preprocessing", "Prepare the customer data."),
        ("03", "Feature selection", "Use age, income and spending score."),
        ("04", "Model training", "Train K-Means and save group labels."),
        ("05", "Evaluation", "Calculate the silhouette score."),
        ("06", "Experiment tracking", "Record settings and scores with MLflow.")
    ]

    for n, title, desc in steps:

        st.markdown(
            f"<div class='card' style='min-height:0;margin-bottom:9px;"
            f"display:flex;gap:15px'><b style='font-size:20px;color:#b69aff'>"
            f"{n}</b><div><b>{title}</b><div class='knote'>{desc}</div>"
            f"</div></div>",
            unsafe_allow_html=True
        )

    st.subheader("Project files")

    items = {
        "Raw dataset": RAW,
        "Segment results": SEG,
        "Saved model": MODEL,
        "DVC pipeline": BASE / "dvc.yaml",
        "DVC lock": BASE / "dvc.lock"
    }

    file_status = [
        {
            "File": name,
            "Status": "Available" if path.exists() else "Not found"
        }
        for name, path in items.items()
    ]

    if get_secret("MLFLOW_TRACKING_URI", ""):
        file_status.append({
            "File": "MLflow tracking",
            "Status": "Connected to DagsHub"
        })
    else:
        file_status.append({
            "File": "Local MLflow database",
            "Status": "Available" if DB.exists() else "Not found"
        })

    st.dataframe(
        pd.DataFrame(file_status),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Recent MLflow runs")

    try:
        runs_df, connection_name, error = get_mlflow_runs()

        if error:
            st.warning(error)
            st.info(
                "Add your DagsHub connection details under "
                "Streamlit Cloud Settings → Secrets."
            )

        else:
            st.success(f"Connected to {connection_name}")

            if runs_df.empty:
                st.info("No MLflow runs were found in this experiment.")
            else:
                st.caption(f"Showing the latest {len(runs_df)} MLflow runs.")

                st.dataframe(
                    runs_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download MLflow run history",
                    runs_df.to_csv(index=False).encode("utf-8"),
                    "mlflow_runs.csv",
                    "text/csv"
                )

    except Exception as e:
        st.error(f"Unable to connect to MLflow: {e}")
        st.caption(
            "Check your DagsHub URL and credentials in Streamlit Secrets."
        )

    st.caption(
        "MLflow tracks experiments. DVC tracks data and pipeline steps."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    "<div style='text-align:center;color:#77819b;font-size:11px;"
    "padding:24px 0 8px'>INSIGHTHUB · CUSTOMER SEGMENTATION "
    "PIPELINE USING MLOPS CONCEPTS</div>",
    unsafe_allow_html=True
)
