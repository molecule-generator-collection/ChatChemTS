from flaml import AutoML
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title("ChatMolGen Model Builder Tool")
st.subheader("This tool support users to build prediction models using AutoML framwork, FLAML.")

if "use_auto_estimator" not in st.session_state:
    st.session_state.use_auto_estimator = True
if "use_auto_metric" not in st.session_state:
    st.session_state.use_auto_metric = True


df = None
uploaded_file = st.file_uploader("Upload CSV file.", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.header("Uploaded File Preview")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.header("FLAML Settings")

    st.subheader("Dataset Preparation")
    smiles_col_name = st.selectbox(
        "Select SMILES column name",
        options=df.columns,
        index=None,
    )
    label_col_name = st.selectbox(
        "Select label column name",
        options=df.columns,
        index=None,
    )
    test_size_ratio = st.number_input(
        "Set test dataset ratio",
        value=0.1,
    )

    st.subheader("Estimator Selection")
    st.checkbox('Use auto', value=True, key='use_auto_estimator')
    estimator_list = st.multiselect(
        "Select estimators used in FLAML's AutoML workflow",
        options=['xgboost', 'xgb_limitdepth', 'rf', 'lgbm', 'lrl1', 'lrl2', 'catboost', 'extra_tree', 'kneighbor'],
        disabled=st.session_state.use_auto_estimator,
    )
    if st.session_state.use_auto_estimator:
        estimator_list = 'auto'
    
    st.subheader("Task Selection")     
    task = st.selectbox(
        "Select task",
        options=['regression'],
        index=0,
    )

    st.subheader("Metric Selection")
    st.checkbox('Use auto', value=True, key='use_auto_metric')
    metric = st.selectbox(
        "Select metric",
        options=["r2", "rmse", "mae", "mse", "accuracy", "roc_auc", "roc_auc_ovr", "roc_auc_ovo", "log_loss", "mape", "f1", "micro_f1", "macro_f1", "ap"],
        index=None,
        disabled=st.session_state.use_auto_metric,
    )
    if st.session_state.use_auto_metric:
        metric = 'auto'

    st.subheader("Budget Time")
    time_budget = st.number_input(
        "Set a time budget in seconds\n(set -1 if you want to set no time limit)",
        value=60,
    )

    if st.button("Run AutoML", type='primary'):
        with st.spinner("AutoML started..."):
            automl = AutoML()

            df_x = df[smiles_col_name].apply(lambda x: Chem.MolFromSmiles(x))
            df_x = df_x.dropna().apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048))
            X = np.array(df_x.tolist())
            y = df[label_col_name]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)

            automl_settings = {
                "time_budget": time_budget,
                "task": task,
                "estimator_list": estimator_list,
                "metric": metric, 
                "log_file_name": "test.log",
            }
            automl.fit(X_train, y_train, **automl_settings)
        with st.spinner("Scatter plot created..."):
            fig, ax = plt.subplots(figsize=(4, 4))
            y_pred = automl.predict(X_test)
            correlation = np.corrcoef(y_pred, y_test)[0, 1]
            ax.scatter(y_pred, y_test, marker='o', s=20, c='dimgray', alpha=0.2)
            ax.text(1.0, 0.1, f"Corr. Coef.: {correlation:.2f}\nBest Estimator: {automl.model.estimator}", transform=ax.transAxes)
            ax.set_xlabel("Prediction Value")
            ax.set_ylabel("Actual Value")
            st.pyplot(fig, use_container_width=False)