import os
import pickle

from chembl_webresource_client.settings import Settings
Settings.Instance().MAX_LIMIT = 1000
Settings.Instance().CACHING = False
from chembl_webresource_client.new_client import new_client
from flaml import AutoML
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.image("logo_dark.png")
st.title("ChatChemTS Model Builder Tool")
st.write("This tool support users to build prediction models using AutoML framwork, FLAML.")

if "use_auto_estimator" not in st.session_state:
    st.session_state.use_auto_estimator = True
if "use_auto_metric" not in st.session_state:
    st.session_state.use_auto_metric = True
if "use_dataset_from_csv" not in st.session_state:
    st.session_state.use_dataset_from_csv = False
if "use_dataset_from_uniprotid" not in st.session_state:
    st.session_state.use_dataset_from_uniprotid = False
if "df" not in st.session_state:
    st.session_state.df = None

def click_button(clicked_type):
    if clicked_type == "csv":
        if not st.session_state.use_dataset_from_csv:
            st.session_state.use_dataset_from_csv = not st.session_state.use_dataset_from_csv
            st.session_state.use_dataset_from_uniprotid = not st.session_state.use_dataset_from_csv
            st.session_state.df = None
    elif clicked_type == "uniprot":
        if not st.session_state.use_dataset_from_uniprotid:
            st.session_state.use_dataset_from_uniprotid = not st.session_state.use_dataset_from_uniprotid
            st.session_state.use_dataset_from_csv = not st.session_state.use_dataset_from_uniprotid
            st.session_state.df = None
    else:
        raise Exception

button_style = """
<style>
div.stButton > button:first-child {
    text-align: center;
    width: 400px;
}
div.stButton {
    text-align: center;
}
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)

st.subheader("Data source to build models")
col1, col2 = st.columns(2)
with col1:
    st.button('Upload CSV file', type='primary', on_click=click_button, kwargs=dict(clicked_type='csv'))
with col2:
    st.button('Fetch data from ChEMBL database', type='primary', on_click=click_button, kwargs=dict(clicked_type='uniprot'))

if st.session_state.use_dataset_from_csv:
    uploaded_file = st.file_uploader("Upload CSV file.", type="csv")
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
if st.session_state.use_dataset_from_uniprotid:
    uniprot_id = st.text_input("Enter UniProt ID of target protein")
    target_api = new_client.target
    molecule_api = new_client.molecule
    activity_api = new_client.activity

    # fetch target-related data
    target_results = pd.DataFrame(
        target_api.filter(
            target_components__accession=uniprot_id
        ).filter(target_type__in=["SINGLE PROTEIN", "PROTEIN COMPLEX"]))
    if not uniprot_id:
        st.stop()
    if target_results.empty:
        st.warning("ChEMBL database contains NO records for the provided UniProt ID. Try another ID.")
        st.stop()
    st.success("ChEMBL database contains records for the provided UniProt ID. Proceed to the next step.")

    st.subheader("Retrieved target information")
    st.dataframe(
        pd.concat([target_results.pref_name.rename("Preferred Name"), target_results.target_chembl_id.rename("ChEMBL ID")],
                  axis=1), hide_index=True
    )

    st.subheader("Deduplicate molecules by selecting representative pChEMBL values")
    duplicate_option = st.selectbox(
        'Select which duplicates (if any) to keep.',
        ('Keep Maximum Value', 'Keep Minimum Value', 'Do Nothing'),
    )

    st.subheader("Retain records by assay type.")
    include_assay_types = st.multiselect(
        "Select assay types to retain records. ref. https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions",
        options=["B", "F", "A", "T", "P", "U"],
        default=["B"]
    )
    
    st.subheader("Filter records containing the following texts in assay descriptions.")
    assay_exclude_text = st.text_input("Eenter words or substrings separated by commas", value='mutat,covalent,irreversible')
    assay_exclude_list = assay_exclude_text.split(',')
    assay_exclude_list = [t.lower() for t in assay_exclude_list if t]  # remove '' and lowercase texts.

    st.subheader("Filter records by activity type.")
    exclude_activity_types = st.multiselect(
        "Select activity types to exclude records from dataset",
        options=["IC50", "XC50", "EC50", "AC50", "Ki", "Kd", "Potency", "ED50"],
        default=[]
    )

    if st.button("Fetch & Clean Data", type='primary'):
        with st.spinner("Processing..."):
            # fetch activity-related data
            #df_activity = pd.read_csv("./chembl_example_uniprot_P00533_rmidx.csv", sep='\t')
            tmp_df_list = []
            for _, tr in target_results.iterrows():
                activities = activity_api.filter(
                        target_chembl_id=tr['target_chembl_id'], assay_type__in=include_assay_types
                    ).filter(pchembl_value__isnull=False).only(
                        "activity_id",
                        "assay_chembl_id",
                        "assay_description",
                        "assay_type",
                        "molecule_chembl_id",
                        "pchembl_value",
                        "standard_type",
                        "standard_value",
                        "standard_units",
                        "standard_relation",
                        "target_organism",
                        "target_chembl_id",
                        "document_chembl_id")
                tmp_df_list.append(pd.DataFrame.from_dict(activities))
            df_activity = pd.concat(tmp_df_list, ignore_index=True)
            df_activity.drop(['units', 'type', 'value', 'relation'], axis=1, inplace=True)
            if duplicate_option == "Keep Maximum Value":
                df_activity = df_activity.loc[df_activity.groupby('molecule_chembl_id')['pchembl_value'].idxmax()].sort_values('activity_id')
            if duplicate_option == "Keep Minimum Value":
                df_activity = df_activity.loc[df_activity.groupby('molecule_chembl_id')['pchembl_value'].idxmin()].sort_values('activity_id')
            if duplicate_option == "Do Nothing":
                pass
            if duplicate_option == "Keep Maximum Value" or duplicate_option == "Keep Minimum Value":
                df_activity.reset_index(drop=True, inplace=True)
            df_activity = df_activity[~df_activity['assay_description'].apply(lambda x: any(remove_text in x.lower() for remove_text in assay_exclude_list))]
            df_activity = df_activity[~df_activity['standard_type'].isin(exclude_activity_types)]

            # fetch molecule-related data
            molecule_provider = molecule_api.filter(
                molecule_chembl_id__in=list(df_activity["molecule_chembl_id"])
                ).only("molecule_chembl_id", "molecule_structures")
            df_molecule = pd.DataFrame.from_dict(molecule_provider)
            df_molecule.dropna(axis=0, how="any", inplace=True)
            df_molecule.drop_duplicates("molecule_chembl_id", keep='first', inplace=True)
            df_molecule['canonical_smiles'] = df_molecule['molecule_structures'].apply(lambda x: x.get('canonical_smiles', None))
            df_molecule.drop("molecule_structures", axis=1, inplace=True)
            df_molecule.dropna(axis=0, how="any", inplace=True)

            # merge both dataframe
            df = pd.merge(df_molecule, df_activity, on="molecule_chembl_id", how='inner')
            df.reset_index(drop=True, inplace=True)
            df['pchembl_value'] = df['pchembl_value'].astype(float)
            df['standard_value'] = df['standard_value'].astype(float)
            # reorder columns
            st.session_state.df = df[[
                "canonical_smiles",
                "pchembl_value",
                "assay_description",
                "standard_type",
                "standard_value",
                "standard_units",
                "standard_relation",
                "assay_chembl_id",
                "assay_type",
                "molecule_chembl_id",
                "target_organism",
                "target_chembl_id",
                "document_chembl_id"]]

if st.session_state.df is not None:
    df = st.session_state.df
    st.header("Dataset Preview")
    st.info("If you want to modify the below dataset, you can download it as a CSV file. Then, go back to the beginning and upload the file.")
    st.text(f"Record count: {len(df)}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.header("FLAML Settings")

    st.subheader("Dataset Preparation")
    smiles_col_name = st.selectbox(
        "Select SMILES column name",
        options=df.columns,
        index=None,
    )
    label_col_name = st.selectbox(
        "Select target column name",
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

    st.subheader("Target Value Standardization")
    st.info("Use this feature if the target values are not standardized or otherwise processed.")
    st.warning("CAUTION: Do not standardize if you are creating a prediction model for use in generating molecules with specific values. e.g., To generate molecules that have LogP value with around 5.0.")
    use_scaler = st.checkbox("Apply standardization for target values")
    if use_scaler:
        output_scaler_name = st.text_input(
            "Enter filename for scaler in pickle format",
            value="standard_scaler.pkl"
        )

    st.subheader("Output Filename For Best Estimator")
    output_fname = st.text_input(
        "Enter filename in pickle format",
        value="flaml_model.pkl"
    )

    if st.button("Run AutoML", type='primary'):
        with st.spinner("AutoML started..."):
            automl = AutoML()

            df_x = df[smiles_col_name].apply(lambda x: Chem.MolFromSmiles(x))
            df_x = df_x.dropna().apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048))
            X = np.array(df_x.tolist())
            y = df[label_col_name]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)
            if use_scaler:
                scaler = StandardScaler()
                y_train = np.squeeze(scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)))
                y_test = np.squeeze(scaler.transform(y_test.to_numpy().reshape(-1, 1)))

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
            if use_scaler:
                y_pred_transformed = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_test_transformed = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                ax.scatter(y_pred_transformed, y_test_transformed, marker='.', s=5, c='dimgray', alpha=0.2)
                min_val = min(y_pred_transformed.min(), y_test_transformed.min())
                max_val = max(y_pred_transformed.max(), y_test_transformed.max())
            else:
                ax.scatter(y_pred, y_test, marker='.', s=5, c='dimgray', alpha=0.2)
                min_val = min(y_pred.min(), y_test.min())
                max_val = max(y_pred.max(), y_test.max())
            ax.text(1.05, 0.05, f"Corr. Coef.: {correlation:.2f}\nBest Estimator: {automl.best_estimator}", fontsize='small', transform=ax.transAxes)
            padding = (max_val - min_val) * 0.05
            ax.set_xlim(min_val-padding, max_val+padding)
            ax.set_ylim(min_val-padding, max_val+padding)
            ax.set_xlabel("Prediction Value")
            ax.set_ylabel("Actual Value")
            st.pyplot(fig, use_container_width=False)

        with open(os.path.join('/app/shared_dir', output_fname), 'wb') as f:
            pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

        if use_scaler:
            with open(os.path.join('/app/shared_dir', output_scaler_name), 'wb') as f:
                pickle.dump(scaler, f)
