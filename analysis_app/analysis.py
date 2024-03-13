import mols2grid
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from rdkit import Chem

st.set_page_config(layout='wide')
st.image("logo_dark.png")
st.title("ChatMolGen Analysis Tool")


@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file) 
    return df


@st.cache_data
def render_dataframe(df):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True)
    

@st.cache_data
def calculate_moving_average(df, window_size):
    return df.rolling(window_size).mean()


@st.cache_data
def create_mols2grid(df, n_items_per_page, grid_height):
    mols2grid_html = mols2grid.MolGrid(df, smiles_col='smiles', size=(170, 130), prerender=False).to_interactive(n_items_per_page=n_items_per_page)
    components.html(mols2grid_html, height=grid_height, scrolling=True)

def filter_dataframe(df: pd.DataFrame, key: str, show_checkbox=True, doModify=False) -> pd.DataFrame:
    """
    This function is based on https://github.com/tylerjrichards/st-filter-dataframe/blob/main/streamlit_app.py written by Streamlit Data Team
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = False
    if show_checkbox:
        modify = st.checkbox("Add filters", key=key)
    if doModify:
        modify = True

    if not modify:
        return df

    df = df.copy()

    modification_container = st.container()
    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns, key=f"{key}_")
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=sorted(list(df[column].unique())),
                    key=f"{key}_{column}"
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                    key=f"{key}_{column}"
                )
                df = df[df[column].between(*user_num_input)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    key=f"{key}_{column}"
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]
    return df

    
def remove_atom_map(smiles: str):
    if pd.isna(smiles) or smiles == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def has_atom_map(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() != 0:
            return True
    return False


df = None
st.header("File upload")
uploaded_file = st.file_uploader("Select the result file of a molecule generation")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if has_atom_map(df['smiles'][10]):
        df['smiles'] = df['smiles'].apply(remove_atom_map)
        df['parents'] = df['parents'].apply(remove_atom_map)
    st.session_state['df'] = df

if 'df' in st.session_state:
    df = st.session_state['df']

if df is not None:
    exclude_columns_for_line_chart = ['generated_id', 'smiles', 'depth', 'elapsed_time','is_through_filter']  # Chemtsv2
    exclude_columns_for_line_chart += ['rank', 'parents', 'mutation_type', 'mutation_rxn']  # XGG

    st.header("Dataframe")
    _df = filter_dataframe(df, key="dataframe")
    render_dataframe(_df)

    with st.sidebar:
        st.header("Molecule Grid Viewer Option")
        n_items_per_page = st.number_input("Enter number of molecules to show", value=24, max_value=128)
        grid_height = st.number_input("Enter the value of grid height in pixel", value=600, max_value=1000)
    st.header("Molecule Grid Viewer")
    create_mols2grid(filter_dataframe(df, key="mols2grid"), n_items_per_page, grid_height)

    with st.sidebar.form(key='line_chart'):
        st.header("Line Chart Option")
        option = st.multiselect(
            'Select columns to show in line chart',
            [c for c in df.columns if c not in exclude_columns_for_line_chart],
            placeholder='reward')
        window_size = st.slider('Window size', value=50, min_value=1, max_value=300)
        submit_button = st.form_submit_button(label="Submit")

    if option != [] and submit_button:
        df_ma = calculate_moving_average(df[option], window_size)
        st.header("Line Chart")
        st.line_chart(df_ma, y=option)
