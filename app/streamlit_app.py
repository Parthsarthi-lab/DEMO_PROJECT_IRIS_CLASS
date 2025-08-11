import streamlit as st
from pathlib import Path
from src.dsproject.predict import load_model, predict_single

MODEL_PATH = Path('artifacts/model/model.joblib')

st.set_page_config(page_title='Iris Classifier', layout='centered')
st.title('🌸 Iris Classifier (Beginner-Friendly)')

with st.expander('About this app'):
    st.write('Train with `python scripts/train.py` and then predict here.')

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input('Sepal length (cm)', 0.0, 10.0, 5.1, 0.1)
    sepal_width  = st.number_input('Sepal width (cm)',  0.0, 10.0, 3.5, 0.1)
with col2:
    petal_length = st.number_input('Petal length (cm)', 0.0, 10.0, 1.4, 0.1)
    petal_width  = st.number_input('Petal width (cm)',  0.0, 10.0, 0.2, 0.1)

if not MODEL_PATH.exists():
    st.warning('Model not found. Please train first: `python scripts/train.py`')
else:
    model = load_model(str(MODEL_PATH))
    if st.button('Predict'):
        out = predict_single(model, [sepal_length, sepal_width, petal_length, petal_width])
        label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        st.success(f"Prediction: **{label_map[out['prediction']]}**")
        if out['proba']:
            st.write('Class probabilities:', out['proba'])
