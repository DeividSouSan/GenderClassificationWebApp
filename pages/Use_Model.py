import pickle

import pandas as pd
import streamlit as st
from model.model import predict_data, preprocesses_input

st.set_page_config(
    page_title="Train and Test",
    page_icon="🤖",
)


st.header("Teste o Modelo")


def load_model(file_path):
    with open(file_path, "rb") as file:
        model = pickle.load(file)

    return model


def clicked():
    df = preprocesses_input(name)

    gender = predict_data(loaded_model, df)

    st.text(f"Genero previsto: {gender}")


model_file = st.file_uploader("Abrir arquivo", type=["pkl"])

if model_file is not None:
    with open("temp_model.pkl", "wb") as file:
        file.write(model_file.getvalue())

    with open("trained_model/model.pkl", "rb") as file:
        loaded_model = pickle.load(file)

    st.success("Modelo carregado com sucesso.")

st.header("Nome: ")
st.markdown(
    """
Digite um nome que você considere masculino ou feminino e veja se o modelo consegue acertar a classificação.
            """
)

name = st.text_input("Nome")
btn = st.button("Classificar", on_click=clicked, disabled=False if model_file else True)
