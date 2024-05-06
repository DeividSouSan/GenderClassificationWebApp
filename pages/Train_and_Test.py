import pandas as pd
import streamlit as st
from model.model import load_data, preprocess_data, train_model
from sklearn.feature_extraction.text import CountVectorizer


st.set_page_config(
    page_title="Train and Test",
    page_icon="🤖",
)

file, X, y = None, None, None


class SelectingDataset:
    def __init__(self):
        global file
        st.header("Selecting Dataset")
        st.markdown(
            """
        Selecione um Dataset que contenha as *features*: Name e Gender. 
        """
        )

        file = st.file_uploader("Dataset", type=["csv"])

        if file is not None:
            df = load_data(file)
            st.dataframe(df)

            if "file" not in st.session_state:
                st.session_state["file"] = True


class PreprocessingData:
    def __init__(self):
        global file
        st.header("Dados Pre-Processados")
        st.markdown(
            """
        O pre-processamento de dados consitiu de:
        - One-hot Enconding: converter os caracteres 'F' e 'M' em 0 e 1.
        - Oversampling: aumento do número de nomes caso haja diferença entre o número de nomes masculinos e o número de nomes femininos.
        - CountVectorizer: conversar dos caracteres dos nomes em um vetor que conta a quantidade de cada caractere possível.
                    """
        )

        if "file" in st.session_state:
            df = load_data(file)
            X, y = preprocess_data(df)

            st.dataframe(X)
            st.dataframe(y)
        else:
            st.warning("Nenhum arquivo selecionado.")


class ConfiguringModel:
    def __init__(self):
        st.header("Configuring Model:")

        self.test_size = st.slider(
            label="Percentage of Test Samples", min_value=0.10, max_value=1.0)
        self.random_state = st.number_input(label="Random Seed", min_value=0)
        self.n_neighbours = st.number_input(
            label="Number of Neighbors", min_value=1)

        file = True if 'file' in st.session_state else False
        btn_disabled = False if file else True

        st.button(
            "Train Model", on_click=self.train_model_callback, disabled=btn_disabled
        )

        if "trained-model" in st.session_state:
            st.success(
                "Modelo treinado com sucesso, acesse Use Model para utiliza-lo.")

    def train_model_callback(self):
        import pickle

        classifier = train_model(
            X, y, self.test_size, self.random_state, self.n_neighbours)

        with open("trained_model/model.pkl", "wb") as f:
            pickle.dump(classifier, f)

        st.session_state["trained-model"] = True


class DownloadModel:
    def __init__(self):
        st.header("Baixar Modelo")
        st.download_button(
            "Baixar Modelo", "trained-model/model.pkl", "model.pkl")


SelectingDataset()
PreprocessingData()
ConfiguringModel()
DownloadModel()
