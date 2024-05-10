import inspect
import os
import time

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

import model.model as model


class TrainAndTestPage:
    """
    Handles all Train and Test page configuration, components and logic.
    """

    def __init__(self):
        st.set_page_config(
            page_title="Train and Test",
            page_icon="🤖",
        )

    def select_dataset_section(self):
        st.header("Selecting Dataset")

        section_desc = """
        Selecione um conjunto de dados (dataset) do tipo CSV e que contenha como colunas o Nome e o Sexo da pessoa. Essas serão as únicas colunas relevantes para o projeto.
        """

        st.markdown(section_desc)

        self.file = st.file_uploader("Dataset", type=["csv"])

        if self.file is not None:
            try:
                self.df = model.load_data(self.file)
                st.dataframe(self.df)
                
                if "file" not in st.session_state:
                    st.session_state["file"] = True

            except Exception as err:
                st.error(err)

            

    def preprocessing_data_section(self):
        st.header("Dados Pre-Processados")

        section_desc = """
        O pre-processamento de dados consitiu de:
        - Selecionar no conjunto de dados somente as colunas relevantes e descartas as outras.

        - Oversampling: aumento do número de nomes caso haja diferença entre o número de nomes masculinos e o número de nomes femininos.

        - One-hot Enconding: converter os caracteres 'F' e 'M' em 0 e 1.


        - CountVectorizer: conversar dos caracteres dos nomes em um vetor que conta a quantidade de cada caractere possível.
        """
        st.markdown(section_desc)

        with st.expander("Ver código fonte."):
            preprocess_data_source = inspect.getsource(model.preprocess_data)

            st.code(preprocess_data_source, line_numbers=True)

        if "file" in st.session_state:
            self.X, self.y = model.preprocess_data(self.df)

            st.subheader("Dataframe de Entrada processado com os dados de cada nome:")
            st.dataframe(self.X)

            st.subheader("Dataframe de Saída processado com os resultado de cada nome:")
            st.dataframe(self.y)
        else:
            st.warning("Nenhum arquivo selecionado.")

    def configuring_model_section(self):
        st.header("Configuring Model:")

        self.test_size = st.slider(
            label="Percentage of Test Samples", min_value=0.10, max_value=1.0
        )
        self.random_state = st.number_input(label="Random Seed", min_value=0)
        self.n_neighbours = st.number_input(label="Number of Neighbors", min_value=1)

        btn_disabled = False if self.file else True

        st.button(
            "Train Model", on_click=self.train_model_callback, disabled=btn_disabled
        )

        if "trained-model" in st.session_state:
            st.success("Modelo treinado com sucesso, acesse Use Model para utiliza-lo.")

    def train_model_callback(self):
        import pickle

        classifier = model.train_model(
            self.X, self.y, self.test_size, self.random_state, self.n_neighbours
        )

        with open("model.pkl", "wb") as file:
            pickle.dump(classifier["Model"], file)

        with open("model_info.txt", "w", newline="\n") as file:
            c_time = os.path.getctime("model.pkl")

            file.write(f"Data de Criação: {time.ctime(c_time)}\n")
            file.write(f"Acurácia: {classifier['Accuracy']}\n")

    def donwload_model_section(self):
        st.header("Baixar Modelo")

        try:
            with open("model_info.txt", "r") as model_info:
                creation = model_info.read()
                accuracy = model_info.read()

                st.text(creation)
                st.text(accuracy)

            with open("model.pkl", "rb") as file:
                st.download_button(
                    label="Baixar Modelo",
                    data=file,
                    file_name="model.pkl",
                    mime="application/octet-stream",
                )
        except Exception as err:
            st.warning(err)

    def load_sections(self):
        self.select_dataset_section()
        self.preprocessing_data_section()
        self.configuring_model_section()
        self.donwload_model_section()


page = TrainAndTestPage()
page.load_sections()
