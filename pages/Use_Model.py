import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

from model.model import preprocesses_input


class UseModelPage:
    def __init__(self):
        st.set_page_config(
            page_title="Train and Test",
            page_icon="🤖",
        )

        st.header("Teste o Modelo")

        page_description = """
        Essa página foi criada com o intuito de permitir o usuário testar o próprio modelo feito na página Train and Test. Dessa maneira, você terá um feedback real de como o modelo iria funcionar se fosse utilizado em alguma aplicação.
        """

        st.markdown(page_description)

    def load_file_section(self) -> None:
        self.model_file = st.file_uploader("Abrir arquivo", type=["pkl"])

        if self.model_file is not None:
            # Salva o modelo carregado em um arquivo temporário
            with open("tmp_model.pkl", "wb") as file:
                file.write(self.model_file.getvalue())

            if os.path.exists("tmp_model.pkl"):
                try:
                    # Usa o arquivo temporário para carregar o modelo
                    with open("tmp_model.pkl", "rb") as file:
                        self.classifier: KNeighborsClassifier = pickle.load(file)

                    st.success("Modelo carregado com sucesso.")
                except Exception as e:
                    st.error(f"Erro ao carregar o modelo: {e}")
        else:
            st.error("Arquivo temporário não encontrado.")

    def test_model_section(self) -> None:
        st.header("Nome: ")
        
        page_description = """
        Digite um nome que você considere masculino ou feminino e veja se o modelo consegue acertar a classificação.
        """
        
        st.markdown(page_description)
        
        self.test_name = st.text_input("Nome: ")
        
        st.button(
            "Classificar",
            on_click=self._clicked,
            disabled=False if self.model_file else True,
        )
        
        if not self.model_file:
            st.warning("Nenhum modelo foi carregado ainda.")
            
        if "predicted" in st.session_state:
            name = st.session_state["predicted"]["name"]
            gender = st.session_state["predicted"]["gender"]
            
            st.text(f"O nome {name} é {gender}.")

    def load_page(self) -> None:
        self.load_file_section()
        self.test_model_section()

    def _clicked(self) -> None:
        df = preprocesses_input(self.test_name)
        
        gender = self.classifier.predict(df)
        
        st.session_state["predicted"] = {
            "name": self.test_name, 
            "gender": "Feminino" if gender[0] == 0 else "Masculino"}


page = UseModelPage()
page.load_page()
