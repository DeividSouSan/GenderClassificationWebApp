import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.title("Gender Classification Model")

st.header("Objetivo")

st.markdown(
    """
Esse webapp foi feito por mim (Deivid Souza Santana) com o intuito de prática um pouco sobre Machine Learning, uma área que tenho grande curiosidade.

O modelo a segui foi feito com base em um Dataset que fornece o nome de uma pessoa e o sexo associado (por exemplo: Deivid, Masculino). Dessa maneira, minha ideia foi criar um modelo que aprendesse a fazer essa classificação por conta própria.
"""
)
