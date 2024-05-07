import pandas as pd
from faker import Faker
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import pickle


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    vectorizer = CountVectorizer(analyzer='char')

    # Seleciona os dados necessários do DataFrame
    df = df.loc[:, ['Name', 'Gender']]

    # Inicio da Verificação de Oversampling
    masc_names = df[df['Gender'] == 'M'].shape[0]
    fem_names = df[df['Gender'] == 'F'].shape[0]

    needed_number_of_names = abs(masc_names - fem_names)

    if needed_number_of_names > 0:
        faker = Faker()
        generated_names = []

        if masc_names < fem_names:
            name = faker.first_name_male()
            sex = 'M'

        elif fem_names < masc_names:
            name = faker.first_name_fem()
            sex = 'F'

        for _ in range(needed_number_of_names):
            generated_names.append([name, sex])

        generated_names_df = pd.DataFrame(
            generated_names, columns=['Name', 'Gender'])

        df = pd.concat([df, generated_names_df], ignore_index=True)

    # One-hot Encoding para altear 'F' e 'M' para 0 e 1
    df['Gender'] = df['Gender'].replace({'F': 0, 'M': 1})

    # Convertendo os nomes em 'tokens' (números)
    names = df['Name'].values  # Esse vetor é considerado um documento de texto

    vectorizer = vectorizer.fit(names)

    # Salvando o Vetorizador preenchido para uso posterior
    with open("vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    count_matrix = vectorizer.transform(names).toarray()

    feature_chars = vectorizer.get_feature_names_out()

    count_letter_df = pd.DataFrame(
        data=count_matrix, columns=feature_chars, index=names)

    # Features
    X = count_letter_df

    # Label
    y = df['Gender']

    return X, y


def train_model(X, y, test_size: float, random_state: int, n_neighbors: int):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Treinando o Modelo
    classifier.fit(X_train.values, y_train.values)

    # Testando o Modelo
    score = classifier.score(X_test, y_test)

    return {"Model": classifier, "Accuracy": score}


def preprocesses_input(name: str):
    # Recupera o vetorizador já preenchido
    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    
    matrix = vectorizer.transform([name]).toarray()

    chars = vectorizer.get_feature_names_out()

    df_sample = pd.DataFrame(matrix, columns=chars, index=[name])

    return df_sample

