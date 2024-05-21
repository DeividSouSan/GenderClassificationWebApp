# Gender Classification Web App with Streamlit
Esse projeto é uma aplicação web onde o usuário pode treinar um modelo de Machine Learning para prever o sexo de uma pessoa a partir do seu nome.

A biblioteca Streamlit e conceitos de Machine Learning foram utilizados durante o desenvolvimento.

O objetivo do projeto era:
1. Entender sobre Machine Learning e criar o meu proprio modelo.
2. Criar uma forma de interação do usuário com o modelo de Machine Learning.

## Screenshots
Início da Página de Treinamento:
![selecting-dataset](https://github.com/DeividSouSan/GenderClassificationWebApp/assets/49818020/47922088-4e40-49b3-8231-6214c246e458)

Início da Página de Uso:
![test-model](https://github.com/DeividSouSan/GenderClassificationWebApp/assets/49818020/04093e03-94ae-47ee-8440-37222b25621e)

## Como Rodar
### Clone o Repositório
Primeiro é necessário *clonar* ou *baixar* o repositório em sua máquina.

Você pode fazer isso rodando o comando abaixo na sua máquina:
```
git clone https://github.com/DeividSouSan/GenderClassificationWebApp.git
```

### Crie um Ambiente Virtual
Em seguida, antes de baixar as dependência, é melhor criar um **ambiente virtual** a fim de isolar as dependencias globais na sua máquina, das dependencias do projeto.

Novamente, no terminal, acesse a pasta do repositório e escreva:
```
python3 -m venv .venv
```

Assim uma pasta chamada `.venv` será criada. Agora, basta apenas ativar o *ambiente virtual*:
```
source .venv/bin/activate
```
### Baixe as Dependências
Com o ambiente virtual configurado, baixe as dependências:

```
pip install requirements.txt
```

### Rode a Aplicação
Finalmente, para rodar a aplicação, basta rodar:

```
streamlit run Home.py
```
## Stack Utilizada
O dataset disponiblizado pode ser encontrado e baixado aqui também:
[UCI ML Repository](https://archive.ics.uci.edu/dataset/591/gender+by+name)

![Python](https://github.com/DeividSouSan/GenderClassificationWebApp/assets/49818020/f983b7fe-59f1-4fff-a9e3-c6746b628ac2)
![Pandas](https://github.com/DeividSouSan/GenderClassificationWebApp/assets/49818020/5445ead4-e476-44d7-ba98-683d9a693581)
![Scikit](https://github.com/DeividSouSan/GenderClassificationWebApp/assets/49818020/e62a7b73-6ade-425b-a06a-645491051a4b)
![Streamlit](https://github.com/DeividSouSan/GenderClassificationWebApp/assets/49818020/383d72f3-2ad0-41a6-9af7-2da9bace42c0)
![VSCode](https://github.com/DeividSouSan/GenderClassificationWebApp/assets/49818020/1861eeb0-9746-4a42-a416-eeab27f018e8)



