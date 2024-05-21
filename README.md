# Gender Classification Web App with Streamlit
Esse projeto é uma aplicação web onde o usuário pode treinar um modelo de Machine Learning para prever o sexo de uma pessoa a partir do seu nome.

A biblioteca Streamlit e conceitos de Machine Learning foram utilizados durante o desenvolvimento.

O objetivo do projeto era:
1. Entender sobre Machine Learning e criar o meu proprio modelo.
2. Criar uma forma de interação do usuário com o modelo de Machine Learning.

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