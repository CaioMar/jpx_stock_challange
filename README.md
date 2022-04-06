# Projeto JPX Stock Challenge

## Iniciando o projeto

Para iniciar um projeto é necessário:

* Ter o Anaconda instalado;
* Ter a API key, arquivo kaggle.json, que pode ser baixado no perfil do seu user kaggle no site da plataforma, presente na pasta .kaggle que fica em sua HOME;
* Criar um ambiente conda usando o arquivo environment.yml com o seguinte comando no terminal:
```bash
conda env create -f environment.yml
```
* Rodar os comandos abaixo no terminal ativar o ambiente virtual e rodar o script de incialização que irá baixar os arquivos usando a api do kaggle:
```bash
conda activate stocks
python startup.py
```