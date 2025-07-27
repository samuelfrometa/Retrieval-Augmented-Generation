Crear entorno virtual

    python -m venv env


Entrar al entorno

    En caso de windows .\env\Scripts\Activate.ps1 o en caso dde linux soruce /env/scripts/activate


Con el comando 

    pip freeze > requirements.txt guardamos las versiones de las librerias instaladas

Instalacion de paquetes en el env

    pip install -r requirements.txt


IA

INSTALAR OLLAMA

y usar los comandos 

ollama pull llama3
ollama pull all-minilm

Crear la carpetaa docs y dentro meter los pdf

python rag.py