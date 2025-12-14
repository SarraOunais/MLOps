# Étape 1 : Utiliser une image Python officielle
FROM python:3.11-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3 : Copier les fichiers du projet dans le conteneur
COPY . /app

# Étape 4 : Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Étape 5 : Exposer le port sur lequel l'application va tourner
EXPOSE 8000

# Étape 6 : Commande pour lancer le service FastAPI
# Assumes que ton fichier principal s'appelle main.py et que l'app FastAPI s'appelle 'app'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
