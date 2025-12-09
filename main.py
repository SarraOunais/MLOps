# main.py 

import warnings
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# Importer les fonctions du pipeline
from model_pipeline import (
    evaluate_model, load_model, predict, prepare_data,
    save_model, train_model
)

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Chemins des fichiers de données
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    # Définir l'expérience MLflow
    mlflow.set_experiment("Loan_Prediction_NaiveBayes")

    # Début du run MLflow
    with mlflow.start_run():

        print("=== Pipeline de Prédiction de Prêts - Modèle Naive Bayes ===\n")

        # Étape 1 — Préparation des données
        print("1. Préparation des données...")
        data = prepare_data(train_path, test_path)
        print(f"Shape X_train: {data['X_train'].shape}")
        print(f"Shape X_test: {data['X_test'].shape}")
        print(f"Shape X_final_test: {data['X_final_test'].shape}")
        print(f"Features utilisées: {list(data['X_train'].columns)}\n")

        # Étape 2 — Entraînement
        print("2. Entraînement du modèle Naive Bayes...")
        model = train_model(data["X_train"], data["y_train"])
        print()

        # Étape 3 — Évaluation
        print("3. Évaluation du modèle...")
        results = evaluate_model(model, data["X_test"], data["y_test"])
        # Ici pas besoin de re-log accuracy, evaluate_model le fait déjà
        print()

        # Étape 4 — Prédictions finales
        print("4. Prédictions sur les données de test...")
        final_predictions = predict(model, data["X_final_test"])
        print(f"Nombre de prédictions: {len(final_predictions)}")
        print(f"Prêts approuvés: {sum(final_predictions == 1)}")
        print(f"Prêts refusés: {sum(final_predictions == 0)}\n")

        # Étape 5 — Sauvegarde
        print("5. Sauvegarde du modèle...")
        save_model(model, "naive_bayes_loan_model.pkl")
        mlflow.sklearn.log_model(model, "naive_bayes_model")
        print()

        # Étape 6 — Test de chargement
        print("6. Test de chargement du modèle...")
        loaded_model = load_model("naive_bayes_loan_model.pkl")
        test_predictions = predict(loaded_model, data["X_test"].head())
        print(f"Prédictions avec modèle chargé: {test_predictions}\n")

        print("=== Pipeline terminé avec succès ===")

if __name__ == "__main__":
    main()
