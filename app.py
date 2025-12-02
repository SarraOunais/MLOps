from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import pandas as pd

from model_pipeline import (
    preprocess_data,
    load_model,
    predict,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    retrain_model
)

# -----------------------------
# Initialisation de l'application
# -----------------------------
app = FastAPI()
MODEL_PATH = "naive_bayes_loan_model.pkl"

# Charger le modèle initial
model = load_model(MODEL_PATH)
trained_columns = model.feature_names_in_


# -----------------------------
# Endpoint /predict
# -----------------------------
class LoanData(BaseModel):
    Loan_ID: str
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float 
    Credit_History: float
    Property_Area: str

@app.post("/predict")
def predict_loan(data: LoanData):
    try:
        df = pd.DataFrame([data.dict()])

        # Prétraitement
        df_processed = preprocess_data(df, is_train=False)

        # Supprimer Loan_ID
        if "Loan_ID" in df_processed:
            df_processed = df_processed.drop("Loan_ID", axis=1)

        # Ajouter les colonnes manquantes
        for col in trained_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        # Réordonner les colonnes
        df_processed = df_processed[trained_columns]

        pred = predict(model, df_processed)[0]
        result = "Approved" if pred == 1 else "Rejected"

        return {"Loan_ID": data.Loan_ID, "prediction": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------
# Endpoint /retrain
# -----------------------------
class RetrainParams(BaseModel):
    var_smoothing: float = 1e-9

@app.post("/retrain")
def retrain_model_endpoint(params: RetrainParams, background_tasks: BackgroundTasks):
    """
    Réentraîne le modèle Naive Bayes avec var_smoothing configurable
    en tâche de fond.
    """
    def retrain_task(params: RetrainParams):
        global model, trained_columns
        try:
            # Charger et préparer les données
            data = prepare_data("data/train.csv", "data/test.csv")
            X_train = data["X_train"]
            y_train = data["y_train"]
            X_test = data["X_test"]
            y_test = data["y_test"]

            # Réentraîner le modèle
            new_model = retrain_model(X_train, y_train, var_smoothing=params.var_smoothing)

            # Évaluer
            results = evaluate_model(new_model, X_test, y_test)
            print(f"Modèle Naive Bayes réentraîné avec accuracy: {results['accuracy']:.4f}")

            # Sauvegarder le modèle
            save_model(new_model, MODEL_PATH)

            # Mettre à jour le modèle en mémoire
            model = new_model
            trained_columns = model.feature_names_in_

        except Exception as e:
            print(f"Erreur lors du réentraînement: {e}")

    # Lancer la tâche en arrière-plan
    background_tasks.add_task(retrain_task, params)

    return {"status": "retraining started", "hyperparameters": params.dict()}
