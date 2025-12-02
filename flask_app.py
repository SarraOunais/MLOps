from flask import Flask, render_template, request
import pandas as pd

from model_pipeline import (
    preprocess_data,
    load_model,
    predict,
    prepare_data,
    retrain_model,
    evaluate_model,
    save_model
)

app = Flask(__name__)


# -----------------------------
# Load initial Naive Bayes model
# -----------------------------
MODEL_PATH = "naive_bayes_loan_model.pkl"
model = load_model(MODEL_PATH)
trained_columns = model.feature_names_in_


# ------------------------------------------------------
# HOME PAGE + Prediction (equivalent to FastAPI /predict)
# ------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global model, trained_columns

    if request.method == "POST":
        form_data = request.form.to_dict()
        df = pd.DataFrame([form_data])

        # Convert numeric fields
        numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Preprocessing
        df_processed = preprocess_data(df, is_train=False)

        # Remove Loan_ID if present
        if "Loan_ID" in df_processed:
            df_processed = df_processed.drop("Loan_ID", axis=1)

        # Add missing columns
        for col in trained_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        df_processed = df_processed[trained_columns]

        # Make prediction
        pred = predict(model, df_processed)[0]
        result = "Approved" if pred == 1 else "Rejected"

        return render_template("result.html", result=result)

    return render_template("index.html")


# ------------------------------------------------------
# RETRAIN PAGE (equivalent to FastAPI /retrain)
# ------------------------------------------------------
@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    global model, trained_columns

    if request.method == "POST":
        # Retrieve hyperparameter
        var_smoothing = float(request.form.get("var_smoothing", 1e-9))

        # Prepare data
        data = prepare_data("data/train.csv", "data/test.csv")
        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]

        # Evaluate old model
        old_results = evaluate_model(model, X_test, y_test)
        old_acc = old_results["accuracy"]
        old_cm = old_results["confusion_matrix"].tolist()

        # Train new Naive Bayes model
        new_model = retrain_model(
            X_train,
            y_train,
            var_smoothing=var_smoothing
        )

        # Evaluate new one
        new_results = evaluate_model(new_model, X_test, y_test)
        new_acc = new_results["accuracy"]
        new_cm = new_results["confusion_matrix"].tolist()

        # Compare & choose best
        if new_acc >= old_acc:
            save_model(new_model, MODEL_PATH)
            model = load_model(MODEL_PATH)
            trained_columns = model.feature_names_in_
            better = "new"
        else:
            save_model(new_model, "naive_bayes_rejected.pkl")
            better = "old"

        comparison = {
            "old_model": {
                "accuracy": old_acc,
                "confusion_matrix": old_cm
            },
            "new_model": {
                "accuracy": new_acc,
                "confusion_matrix": new_cm,
                "hyperparameters": {"var_smoothing": var_smoothing}
            },
            "better_model": better
        }

        return render_template("retrain.html", comparison=comparison)

    return render_template("retrain.html", comparison=None)


# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)


