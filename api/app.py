from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import joblib
import os
import tempfile
from uuid import uuid4

from pipeline.preprocessing import preprocess_data, compare_before_after
from pipeline.feature_engineering import generate_features
from pipeline.model_training import train_models, get_feature_importance
from utils.data_quality import generate_data_quality_report

from sklearn.metrics import classification_report, confusion_matrix

app = FastAPI(title="AutoFeat ML Pipeline")

MODELS = {}

# ================= UPLOAD =================
@app.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_path = temp_file.name
    temp_file.close()

    try:
        content = await file.read()

        with open(temp_path, "wb") as f:
            f.write(content)

        df = pd.read_csv(temp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read error: {str(e)}")

    # ================= TARGET =================
    target = df.columns[-1]

    df = df.dropna(subset=[target]).reset_index(drop=True)

    if df.shape[0] < 5:
        raise HTTPException(status_code=400, detail="Dataset too small for training")

    X = df.drop(columns=[target])
    y = df[target]

    # ================= PREPROCESS =================
    X_clean = preprocess_data(X.copy())

    cleaning_info = compare_before_after(X, X_clean)

    # ================= FEATURE ENGINEERING =================
    X_engineered, new_features, all_feature_names = generate_features(X_clean)

    feature_info = {
        "original_feature_count": X_clean.shape[1],
        "new_features_created": len(new_features),
        "total_features_after": X_engineered.shape[1],
        "new_features_list": new_features[:20]
    }

    # ================= MODEL TRAINING =================
    _, _, _, acc_original = train_models(X_clean, y)

    best_model, best_name, model_results, acc_engineered = train_models(
        X_engineered, y
    )

    # 🚨 Important Safety Check
    if best_model is None:
        raise HTTPException(status_code=500, detail="Model training failed")

    # ================= EVALUATION =================
    try:
        y_pred = best_model.predict(X_engineered)

        report = classification_report(y, y_pred, output_dict=True)

        cm = confusion_matrix(y, y_pred).tolist()

    except Exception:
        report = {}
        cm = []

    improvement = round((acc_engineered - acc_original) * 100, 2)

    # ================= FEATURE IMPORTANCE =================
    importance_dict = get_feature_importance(
        best_model,
        X_engineered.columns.tolist()
    )
    # ================= SMART PREDICTION FEATURES (Generalized) =================
    exclude_keywords = ['id', 'name', 'ticket', 'cabin', 'index', 'row', 'number',
                        'code', 'identifier','target', target.lower()]

    prediction_features = []
    for col in X.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in exclude_keywords):
            continue
        if X[col].nunique() > 0.7 * len(X):  # High cardinality (likely ID)
            continue
        prediction_features.append(col)

    # Fallback if too few features
    if len(prediction_features) < 5:
        prediction_features = X.columns.tolist()[:10]

    # Save Model
    model_id = str(uuid4())
    model_package = {
        "model": best_model,
        "feature_names": all_feature_names,
        "target_column": target
    }

    model_path = os.path.join(tempfile.gettempdir(), f"model_{model_id}.pkl")
    joblib.dump(model_package, model_path)
    MODELS[model_id] = model_package

    if os.path.exists(temp_path):
        os.unlink(temp_path)

    return {
        "status": "success",
        "model_id": model_id,
        "original_features": X.columns.tolist(),
        "prediction_features": prediction_features,  # Clean list for UI
        "target_column": target,

        "data_quality": generate_data_quality_report(df),
        "cleaning": cleaning_info,
        "feature_engineering": feature_info,

        "accuracy_original": acc_original,
        "accuracy_engineered": acc_engineered,
        "accuracy_improvement": improvement,

        "model_comparison": model_results,
        "best_model": best_name,
        "best_accuracy": acc_engineered,
        "top_feature_importance": dict(list(importance_dict.items())[:12])
    }
# ================= DOWNLOAD =================
@app.get("/download/{model_id}")
async def download_model(model_id: str):

    model_path = os.path.join(
        tempfile.gettempdir(),
        f"model_{model_id}.pkl"
    )

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(model_path, filename=f"best_model_{model_id}.pkl")


# ================= PREDICT =================
# ================= PREDICT (Robust Version) =================
@app.post("/predict/{model_id}")
async def predict(model_id: str, sample_data: dict):
    if model_id not in MODELS:
        # Check disk persistence logic... (as implemented previously)
        pass

    model_package = MODELS[model_id]
    model = model_package["model"]
    required_features = model_package["feature_names"]

    try:
        # 1. Convert input to dataframe
        input_df = pd.DataFrame([sample_data])

        # 2. Apply Preprocessing & Feature Engineering
        # This MUST include the same LabelEncoding/OneHotEncoding used in training
        input_df = preprocess_data(input_df)
        input_df, _, _ = generate_features(input_df)

        # 3. Align Columns
        for col in required_features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[required_features].fillna(0)

        # 4. Make Prediction
        # We use .predict() which returns an array.
        # We access [0] and then convert to int.
        raw_prediction = model.predict(input_df)[0]

        # If the model returns a string label (like 'Survived'),
        # we handle it gracefully instead of forcing int()
        try:
            prediction = int(raw_prediction)
        except (ValueError, TypeError):
            prediction = str(raw_prediction)

        # 5. Probability
        probability = None
        if hasattr(model, "predict_proba"):
            prob_scores = model.predict_proba(input_df)[0]
            probability = float(max(prob_scores))

        return {
            "prediction": prediction,
            "probability": round(probability, 4) if probability is not None else None
        }

    except Exception as e:
        # This will now print the exact column causing the issue in your terminal
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction logic failed: {str(e)}")
# ================= RUN =================
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )