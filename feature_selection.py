from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pipeline.preprocessing import preprocess_data
from pipeline.feature_engineering import generate_features

app = FastAPI()

best_model = None
best_model_name = None
best_features = None


@app.post("/upload")
async def upload_dataset(file: UploadFile):

    global best_model
    global best_model_name
    global best_features

    # read dataset
    df = pd.read_csv(file.file)

    # assume last column is target
    target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target]

    original_features = X.shape[1]

    # preprocessing
    X_clean = preprocess_data(X)

    # feature engineering
    X_engineered = generate_features(X_clean)

    engineered_features = X_engineered.shape[1]

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42
    )

    # models to compare
    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier()
    }

    results = {}

    best_accuracy = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        results[name] = round(acc, 3)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    # save best model
    joblib.dump(best_model, "trained_model.pkl")

    best_features = X_engineered.columns

    # feature importance (only if supported)
    feature_importance = {}

    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(
            zip(best_features, best_model.feature_importances_)
        )

    return {
        "original_features": original_features,
        "engineered_features": engineered_features,
        "model_results": results,
        "best_model": best_model_name,
        "feature_importance": feature_importance
    }


@app.get("/download_model")
def download_model():

    return FileResponse(
        "trained_model.pkl",
        media_type="application/octet-stream",
        filename="trained_model.pkl"
    )


@app.post("/predict")
def predict(data: dict):

    model = joblib.load("trained_model.pkl")

    df = pd.DataFrame([data])

    # preprocessing
    df_clean = preprocess_data(df)

    # feature engineering
    df_features = generate_features(df_clean)

    # align columns with training
    for col in best_features:
        if col not in df_features:
            df_features[col] = 0

    df_features = df_features[best_features]

    prediction = model.predict(df_features)

    return {"prediction": int(prediction[0])}