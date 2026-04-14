from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import pandas as pd


def train_models(X, y):

    # Copy dataset
    X = X.copy()

    # ================= HANDLE EMPTY DATA =================
    if X.shape[0] == 0:
        raise ValueError("Dataset is empty after preprocessing")

    # ================= HANDLE MISSING VALUES =================
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    if len(cat_cols) > 0:
        for col in cat_cols:
            X[col] = X[col].fillna(X[col].mode()[0])

        # encode categorical
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # ================= ALIGN X AND y =================
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # ================= TRAIN TEST SPLIT =================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ================= MODELS =================
    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        )
    }

    results = {}

    best_model = None
    best_name = ""
    best_acc = 0

    # ================= TRAIN MODELS =================
    for name, model in models.items():

        try:

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = round(accuracy_score(y_test, preds), 4)

            results[name] = acc

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = name

        except Exception as e:

            # Skip model if it fails
            results[name] = 0

    return best_model, best_name, results, best_acc


# ================= FEATURE IMPORTANCE =================
def get_feature_importance(model, feature_names):

    if hasattr(model, "feature_importances_"):

        importances = model.feature_importances_

        return dict(
            sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
        )

    return {}