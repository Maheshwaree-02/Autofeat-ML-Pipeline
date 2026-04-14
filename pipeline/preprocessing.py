import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Handle missing values
    if num_cols:
        num_imputer = SimpleImputer(strategy='mean')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # One-hot encode categorical columns
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Scale numeric features
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    df = df.reset_index(drop=True)
    return df


def compare_before_after(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    return {
        "rows": {"before": len(before), "after": len(after)},
        "missing_values": {
            "before": int(before.isnull().sum().sum()),
            "after": int(after.isnull().sum().sum())
        },
        "columns": {
            "before": len(before.columns),
            "after": len(after.columns)
        },
        "new_columns_created": len(after.columns) - len(before.columns)
    }