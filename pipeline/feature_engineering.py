import pandas as pd

def generate_features(df: pd.DataFrame):

    df_new = df.copy()

    original_cols = list(df_new.columns)

    num_cols = df_new.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # ================= FEATURE INTERACTIONS =================
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):

            col1 = num_cols[i]
            col2 = num_cols[j]

            df_new[f"{col1}_plus_{col2}"] = df_new[col1] + df_new[col2]
            df_new[f"{col1}_mul_{col2}"] = df_new[col1] * df_new[col2]

    # ================= SQUARE FEATURES =================
    for col in num_cols:
        df_new[f"{col}_square"] = df_new[col] ** 2

    # ================= STATISTICAL FEATURES =================
    if len(num_cols) >= 2:
        df_new["mean_numeric"] = df_new[num_cols].mean(axis=1)
        df_new["std_numeric"] = df_new[num_cols].std(axis=1)

    # Replace NaN created by std calculation
    df_new = df_new.fillna(0)

    new_features = [col for col in df_new.columns if col not in original_cols]

    all_feature_names = list(df_new.columns)

    return df_new, new_features, all_feature_names