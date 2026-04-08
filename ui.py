import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AutoFeat ML Pipeline",
    page_icon="🤖",
    layout="wide"
)

# ================= HEADER =================
st.title("🤖 AutoFeat ML Pipeline")
st.caption("End-to-End Automated Machine Learning with Feature Engineering")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📌 Pipeline Steps")

    st.markdown("""
    **Workflow**

    1️⃣ Upload Dataset  
    2️⃣ Data Quality Analysis  
    3️⃣ Data Cleaning  
    4️⃣ Feature Engineering  
    5️⃣ Model Training  
    6️⃣ Download Model  
    7️⃣ Trial Prediction
    """)

    st.info("Dataset must contain **target variable in last column**")

# ================= SESSION =================
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.results = None

# ================= DATASET UPLOAD =================
st.header("📤 Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:

    if st.button("🚀 Run AutoML Pipeline", use_container_width=True):

        with st.spinner("Running pipeline... please wait"):

            response = requests.post(
                "http://127.0.0.1:8000/upload",
                files={"file": uploaded_file}
            )

            if response.status_code == 200:

                st.session_state.results = response.json()
                st.session_state.processed = True
                st.success("Pipeline completed successfully!")

            else:
                st.error(response.text)

# ================= SHOW RESULTS =================
if st.session_state.processed:

    results = st.session_state.results

    # ================= DATA QUALITY =================
    st.header("📊 Dataset Overview")

    quality = results["data_quality"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", quality["shape"]["rows"])
    c2.metric("Columns", quality["shape"]["columns"])
    c3.metric("Duplicate Rows", quality["duplicate_rows"])
    c4.metric("Unique Classes", quality["target_info"]["unique_classes"])

    st.subheader("Missing Values")

    miss_df = pd.DataFrame({
        "Column": quality["missing_values"].keys(),
        "Missing Count": quality["missing_values"].values(),
        "Missing %": quality["missing_percentage"].values()
    })

    st.dataframe(miss_df, use_container_width=True)

    st.divider()

    # ================= DATA CLEANING =================
    st.header("🧹 Data Cleaning")

    cleaning = results["cleaning"]

    col1, col2 = st.columns(2)

    with col1:
        st.info("Before Cleaning")
        st.metric("Rows", cleaning["rows"]["before"])
        st.metric("Missing", cleaning["missing_values"]["before"])

    with col2:
        st.success("After Cleaning")
        st.metric("Rows", cleaning["rows"]["after"])
        st.metric("Missing", cleaning["missing_values"]["after"])

    st.divider()

    # ================= FEATURE ENGINEERING =================
    st.header("⚙️ Feature Engineering")

    feat = results["feature_engineering"]

    st.metric(
        "Feature Growth",
        f"{feat['original_feature_count']} → {feat['total_features_after']}"
    )

    if feat["new_features_created"] > 0:

        new_df = pd.DataFrame({"New Features": feat["new_features_list"]})

        st.dataframe(new_df, use_container_width=True)

    st.divider()

    # ================= ACCURACY IMPACT =================
    st.header("📈 Feature Engineering Impact")

    acc_original = results.get("accuracy_original", 0)
    acc_engineered = results.get("accuracy_engineered", 0)
    improvement = results.get("accuracy_improvement", 0)

    col1, col2, col3 = st.columns(3)

    col1.metric("Original Accuracy", f"{acc_original:.4f}")
    col2.metric("After FE", f"{acc_engineered:.4f}")
    col3.metric("Improvement", f"{improvement}%")

    impact_df = pd.DataFrame({
        "Stage": ["Original", "Feature Engineered"],
        "Accuracy": [acc_original, acc_engineered]
    })

    st.bar_chart(impact_df.set_index("Stage"))

    st.divider()

    # ================= MODEL COMPARISON =================
    st.header("🏆 Model Comparison")

    comp = results["model_comparison"]

    model_df = pd.DataFrame({
        "Model": comp.keys(),
        "Accuracy": comp.values()
    }).sort_values("Accuracy", ascending=False)

    st.dataframe(model_df, use_container_width=True)

    st.bar_chart(model_df.set_index("Model"))

    st.success(
        f"Best Model → **{results['best_model']}** "
        f"(Accuracy {results['best_accuracy']:.4f})"
    )

    st.divider()

    # ================= MODEL EVALUATION =================
    st.header("📊 Model Evaluation")

    report = results.get("classification_report")

    if report:

        st.subheader("Classification Report")

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df, use_container_width=True)

    cm = results.get("confusion_matrix")

    if cm:

        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots(figsize=(4, 3))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

    st.divider()

    # ================= FEATURE IMPORTANCE =================
    st.header("🔝 Feature Importance")

    importance = results.get("top_feature_importance", {})

    if importance:

        imp_df = pd.DataFrame({
            "Feature": importance.keys(),
            "Importance": importance.values()
        })

        st.bar_chart(imp_df.set_index("Feature"))

    else:

        st.info("Feature importance not available")

    st.divider()

    # ================= DOWNLOAD MODEL =================
    st.header("📥 Download Trained Model")

    download_url = f"http://127.0.0.1:8000/download/{results['model_id']}"

    st.link_button("Download Model (.pkl)", download_url)

    st.divider()

    # ====================== 7. TRIAL PREDICTION ======================
    st.header("7. 🔮 Trial Prediction")

    st.info("Enter values for **relevant original features** only. Target & ID columns are hidden.")

    if "prediction_features" in results and len(results["prediction_features"]) > 0:
        pred_features = results["prediction_features"][:12]

        with st.form("prediction_form"):
            st.subheader(f"Enter Values for {len(pred_features)} Features")

            sample_input = {}
            cols = st.columns(3)

            for idx, feat in enumerate(pred_features):
                with cols[idx % 3]:
                    sample_input[feat] = st.number_input(
                        label=feat,
                        value=0.0,
                        format="%.4f",
                        step=0.1
                    )

            submitted = st.form_submit_button("🔮 Make Prediction",
                                            type="primary",
                                            use_container_width=True)

            if submitted:
                with st.spinner("Predicting..."):
                    try:
                        predict_url = f"http://127.0.0.1:8000/predict/{results['model_id']}"
                        response = requests.post(predict_url, json=sample_input)

                        if response.status_code == 200:
                            data = response.json()
                            st.success(f"**Prediction: {data.get('predicted_class')}**")
                            if data.get("probability"):
                                st.info(f"Confidence: **{data['probability']*100:.1f}%**")
                        else:
                            st.error(response.text)
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.warning("No prediction features available. Please re-process dataset.")