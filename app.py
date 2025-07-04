import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from q_learning import clean_and_train_svm

st.set_page_config(page_title="Mental Health SVM", layout="wide")
st.title("🧠 Mental Health Prediction - Input & Train SVM")

# Inisialisasi session state untuk menyimpan data
if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame()

# Form input manual
with st.form("input_form"):
    gender = st.selectbox("Choose your gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 15, 60, 25)
    course = st.selectbox("What is your course?", ["CS", "EE", "ME", "Other"])
    year = st.selectbox("Current year of Study", [1, 2, 3, 4])
    cgpa = st.number_input("What is your CGPA?", min_value=0.0, max_value=4.0, step=0.01)
    marital = st.selectbox("Marital status", ["Single", "Married"])
    depression = st.selectbox("Do you have Depression?", ["Yes", "No"])
    anxiety = st.selectbox("Do you have Anxiety?", ["Yes", "No"])
    panic = st.selectbox("Do you have Panic attack?", ["Yes", "No"])
    seek_help = st.selectbox("Did you seek any specialist for a treatment?", ["Yes", "No"])

    submitted = st.form_submit_button("➕ Tambahkan Data")

    if submitted:
        new_data = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Course": course,
            "Year": year,
            "CGPA": cgpa,
            "Marital": marital,
            "Depression": depression,
            "Anxiety": anxiety,
            "PanicAttack": panic,
            "SeekHelp": seek_help
        }])

        st.session_state["data"] = pd.concat([st.session_state["data"], new_data], ignore_index=True)
        st.success("✅ Data berhasil ditambahkan!")

# Tampilkan data yang sudah dimasukkan
st.subheader("📋 Data Terkumpul")
st.dataframe(st.session_state["data"])

# Training SVM jika data > 20
if len(st.session_state["data"]) >= 20:
    st.subheader("🧠 Training SVM Model...")

    try:
        # Target: Depression
        result = clean_and_train_svm(st.session_state["data"], target_column="Depression")

        st.success(f"✅ Train Accuracy: {result['train_accuracy']:.2f}")
        st.success(f"✅ Test Accuracy: {result['test_accuracy']:.2f}")

        st.subheader("📈 Classification Report")
        st.json(result["report"])
        

        st.subheader("🧾 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error saat melatih SVM: {e}")
else:
    st.info(f"🕒 Masukkan minimal 20 data untuk memulai training (sekarang: {len(st.session_state['data'])})")
