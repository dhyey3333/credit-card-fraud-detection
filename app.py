import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Fraud Detection AI", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #020617, #000000);
    color: white;
}
h1 {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #22c55e);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>💳 Fraud Detection AI</h1>", unsafe_allow_html=True)

threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.3, step=0.01)
menu = st.sidebar.radio("Menu", ["Manual", "CSV Analysis", "Dashboard"])

# -----------------------------
# PDF FUNCTION
# -----------------------------
def generate_pdf(fraud, total, precision, recall, f1, cm_path):
    doc = SimpleDocTemplate("fraud_report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Fraud Detection Report", styles['Title']))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Total Transactions: {total}", styles['Normal']))
    content.append(Paragraph(f"Fraud Transactions: {fraud}", styles['Normal']))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Precision: {precision:.3f}", styles['Normal']))
    content.append(Paragraph(f"Recall: {recall:.3f}", styles['Normal']))
    content.append(Paragraph(f"F1 Score: {f1:.3f}", styles['Normal']))
    content.append(Spacer(1, 20))

    content.append(Image(cm_path, width=400, height=300))
    doc.build(content)

    return "fraud_report.pdf"

# -----------------------------
# GAUGE
# -----------------------------
def fraud_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%", 'font': {'size': 50}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ffffff"},
            'steps': [
                {'range': [0, 30], 'color': "#00ff88"},
                {'range': [30, 70], 'color': "#ffd000"},
                {'range': [70, 100], 'color': "#ff1a1a"}
            ]
        }
    ))
    fig.update_layout(paper_bgcolor="#020617", font={'color': "white"})
    return fig

# =========================================================
# 🔹 MANUAL
# =========================================================
if menu == "Manual":

    st.subheader("🔍 Manual Transaction Check")

    col1, col2 = st.columns(2)
    amt = col1.number_input("Amount ₹", value=1000.0)
    time_val = col2.number_input("Time", value=40000.0)

    if st.button("Analyze Transaction"):

        x = np.zeros(len(FEATURES))
        x[0] = time_val
        x[1] = amt

        prob = model.predict_proba(scaler.transform([x]))[0][1]

        if prob >= threshold:
            st.error(f"🚨 Fraud ({prob:.2%})")
        else:
            st.success(f"✅ Legit ({prob:.2%})")

        st.plotly_chart(fraud_gauge(prob), use_container_width=True)

# =========================================================
# 🔹 CSV ANALYSIS
# =========================================================
elif menu == "CSV Analysis":

    st.subheader("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        data = pd.read_csv(file)

        # SAMPLE VIEW
        st.write(f"Total Rows: {len(data)}")
        st.dataframe(data.sample(min(200, len(data))))

        if st.button("Analyze Dataset"):

            original = data.copy()

            for col in FEATURES:
                if col not in data.columns:
                    data[col] = 0

            probs = model.predict_proba(scaler.transform(data[FEATURES]))[:, 1]
            data["Fraud"] = (probs >= threshold).astype(int)

            fraud = data["Fraud"].sum()
            total = len(data)

            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud", fraud)
            c2.metric("Total", total)
            c3.metric("Rate", f"{(fraud/total)*100:.2f}%")

            # ---------------- CONFUSION MATRIX ----------------
            if "Class" in original.columns:

                y_true = original["Class"]
                y_pred = data["Fraud"]

                cm = confusion_matrix(y_true, y_pred)

                fig = ff.create_annotated_heatmap(
                    z=cm,
                    x=["Pred Legit", "Pred Fraud"],
                    y=["Actual Legit", "Actual Fraud"],
                    colorscale="Viridis"
                )
                fig.update_layout(paper_bgcolor="#020617", font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)

                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                # SAVE CM IMAGE FOR PDF
                cm_path = "cm.png"
                plt.figure(figsize=(4,4))
                sns.heatmap(cm, annot=True, fmt='d')
                plt.savefig(cm_path)
                plt.close()

                pdf = generate_pdf(fraud, total, precision, recall, f1, cm_path)

                # ✅ FIXED DOWNLOAD BUTTON
                with open(pdf, "rb") as f:
                    st.download_button(
                        label="📄 Download Report",
                        data=f,
                        file_name="fraud_report.pdf",
                        mime="application/pdf"
                    )

            # ---------------- FRAUD TABLE ----------------
            st.markdown("## 🚨 Fraud Transactions")
            st.dataframe(original[data["Fraud"] == 1].head(20))

# =========================================================
# 🔹 DASHBOARD
# =========================================================
elif menu == "Dashboard":

    st.subheader("📊 Fraud Analytics Dashboard")

    file = st.file_uploader("Upload CSV for Dashboard", type=["csv"])

    if file is None:
        st.warning("Upload dataset to view dashboard")
        st.stop()

    data = pd.read_csv(file)

    for col in FEATURES:
        if col not in data.columns:
            data[col] = 0

    probs = model.predict_proba(scaler.transform(data[FEATURES]))[:, 1]
    data["Fraud"] = (probs >= threshold).astype(int)

    fraud = data["Fraud"].sum()
    total = len(data)

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud", fraud)
    c2.metric("Total", total)
    c3.metric("Rate", f"{(fraud/total)*100:.2f}%")

    st.markdown("---")

    fig1 = px.pie(
        names=["Legit", "Fraud"],
        values=[total - fraud, fraud],
        title="Fraud Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(data, x="Amount", nbins=50, title="Amount Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        data.sample(min(2000, len(data))),
        x="Time",
        y="Amount",
        color="Fraud",
        title="Time vs Amount"
    )
    st.plotly_chart(fig3, use_container_width=True)