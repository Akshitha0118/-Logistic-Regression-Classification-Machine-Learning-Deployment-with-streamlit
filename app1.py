import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Logistic Regression Dashboard",
    layout="wide"
)

# ---------------- LOAD CSS ----------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------- TITLE ----------------
st.markdown('<div class="title">📊 Logistic Regression Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Actual Data & Future Prediction Visualization</div>', unsafe_allow_html=True)

# =====================================================
# 1️⃣ LOAD ACTUAL DATA
# =====================================================
@st.cache_data
def load_actual_data():
    return pd.read_csv("logit classification.csv")

actual_df = load_actual_data()

st.subheader("📊 Actual Dataset Preview")
st.dataframe(actual_df.head())

# =====================================================
# 2️⃣ LOAD FUTURE DATA
# =====================================================
@st.cache_data
def load_future_data():
    return pd.read_csv("final1.csv")

future_df = load_future_data()

st.subheader("📊 Future Dataset Preview")
st.dataframe(future_df.head())

# =====================================================
# 3️⃣ ENCODE CATEGORICAL COLUMNS
# =====================================================
label_encoders = {}

for col in actual_df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    actual_df[col] = le.fit_transform(actual_df[col])
    label_encoders[col] = le

# ---------------- FEATURES & TARGET ----------------
X = actual_df.iloc[:, [2, 3]]   # feature columns
y = actual_df.iloc[:, -1]       # target column

# ---------------- TRAIN TEST SPLIT ----------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# ---------------- SCALING ----------------
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# ---------------- MODEL ----------------
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# =====================================================
# 4️⃣ METRICS
# =====================================================
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

st.markdown("### 📈 Model Performance")

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.write("Confusion Matrix")
col2.write(cm)

# =====================================================
# 5️⃣ ROC CURVE
# =====================================================
y_pred_prob = classifier.predict_proba(x_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

st.markdown("### 📉 ROC Curve")

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
ax.grid()

st.pyplot(fig)

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    "<hr><center>Built with ❤️ using Streamlit & Logistic Regression</center>",
    unsafe_allow_html=True
)

