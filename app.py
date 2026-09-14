import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Dashboard Churn", layout="wide")

COULEUR_RESTE = "#2a78d6"
COULEUR_CHURN = "#eb6834"

# --- Chargement mis en cache pour ne pas recharger à chaque interaction ---
@st.cache_resource
def charger_modele():
    modele = joblib.load("models/random_forest_final.pkl")
    with open("models/random_forest_final_meta.json") as f:
        meta = json.load(f)
    return modele, meta

@st.cache_data
def charger_donnees():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    return train_df, test_df

modele, meta = charger_modele()
train_df, test_df = charger_donnees()
X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]
y_proba_test = modele.predict_proba(X_test)[:, 1]

st.title("Dashboard — Prédiction du churn client")

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("Taux de churn (global)", f"{train_df['Churn'].mean():.1%}")
col2.metric("AUC (validation croisée)", f"{meta['cv_auc_mean']:.3f}")
col3.metric("Clients (jeu de test)", f"{len(test_df)}")

st.divider()

# --- Exploration du churn par variable ---
st.subheader("Taux de churn par variable")
cat_vars = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen",
            "Partner", "Dependents", "TechSupport", "OnlineSecurity", "gender"]
variable = st.selectbox("Choisir une variable", cat_vars)

taux = train_df.groupby(variable)["Churn"].mean().sort_values(ascending=False)
fig1 = go.Figure(go.Bar(x=taux.index.astype(str), y=taux.values, marker_color=COULEUR_CHURN,
                         hovertemplate="%{x}<br>Taux de churn : %{y:.1%}<extra></extra>"))
fig1.update_layout(yaxis_tickformat=".0%", template="plotly_white", height=400)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# --- Seuil de décision interactif : le point fort de Streamlit vs un notebook statique ---
st.subheader("Effet du seuil de décision (jeu de test)")
seuil = st.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05)

y_pred_s = (y_proba_test >= seuil).astype(int)
colA, colB, colC = st.columns(3)
colA.metric("Précision", f"{precision_score(y_test, y_pred_s, zero_division=0):.1%}")
colB.metric("Rappel", f"{recall_score(y_test, y_pred_s):.1%}")
colC.metric("F1", f"{f1_score(y_test, y_pred_s):.3f}")

cm = confusion_matrix(y_test, y_pred_s)
fig_cm = go.Figure(data=go.Heatmap(
    z=cm, x=["Prédit : Reste", "Prédit : Churn"], y=["Réel : Reste", "Réel : Churn"],
    colorscale=[[0, "#fcfcfb"], [1, COULEUR_CHURN]], showscale=False,
    text=cm, texttemplate="%{text}",
))
fig_cm.update_layout(template="plotly_white", height=350)
st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

# --- Importance des variables ---
st.subheader("Variables les plus importantes")
noms_colonnes = modele.named_steps["prep"].get_feature_names_out()
importances = pd.Series(
    modele.named_steps["clf"].feature_importances_, index=noms_colonnes
).sort_values(ascending=True).tail(10)
fig3 = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h", marker_color=COULEUR_CHURN))
fig3.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# --- Simulateur : prédire le churn d'un client fictif ---
st.subheader("Simulateur : estimer le risque de churn d'un client")
with st.form("simulateur"):
    col1, col2 = st.columns(2)
    with col1:
        contract = st.selectbox("Type de contrat", sorted(train_df["Contract"].unique()))
        internet = st.selectbox("Service internet", sorted(train_df["InternetService"].unique()))
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
    with col2:
        monthly = st.slider("Charges mensuelles (€)", 18.0, 120.0, 70.0)
        payment = st.selectbox("Mode de paiement", sorted(train_df["PaymentMethod"].unique()))
    valider = st.form_submit_button("Prédire")

if valider:
    # On part d'un client "gabarit" du test set et on écrase les champs choisis
    client = X_test.iloc[[0]].copy()
    client["Contract"] = contract
    client["InternetService"] = internet
    client["tenure"] = tenure
    client["MonthlyCharges"] = monthly
    client["PaymentMethod"] = payment
    client["TotalCharges"] = tenure * monthly  # approximation simple

    proba = modele.predict_proba(client)[0, 1]
    st.metric("Probabilité de churn estimée", f"{proba:.1%}")
    if proba >= 0.5:
        st.warning("Client à risque — une action de rétention est recommandée.")
    else:
        st.success("Client jugé stable.")