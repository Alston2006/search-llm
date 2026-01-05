import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# -------------------- CONFIG --------------------
load_dotenv()

st.set_page_config(
    page_title="TheraLink AI",
    page_icon="🧠",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 1.5rem;
}
.card {
    background: white;
    padding: 1.2rem;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}
.badge-green {
    color: white;
    background: #2ecc71;
    padding: 0.3rem 0.7rem;
    border-radius: 12px;
}
.badge-yellow {
    color: black;
    background: #f1c40f;
    padding: 0.3rem 0.7rem;
    border-radius: 12px;
}
.badge-red {
    color: white;
    background: #e74c3c;
    padding: 0.3rem 0.7rem;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- GEMINI + LANGCHAIN --------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2
)

# -------------------- QUESTIONS --------------------
QUESTIONS = [
    "How stable was your mood today?",
    "How intense was your anxiety today?",
    "How well could you regulate your emotions?",
    "How irritable did you feel?",
    "How calm did you feel overall?",
    "How frequent were negative thoughts?",
    "How much did you overthink today?",
    "How focused were you?",
    "How difficult was decision-making?",
    "How clear did your thinking feel?",
    "How would you rate your sleep quality?",
    "Were you satisfied with your sleep duration?",
    "How was your energy level today?",
    "How was your appetite?",
    "Did you feel physical tension or restlessness?",
    "How motivated did you feel?",
    "How socially connected did you feel?",
    "How stressful was work/study today?",
    "Did you feel overwhelmed?",
    "Overall mental well-being today?"
]

# -------------------- SESSION STATE INIT --------------------
if "patients" not in st.session_state:
    st.session_state["patients"] = {}

if "selected_patient" not in st.session_state:
    st.session_state["selected_patient"] = None

# -------------------- SIDEBAR --------------------
st.sidebar.title("🧠 TheraLink AI")
role = st.sidebar.radio("Select Role", ["Patient", "Doctor"])

# -------------------- PATIENT VIEW --------------------
if role == "Patient":
    st.title("📝 Daily 2-Minute Mental Health Check-In")
    st.caption("This tool does not provide therapy or advice.")

    patient_id = st.text_input("Patient ID (Demo)", value="patient_001")

    progress = st.progress(0)
    responses = []

    for i, q in enumerate(QUESTIONS):
        val = st.slider(q, 1, 5, 3)
        responses.append(val)
        progress.progress((i + 1) / len(QUESTIONS))

    if st.button("✅ Submit Check-In"):
        score = round(np.mean(responses), 2)
        date = pd.Timestamp.today().date()

        if patient_id not in st.session_state["patients"]:
            st.session_state["patients"][patient_id] = {
                "history": [],
                "notes": "",
                "verified": False
            }

        st.session_state["patients"][patient_id]["history"].append({
            "Date": date,
            "Score": score,
            "Responses": responses
        })

        st.success("Check-in submitted successfully.")

# -------------------- DOCTOR VIEW --------------------
if role == "Doctor":
    st.title("🏥 Clinic Dashboard")

    patients = st.session_state["patients"]

    if not patients:
        st.warning("No patient data available.")
    else:
        # -------- MULTI-PATIENT OVERVIEW --------
        overview = []

        for pid, pdata in patients.items():
            last = pdata["history"][-1]
            overview.append({
                "Patient ID": pid,
                "Latest Score": last["Score"],
                "Risk": "🔴" if last["Score"] <= 2 else "🟡" if last["Score"] <= 3.5 else "🟢"
            })

        overview_df = pd.DataFrame(overview)

        st.markdown("### 📋 Patient Triage Overview")
        st.dataframe(overview_df, use_container_width=True)

        selected = st.selectbox(
            "Select Patient",
            overview_df["Patient ID"]
        )

        st.session_state["selected_patient"] = selected
        pdata = patients[selected]
        history = pd.DataFrame(pdata["history"])

        st.markdown("---")
        st.markdown(f"## 🧠 Patient: `{selected}`")

        # -------- CURRENT STATUS --------
        latest_score = history.iloc[-1]["Score"]

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("### 📈 Wellness Trend")
            st.line_chart(history.set_index("Date")["Score"])

        with col2:
            st.markdown("### 🚦 Risk Status")
            if latest_score <= 2:
                st.markdown("<span class='badge-red'>RED ZONE</span>", unsafe_allow_html=True)
            elif latest_score <= 3.5:
                st.markdown("<span class='badge-yellow'>YELLOW ZONE</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge-green'>GREEN ZONE</span>", unsafe_allow_html=True)
            st.metric("Wellness Score", latest_score)

        # -------- AI SUMMARY --------
        st.markdown("### 🧾 Session Prep Summary")

        prompt = PromptTemplate(
            input_variables=["score"],
            template="""
You are an AI clinical assistant supporting a licensed mental health professional.

Rules:
- No diagnosis
- No advice
- Neutral language

Data:
Average wellness score today: {score}

Task:
Generate a short 110 words factual summary for therapist preparation.
"""
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        summary = chain.run(score=latest_score)
        st.info(summary)

        # -------- DOCTOR VERIFICATION --------
        st.markdown("### 🩺 Clinician Verification")
        verify = st.toggle("Verify AI Alert")

        pdata["verified"] = verify
        if verify:
            st.success("Verified by clinician")
        else:
            st.info("Awaiting verification")

        # -------- DOCTOR NOTES --------
        st.markdown("### 🧑‍⚕️ Doctor Notes (Manual)")
        notes = st.text_area(
            "Clinical observations & session notes",
            value=pdata["notes"],
            height=150
        )
        pdata["notes"] = notes

        st.markdown("---")
        st.caption("TheraLink AI — Human-Centric Clinical Triage (Prototype)")


