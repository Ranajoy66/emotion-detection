# app.py
import streamlit as st
import os
import json
import pandas as pd
import random
import requests, base64
import io

from questions import questions
from scripts.predict_lr import predict_with_probs  # MiniLM version

# ===== Streamlit Config =====
st.set_page_config(
    page_title="Depression Counselling Survey",
    layout="centered",
    page_icon="🧠"
)

# Hide sidebar completely
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stAppViewBlockContainer"] {
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ===== Background + Base Layout ===== */
.stApp {
    background: linear-gradient(135deg, #f7fbff 0%, #d4e9ff 50%, #a5d4ff 100%);
    color: #00264d;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== Headings ===== */
h1, h2, h3, h4, h5, h6 {
    color: #004a9f !important;
    font-weight: 700;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
}

/* ===== General Text ===== */
p, span, label, div {
    color: #002b5b !important;
}

/* ===== Text Inputs & Text Areas ===== */
textarea, input[type="text"], input[type="email"], input[type="password"], input[type="number"] {
    background-color: rgba(255,255,255,0.95) !important;
    border: 1px solid #b3dbff !important;
    border-radius: 10px !important;
    padding: 0.6em !important;
    color: #002b5b !important;
    box-shadow: inset 0 1px 4px rgba(0, 102, 204, 0.1);
    caret-color: #004a9f !important; /* Blinker color */
}
textarea:focus, input[type="text"]:focus {
    border-color: #0091ff !important;
    box-shadow: 0 0 6px rgba(0,145,255,0.4);
}

/* ===== Placeholder Styling ===== */
input::placeholder, textarea::placeholder {
    color: #335b85 !important; /* Readable muted navy */
    opacity: 0.9 !important;
}

/* ===== Buttons ===== */
.stButton>button {
    background: linear-gradient(90deg, #007bff, #00b4ff);
    color: white !important;
    border-radius: 30px !important;
    padding: 0.7em 1.7em !important;
    font-weight: 600;
    border: none !important;
    transition: all 0.3s ease;
    box-shadow: 0 3px 10px rgba(0, 102, 255, 0.25);
}
.stButton>button:hover {
    background: linear-gradient(90deg, #005ecb, #00a3e6);
    transform: scale(1.05);
    box-shadow: 0 6px 15px rgba(0, 102, 255, 0.35);
}

/* ===== Alerts ===== */
[data-testid="stAlert"] {
    border-radius: 15px !important;
    padding: 1em !important;
    border-left: 6px solid #0091ff !important;
    background-color: rgba(255,255,255,0.9) !important;
    box-shadow: 0 3px 12px rgba(173, 216, 230, 0.4);
}
[data-testid="stAlert-success"] {
    border-left-color: #00c292 !important;
    background: linear-gradient(90deg, rgba(220,255,239,0.8), rgba(255,255,255,0.8));
}
[data-testid="stAlert-error"] {
    border-left-color: #ff4d4f !important;
    background: linear-gradient(90deg, rgba(255,230,230,0.8), rgba(255,255,255,0.8));
}
[data-testid="stAlert-warning"] {
    border-left-color: #ffc107 !important;
    background: linear-gradient(90deg, rgba(255,250,204,0.8), rgba(255,255,255,0.8));
}
[data-testid="stAlert-info"] {
    border-left-color: #0091ff !important;
    background: linear-gradient(90deg, rgba(204,238,255,0.8), rgba(255,255,255,0.8));
}

/* ===== DataFrames & Charts ===== 
.stDataFrame, .stPlotlyChart, .stAltairChart, .stVegaLiteChart, .stPyplotChart {
    background: rgba(255,255,255,0.95);
    border-radius: 20px;
    padding: 1em;
    box-shadow: 0 4px 20px rgba(0, 0, 128, 0.15);
    margin-top: 1em;
}
*/

/* ===== Section Headers ===== */
.stMarkdown h2, .stMarkdown h3 {
    border-left: 5px solid #007bff;
    padding-left: 10px;
}

/* ===== Footer Hide ===== */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ===== Fix dark download and options toolbar ===== */

/* Pop-up menu (sort, pin, hide, etc.) */
[data-testid="stDataFrame"] [role="menu"], 
[data-testid="stDataFrame"] [data-baseweb="popover"] {
    background-color: #ffffff !important;
    color: #004a9f !important;
    border: 1px solid #b3dbff !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 10px rgba(0, 102, 255, 0.15) !important;
}

/* Menu items (sort ascending, etc.) */
[data-testid="stDataFrame"] [role="menuitem"] {
    color: #004a9f !important;
    font-weight: 500 !important;
    background: transparent !important;
    font-size: 0.95rem !important;
}

[data-testid="stDataFrame"] [role="menuitem"]:hover {
    background-color: #e6f2ff !important;
    color: #002b5b !important;
    border-radius: 6px !important;
}

/* Search/filter input inside menu */
[data-testid="stDataFrame"] [role="menu"] input {
    background-color: #f6fbff !important;
    color: #004a9f !important;
    border: 1px solid #80bfff !important;
}

/* Icon buttons (Download, fullscreen, etc.) */
[data-testid="StyledFullScreenButton"],
[data-testid="StyledDownloadButton"] {
    background-color: #ffffff !important;
    color: #004a9f !important;
    border-radius: 8px !important;
    border: 1px solid #b3dbff !important;
    box-shadow: 0 2px 5px rgba(0, 102, 255, 0.2);
    transition: all 0.3s ease;
    font-weight: 600 !important;
}

[data-testid="StyledFullScreenButton"]:hover,
[data-testid="StyledDownloadButton"]:hover {
    background-color: #e8f3ff !important;
    color: #002b5b !important;
    transform: scale(1.05);
}

/* Fix icons inside buttons */
button svg {
    fill: #004a9f !important;
    stroke: #004a9f !important;
}
button:hover svg {
    fill: #002b5b !important;
    stroke: #002b5b !important;
}

/* Tooltip that appears when hovering on icons */
[data-baseweb="tooltip"] {
    background-color: #ffffff !important;
    color: #004a9f !important;
    border: 1px solid #b3dbff !important;
    box-shadow: 0 4px 15px rgba(0, 102, 255, 0.25);
    font-size: 0.85rem !important;
}
div[data-testid="stButton"][key="logout"] > button {
  background: linear-gradient(90deg, #ff416c, #ff4b2b) !important; /* red-pink gradient */
  color: white !important;
  font-weight: 800 !important;
  border-radius: 40px !important;
  padding: 0.8em 2em !important;
  font-size: 1.1rem !important;
  box-shadow: 0 4px 20px rgba(255, 65, 108, 0.3) !important;
  transition: all 0.3s ease !important;
}
            
div[data-testid="stButton"][key="start_btn"] > button:hover {
  background: linear-gradient(90deg, #ff4b2b, #ff416c) !important;
  transform: scale(1.05) !important;
  box-shadow: 0 6px 25px rgba(255, 75, 43, 0.45) !important;
}
        div.stButton > button,
        div.stButton > button p,
        div[data-testid="stButton"] > button,
        div[data-testid="stButton"] > button p {
            color: white !important;
        }
</style>
""", unsafe_allow_html=True)


# ===== Paths =====
RESULTS_CSV = "data/session_results.csv"
os.makedirs("data", exist_ok=True)

# ===== GitHub Config =====
GITHUB_REPO = "Ranajoy66/emotion-detection"
FILE_PATH = "data/session_results.csv"  # path inside repo
BRANCH = "main"
TOKEN = st.secrets["GITHUB_TOKEN"]  # add to .streamlit/secrets.toml

# ===== Functions =====
def update_csv_on_github(df_new):
    """Upload session_results.csv to GitHub safely, avoid duplicates"""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {TOKEN}"}

    # Check if file exists
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        data = res.json()
        sha = data["sha"]
        csv_content = base64.b64decode(data["content"]).decode()
        df_old = pd.read_csv(io.StringIO(csv_content))
        df_final = pd.concat([df_old, df_new], ignore_index=True)
        # Remove duplicate PatientID rows, keep the last submission
        df_final = df_final.drop_duplicates(subset=["PatientID"], keep="last")
    else:
        sha = None
        df_final = df_new

    # Convert to CSV
    csv_data = df_final.to_csv(index=False)
    encoded = base64.b64encode(csv_data.encode()).decode()

    # Commit
    commit_data = {
        "message": "Update session results from Streamlit app",
        "content": encoded,
        "branch": BRANCH,
    }
    if sha:
        commit_data["sha"] = sha

    upload = requests.put(url, headers=headers, json=commit_data)

    if upload.status_code in [200, 201]:
        st.success("✅ Data successfully saved to GitHub!")
    else:
        st.error(f"❌ GitHub update failed: {upload.json()}")


# ===== Load Metrics =====
METRICS_PATH = "models/minilm_emotion/metrics.json"
with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

classes = metrics["classes"]

# ===== Streamlit Config =====
st.set_page_config(page_title="Depression Counselling Survey", layout="centered")
st.title("🧠 Interactive Emotion Counselling")

def redirect_to_home():
    st.switch_page("Home.py")

col1, col2, col3 = st.columns([2, 2, 1])
with col3:
    if st.button("Logout", key="logout"):
        redirect_to_home()

# ===== Patient Info =====
patient_id = st.text_input("Enter Patient ID:", "")
if not patient_id:
    st.warning("Please enter a Patient ID to start.")
    st.stop()

# ===== Session State Initialization =====
if "q_index" not in st.session_state:
    st.session_state.q_index = 0
    st.session_state.responses = []
    st.session_state.predictions = []
    st.session_state.probabilities = []
    st.session_state.questions_subset = random.sample(questions, 10)
    st.session_state.uploaded_to_github = False  # ✅ prevent repeated upload

# ===== Ask Random Questions =====
if st.session_state.q_index < len(st.session_state.questions_subset):
    question = st.session_state.questions_subset[st.session_state.q_index]
    st.subheader(f"Q{st.session_state.q_index + 1}: {question}")
    answer = st.text_area("Your response:", key=f"ans_{st.session_state.q_index}")

    if st.button("Next Question"):
        if answer.strip():
            pred, probs, _ = predict_with_probs(answer)
            st.session_state.responses.append(answer)
            st.session_state.predictions.append(pred)
            st.session_state.probabilities.append(probs)
            st.session_state.q_index += 1
            st.rerun()  # now safe since upload flag prevents duplicates
        else:
            st.warning("Please provide an answer before continuing.")

else:
    st.success("✅ Survey completed!")

    # ===== Aggregate Results =====
    prob_df = pd.DataFrame(st.session_state.probabilities)
    mean_probs = prob_df.mean().to_dict()
    mean_probs = {cls: round(mean_probs.get(cls, 0) * 100, 2) for cls in classes}

    session_result = {"PatientID": patient_id}
    session_result.update(mean_probs)
    new_df = pd.DataFrame([session_result])

    # ===== Save to GitHub once =====
    if not st.session_state.uploaded_to_github:
        update_csv_on_github(new_df)
        st.session_state.uploaded_to_github = True

    # ===== Show Results =====
    st.subheader("📊 Your Session Summary")
    st.write(session_result)
    st.bar_chart(pd.DataFrame([mean_probs]).T.rename(columns={0: "Probability %"}))

    st.info("Session saved successfully! You can now view it in GitHub / Power BI.")
