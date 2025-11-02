import streamlit as st 
import time 
from github_utils import update_github_db_and_csv, DB_PATH 
import sqlite3 
 
st.set_page_config(page_title="Login / Register - Depression Analysis", page_icon="💬", layout="centered") 
 
# ==================== DATABASE LOGIN FUNCTION ==================== # 
def login_user(email, password): 
 conn = sqlite3.connect('data/users.db') 
 c = conn.cursor() 
 c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password)) 
 data = c.fetchone() 
 conn.close() 
 if data: 
    user_dict = {"name": data[0], "email": data[1], "password": data[2]} 
    return user_dict 
 
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, #d0e7ff 50%, #a9d6ff 100%);
            color: #002b5b;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #004a9f !important;
        }
        label, p, span {
            color: #002b5b !important;
        }
        input, textarea {
            background-color: #f6fbff !important;
            color: #002b5b !important;
            border: 1px solid #70b5ff !important;
            border-radius: 10px !important;
            caret-color: #004a9f !important; /* Blinker color */
        }
        .stTextInput>div>div>input {
            background-color: #f6fbff;
            color: #002b5b;
            border: 1px solid #70b5ff;
            border-radius: 10px;
            padding: 0.6em;
        }
        .stButton>button {
            background: linear-gradient(90deg, #0066ff, #00aaff);
            color: white !important;
            border-radius: 25px !important;
            padding: 0.5em 2em !important;
            border: none !important;
            font-weight: bold;
            transition: 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #004aad, #0097e6);
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0, 102, 255, 0.4);
        }
        .tab-header {
            text-align: center;
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 1em;
            color: #003366;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .success-box {
            border: 1px solid #00aaff;
            border-radius: 10px;
            padding: 1em;
            background-color: #e8f5ff;
            color: #003366;
            text-align: center;
            margin-top: 1em;
        }
        .stTabs [role="tablist"] {
            justify-content: center;
            border-bottom: 2px solid #80bfff;
        }
        .stTabs [role="tab"] {
            color: #003366 !important;
            font-weight: bold;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            color: #004aad !important;
        }
        div[data-testid="stButton"][key="start_btn"] > button {
        background: linear-gradient(90deg, #ff416c, #ff4b2b) !important; /* red-pink gradient */
        color: white !important;
        font-weight: 800 !important;
        border-radius: 40px !important;
        padding: 0.8em 2em !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 20px rgba(255, 65, 108, 0.3) !important;
        transition: all 0.3s ease !important;
        }
        div.stButton > button,
        div.stButton > button p,
        div[data-testid="stButton"] > button,
        div[data-testid="stButton"] > button p {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)
 
# ==================== SESSION VARIABLES ==================== # 
if "users" not in st.session_state: 
 st.session_state.users = {} 
 
if "current_user" not in st.session_state: 
 st.session_state.current_user = None 
 
# <CHANGE> Added tab_selection to track which tab should be active
if "tab_selection" not in st.session_state: 
 st.session_state.tab_selection = "login"  # "login" or "register"
 
# <CHANGE> Changed prefill to store email only (removed password for security)
if "prefill_email" not in st.session_state: 
 st.session_state.prefill_email = "" 
 
# ==================== HEADER ==================== # 
st.markdown("<div class='tab-header'> 🧠 Welcome to Depression Analysis </div>", unsafe_allow_html=True) 
def redirect_to_home():
    st.switch_page("Home.py")

col1, col2, col3 = st.columns([2, 2, 1])
with col3:
    if st.button("<- Home", key="home"):
        redirect_to_home()
st.markdown(" Please log in or create an account to continue your mental wellness journey. ", unsafe_allow_html=True) 
 
# <CHANGE> Created custom tab buttons using columns instead of st.tabs()
col1, col2 = st.columns(2)
with col1:
    if st.button("🔑 Login", use_container_width=True, key="btn_login"):
        st.session_state.tab_selection = "login"
        st.rerun()

with col2:
    if st.button("📝 Register", use_container_width=True, key="btn_register"):
        st.session_state.tab_selection = "register"
        st.rerun()

st.divider()

# ==================== LOGIN TAB ==================== # 
# <CHANGE> Use conditional rendering based on tab_selection instead of st.tabs()
if st.session_state.tab_selection == "login":
 st.subheader("Login to your account") 
 
 # <CHANGE> Prefill email from session state, remove password from prefill
 email = st.text_input("Email", key="login_email", value=st.session_state.prefill_email) 
 password = st.text_input("Password", type="password", key="login_pass") 
 
 if st.button("Login", use_container_width=True):
    user = login_user(email, password)
    if user and user["password"] == password: 
        st.session_state.current_user = email 
        st.success(f"Welcome back, {user['name']}! 💙") 
        st.markdown(" Redirecting you to your chat... ", unsafe_allow_html=True) 
        time.sleep(1.5) 
        st.switch_page("pages/app.py") 
    else: 
        st.error("Invalid email or password. Please try again.") 
 
# ==================== REGISTER TAB ==================== # 
# <CHANGE> Use conditional rendering based on tab_selection instead of st.tabs()
elif st.session_state.tab_selection == "register":
 st.subheader("Create a new account") 
 fullname = st.text_input("Full Name", key="reg_name") 
 email_reg = st.text_input("Email", key="reg_email") 
 password_reg = st.text_input("Password", type="password", key="reg_pass") 
 cpass = st.text_input("Confirm Password", type="password", key="reg_conf") 
 token = st.secrets["GITHUB_TOKEN"] 
 
 if st.button("Register", use_container_width=True):
    if not fullname or not email_reg or not password_reg:
        st.error("Please fill in all fields.")
    elif password_reg != cpass:
        st.error("Passwords do not match.")
    elif email_reg in st.session_state.users:
        st.error("Email already registered. Please log in.")
    else:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute(
                'INSERT INTO users (fullname, email, password, cpass) VALUES (?, ?, ?, ?)',
                (fullname, email_reg, password_reg, cpass)
            )
            conn.commit()
            st.success("🎉 Account created successfully!")

            if token:
                update_github_db_and_csv(token)

            # <CHANGE> After successful registration, prefill login info & switch tab
            st.session_state.prefill_email = email_reg
            st.session_state.tab_selection = "login"
            st.info("Redirecting you to Login tab...")
            time.sleep(1)
            st.rerun()

        except sqlite3.IntegrityError:
            st.error("⚠️ Email already exists in the database. Please try logging in.")

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")

        finally:
            conn.close()

# ... existing code ...

# ==================== FOOTER ==================== #
st.markdown("""
    <hr style="border: 0.5px solid #80bfff; margin-top: 2em;">
    <p style='text-align:center; color:#004aad;'>© 2025 RERF | Depression Analysis Assistant 💬</p>
""", unsafe_allow_html=True)
