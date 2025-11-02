import streamlit as st
import base64
from PIL import Image

st.set_page_config(
    page_title="Emotion Analysis | Home", page_icon="🧠", layout="wide"
)

# Hide sidebar completely
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stAppViewBlockContainer"] {
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# --- 🌈 Combined White–Blue Theme CSS ---
st.markdown(
    """
    <style>
        /* Background + base text */
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, #d0e7ff 50%, #a9d6ff 100%);
        }
        button:has-text("Let's Chat 💬") {
            color: white !important;
            background: linear-gradient(90deg, #0066ff, #00aaff) !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #004a9f !important;
        }
        label, p, span {
            color: #002b5b !important;
        }

        div.stButton > button,
        div.stButton > button p,
        div[data-testid="stButton"] > button,
        div[data-testid="stButton"] > button p {
            color: white !important;
        }


        /* Navbar Buttons */
        div.stButton > button,
        div[data-testid="stButton"][key="login_nav"] > button {
            background: linear-gradient(90deg, #0066ff, #00aaff) !important;
            color: white !important;
            border-radius: 25px !important;
            padding: 0.5em 1.5em !important;
            border: none !important;
            font-weight: 700 !important;
            transition: 0.3s ease-in-out !important;
            text-decoration: none !important;
            display: inline-block !important; /* so it matches .login-button anchors */
            width: auto !important; /* override use_container_width if you want auto */
        }

        /* hover */
        div.stButton > button:hover,
        div[data-testid="stButton"] > button:hover {
            background: linear-gradient(90deg, #004aad, #0097e6) !important;
            transform: scale(1.05) !important;
            box-shadow: 0 4px 15px rgba(0, 102, 255, 0.4) !important;
            color: white !important;
        }

        /* keep your HTML link buttons consistent */
        .login-button a {
            background: linear-gradient(90deg, #0066ff, #00aaff) !important;
            color: white !important;
            border-radius: 25px !important;
            padding: 0.5em 1.5em !important;
            border: none !important;
            font-weight: bold !important;
            transition: 0.3s ease-in-out !important;
            text-decoration: none !important;
            display: inline-block !important;
            margin-bottom: 1em !important;
        }
        .login-button a:hover {
            background: linear-gradient(90deg, #004aad, #0097e6) !important;
            transform: scale(1.05) !important;
            box-shadow: 0 4px 15px rgba(0, 102, 255, 0.4) !important;
            color: white !important;
        }

        /* Navbar Layout */
        .nav-bar {
            display: flex;
            justify-content: flex-end;
            margin: 1em 2em 0 0;
            gap: 1em;
        }

        /* Page Titles */
        .big-title {
            font-size: 4em;
            font-weight: bold;
            color: #003366;
            text-align: center;
            padding-top: 1em;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
            animation: fadeIn 2s ease-in-out;
        }
        .subtitle {
            font-size: 1.5em;
            color: #004a9f;
            text-align: center;
            margin-top: 1em;
            animation: fadeIn 3s ease-in-out;
        }

        /* Center Button */
        .center-button {
            display: flex;
            justify-content: center;
            margin-top: 2em;
            animation: fadeIn 4s ease-in-out;
        }
        .button-style button {
            font-size: 1.2em;
            padding: 0.75em 2em;
            border-radius: 40px;
            background: linear-gradient(90deg, #0066ff, #00aaff);
            border: none;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .button-style button:hover {
            background: linear-gradient(90deg, #004aad, #0097e6);
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0, 102, 255, 0.4);
        }

        /* Image + Animation - MOBILE */
       /* 🌐 Responsive layout fix for mobile screens */
        @media (max-width: 900px) {
            /* Stack the image and text vertically */
            .fade-in-section {
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
                text-align: center !important;
            }

            /* Fix spacing between image and text */
            .fade-in-section img {
                max-width: 90% !important;
                height: auto !important;
                margin-bottom: 1.5em !important;
            }

            /* Adjust text box alignment */
            .fade-in-section div {
                margin: 0 !important;
                padding: 0 1em !important;
                text-align: center !important;
            }

            /* Keep text nicely readable */
            .fade-in-section p {
                width: 90% !important;
                margin: auto !important;
                line-height: 1.5em !important;
                white-space: normal !important;
            }
        }

        .image-bounce {
            animation: bounceIn 2s ease-in-out;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-radius: 12px;
        }
        .image-bounce:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 20px rgba(0, 0, 128, 0.2);
        }
        @keyframes fadeInUp {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes bounceIn {
            0% { transform: scale(0.9); opacity: 0; }
            60% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(1); }
        }
        footer {visibility: hidden;}
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* ========== Shared button look (desktop) ========== */
div.stButton > button,
div[data-testid="stButton"] > button {
  background: linear-gradient(90deg, #0066ff, #00aaff) !important;
  color: #fff !important;
  border-radius: 25px !important;
  padding: 0.5em 1.5em !important;
  border: none !important;
  font-weight: 700 !important;
  transition: transform .18s ease, box-shadow .18s ease !important;
  display: inline-block !important;
  width: auto !important;            /* desktop: auto width */
  box-shadow: none !important;
}

/* hover for st.button */
div.stButton > button:hover,
div[data-testid="stButton"] > button:hover {
  background: linear-gradient(90deg, #004aad, #0097e6) !important;
  transform: scale(1.03) !important;
  box-shadow: 0 6px 18px rgba(0,102,255,0.25) !important;
  color: #fff !important;
}

/* ========== HTML link buttons you already use ========== */
.login-button a {
  background: linear-gradient(90deg, #0066ff, #00aaff) !important;
  color: #fff !important;
  border-radius: 25px !important;
  padding: 0.5em 1.5em !important;
  border: none !important;
  font-weight: 700 !important;
  display: inline-block !important;
  text-decoration: none !important;
  transition: transform .18s ease, box-shadow .18s ease !important;
}
.login-button a:hover {
  background: linear-gradient(90deg, #004aad, #0097e6) !important;
  transform: scale(1.03) !important;
  box-shadow: 0 6px 18px rgba(0,102,255,0.25) !important;
  color: #fff !important;
}

/* ========== Responsive behaviour (MOBILE) ========== */
@media (max-width: 900px) {

  /* Stack your top "columns" visually and center them */

  .stContainer .stColumns, /* fallback generic */
  .stContainer > div[role="group"] {
      display: flex !important;
      flex-direction: column !important;
      gap: 0.6rem !important;
      align-items: center !important;
  }

  /* Make buttons and link anchors full width and touch friendly on small screens */
  div.stButton > button,
  div[data-testid="stButton"] > button,
  .login-button a,
  .button-style button,
  .center-button a button {
      width: 90% !important;
      max-width: 420px !important;
      padding: 0.8em 1.2em !important;
      font-size: 1.05rem !important;
      display: block !important;
      text-align: center !important;
    }

  /* scale down big headings on mobile */
  .big-title { font-size: 2.2rem !important; line-height: 1.05 !important; padding-top: .8rem !important; }
  .subtitle { font-size: 1rem !important; }

  /* responsive images */
  img.image-bounce { max-width: 92% !important; height: auto !important; }

  /* make the main content padding smaller on mobile */
  [data-testid="stAppViewBlockContainer"] { padding-left: 1rem !important; padding-right: 1rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# 🎨 Unique Style for "Start Your Journey" Button
st.markdown("""
<style>
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

div[data-testid="stButton"][key="start_btn"] > button:hover {
  background: linear-gradient(90deg, #ff4b2b, #ff416c) !important;
  transform: scale(1.05) !important;
  box-shadow: 0 6px 25px rgba(255, 75, 43, 0.45) !important;
}
</style>
""", unsafe_allow_html=True)


def redirect_to_login():
    st.switch_page("pages/Login.py")


# --- Top Navigation Bar ---
st.markdown('<div class="top-nav-wrapper">', unsafe_allow_html=True)

with st.container():
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col1:
        if st.button("Let's Chat 💬", key="login_nav", use_container_width=True):
            redirect_to_login()
    with col3:
        st.markdown(
            "<div class='login-button'><a href='#videos'>Video Chat</a></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            "<div class='login-button'><a href='#voice'>Voice Chat</a></div>",
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            "<div class='login-button'><a href='#contact'>Contact Us</a></div>",
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


# --- Big heading and subtitle ---
st.markdown(
    """
    <div class="big-title">🧠 Emotion Analysis</div>
    <div class="subtitle">Your mindful AI companion for emotional well-being.<br />Powered by Mindful AI.</div>
""",
    unsafe_allow_html=True,
)

# --- CTA Button ---


# center with columns (guaranteed)
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    if st.button("Start Your Journey", key="start_btn"):
        redirect_to_login()



# --- Intro Section ---
st.markdown(
    """
    <div id="start" class="fade-in-section" style="margin-top: 8em;">
        <h2 style="color:#004aad; text-align:center">Welcome to the Mindful AI Experience</h2>
        <p style="color:#002b5b; text-align:center; max-width: 700px; margin: auto;">
            This app helps you understand your emotions and provides guidance through text, audio, and video conversations.
            Powered by ## — a machine learning model trained to support your mental wellness journey.
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

# --- Mindful AI Section ---
st.markdown(
    """
    <div id="login" class="fade-in-section" style="margin-top: 5em; text-align: center;">
        <h3 style="color:#004aad;">Mindful AI</h3>
        <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 1em; margin-top: 2em;">
            <div>
                <img src="https://static.vecteezy.com/system/resources/thumbnails/022/756/603/small/conceptual-image-of-a-human-head-with-colorful-brain-and-autumn-leaves-mental-health-concept-ai-generated-artwork-photo.jpg" alt="image" class="image-bounce" style="max-width: 300px;"/>
                <p>Mind is a powerful thing.</p>
            </div>
            <div>
                <img src="https://t4.ftcdn.net/jpg/02/71/09/15/360_F_271091539_7fVM1L9nhbpHlQAbEs8ZIrAEPAd9PxlM.jpg" alt="image" class="image-bounce" style="max-width: 300px;"/>
                <p>Be mindful of your thoughts.</p>
            </div>
            <div>
                <img src="https://psychologyfanatic.com/wp-content/uploads/2024/01/Psychological-Disorders.-Psychology-Fanatic-article-feature-image.webp" alt="image" class="image-bounce" style="max-width: 300px;"/>
                <p>Take care of your mental health.</p>
            </div>
        </div>
        <p style="margin-top: 2em;">Your mindful AI companion. Share and explore your thoughts.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# --- Chat Modes (Text / Audio / Video) ---
st.markdown(
    """
    <div id="text" class="fade-in-section" style="display: flex; justify-content: space-between; align-items: center; margin-top: 5em;">
        <img src="https://tse2.mm.bing.net/th/id/OIP.ZVVA9mmlcNAYlLRJiSayIwHaHa?cb=thfc1&rs=1&pid=ImgDetMain&o=7&rm=3" alt="image" class="image-bounce" style="max-width: 300px;"/>
        <div style="margin-left: 2em; text-align: center;">
            <h3 style="color:#004aad;">Text Chat</h3>
            <p>
                Express your thoughts freely through text. Our AI companion listens without judgment and provides mindful, empathetic responses
                that help you reflect on your emotions. Whether you’re journaling your day or exploring your mood patterns, text-based reflection
                can guide you toward self-awareness and emotional clarity.
            </p>
        </div>
    </div>            

    <div id="voice" class="fade-in-section" style="display: flex; justify-content: space-between; align-items: center; margin-top: 5em;">
        <div style="margin-right: 2em; margin-bottom: 2em; text-align: center;">
            <h3 style="color:#004aad;">Audio Chat</h3>
            <p>
                Sometimes words flow better when spoken. With audio chat, you can talk naturally and let your feelings out loud.
                The AI will listen, analyze tone and pace, and gently guide you with calming, reflective feedback.
                Ideal for moments when typing feels too heavy and you just want someone—or something—to listen.
            </p>
        </div>
        <img src="https://media.istockphoto.com/id/1300081009/vector/woman-talking-with-a-counselor-or-therapist-using-her-mobile-phone-online-psychotherapy.jpg?s=1024x1024&w=is&k=20&c=iqjiycxG7z3mZVWUPMwimSQ3gVKqPNZmEiewiRx60I0=" alt="image" class="image-bounce" style="max-width: 400px;"/>
    </div>            

    <div id="videos" class="fade-in-section" style="display: flex; justify-content: space-between; align-items: center; margin-top: 5em;">
        <img src="https://media.istockphoto.com/id/1212577571/vector/person-holding-smartphone-and-calling-via-video-chat-mobile-app.jpg?s=612x612&w=0&k=20&c=flloizEAzFR6wQlsl1EAK_nbpSfYJIg-KnHcTB2-OWc=" alt="image" class="image-bounce" style="max-width: 300px;"/>
        <div style="margin-left: 2em; text-align: center;">
            <h3 style="color:#004aad;">Video Chat</h3>
            <p>
                Engage in a more human connection through video-based interaction. Observe expressions, feel empathy, and build a deeper
                connection with your AI wellness companion. Perfect for those who value visual cues and a face-to-face experience
                in their emotional wellness journey. See, speak, and heal—one mindful moment at a time.
            </p>
        </div>
    </div>            
""",
    unsafe_allow_html=True,
)