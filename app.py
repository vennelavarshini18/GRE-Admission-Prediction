import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.save")

st.set_page_config(page_title="Graduate Admission Predictor", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #6c757d;
    }
    .result-box {
        border-radius: 12px;
        padding: 20px;
        background: #e3f2fd;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        color: #0d47a1;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🎓 Graduate Admission Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict your chance of admission using machine learning</p>', unsafe_allow_html=True)
st.markdown("---")

with st.expander("📥 Enter Your Academic Profile", expanded=True):
    gre = st.number_input("📈 GRE Score", min_value=260, max_value=340, value=320)
    toefl = st.number_input("📝 TOEFL Score", min_value=0, max_value=120, value=110)
    univ = st.slider("🏫 University Rating", 1, 5, 3)
    sop = st.slider("🗣️ SOP Strength", 1.0, 5.0, 3.0, step=0.5)
    lor = st.slider("📃 LOR Strength", 1.0, 5.0, 3.0, step=0.5)
    cgpa = st.number_input("📚 CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.5, step=0.1)
    research = st.selectbox("🔬 Research Experience", ["No", "Yes"])
    research_val = 1 if research == "Yes" else 0

st.markdown("### 🔎 Click the button below to predict your result:")
if st.button("🚀 Predict Admission Chance", use_container_width=True):
    input_data = np.array([[gre, toefl, univ, sop, lor, cgpa, research_val]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0][0]
    percent = prediction * 100

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if percent >= 85:
        st.markdown(
            '<div class="result-box">🎉 <strong>Excellent Profile!</strong><br>'
            'Your predicted chance of admission is <strong>{:.2f}%</strong>.<br>'
            'You are a very competitive candidate for top universities! 🌟</div>'.format(percent),
            unsafe_allow_html=True
        )
        st.balloons()
    elif percent >= 70:
        st.markdown(
            '<div class="result-box">🟡 <strong>Good Profile</strong><br>'
            'Your predicted chance of admission is <strong>{:.2f}%</strong>.<br>'
            'You have a decent chance. Consider applying widely and improving your SOP/LOR. 🚀</div>'.format(percent),
            unsafe_allow_html=True
        )
    elif percent >= 50:
        st.markdown(
            '<div class="result-box">🟠 <strong>Average Profile</strong><br>'
            'Your predicted chance is <strong>{:.2f}%</strong>.<br>'
            'Improve specific areas like GRE, CGPA, or research to boost your chances. 💪</div>'.format(percent),
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box">🔴 <strong>Low Predicted Chance</strong><br>'
            'Your estimated admission probability is <strong>{:.2f}%</strong>.<br>'
            'Don’t lose hope — apply strategically, enhance your SOP, and consider safer universities. 📚</div>'.format(percent),
            unsafe_allow_html=True
        )

with st.sidebar:
    st.header("📘 Tips")
    st.markdown("""
    - Aim for GRE > 320 and TOEFL > 105  
    - Strong SOP and LOR can help borderline profiles  
    - Research experience adds a boost 🔬  
    - CGPA is a major factor in top universities  
    """)
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & TensorFlow")

st.markdown("""
    <hr style="border:0.5px solid #eee">
    <center><small>📊 Powered by Keras Neural Network | Dataset: GRE Admissions</small></center>
""", unsafe_allow_html=True)
