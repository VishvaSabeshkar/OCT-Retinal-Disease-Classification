import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
from recommendation import cnv,dme,drusen,normal
import os

# Page configuration
st.set_page_config(
    page_title="OCT Retinal Analysis Platform",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Super Light, Airy, Cool Design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Super light airy background with hint of blue */
    .stApp {
        background: linear-gradient(135deg, #F0F9FF 0%, #FFFFFF 50%, #F0FDFA 100%);
    }
    
    /* Very light sidebar - airy blue */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E0F2FE 0%, #F0F9FF 100%);
        box-shadow: 2px 0 15px rgba(14, 165, 233, 0.08);
    }
    
    [data-testid="stSidebar"] h1 {
        color: #0EA5E9;
        font-weight: 600;
        font-size: 1.5rem;
        text-align: center;
        padding: 0;
        margin: 0;
    }
    
    [data-testid="stSidebar"] h2 {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] p {
        color: #64748B !important;
        line-height: 1.4 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: #0EA5E9 !important;
        font-weight: 500;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stSelectbox {
        margin-top: 0.8rem;
    }
    
    /* Light, bright headers */
    h1 {
        color: #0EA5E9;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #06B6D4;
        font-weight: 700;
        font-size: 2rem;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    h4 {
        color: #64748B;
        font-size: 1.4rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    h5 {
        color: #0EA5E9;
        font-size: 1.2rem;
        margin-top: 1.5rem;
        font-weight: 600;
    }
    
    /* Light gray text - not dark at all */
    p, li {
        color: #64748B;
        line-height: 1.8;
        font-size: 1.05rem;
        font-weight: 400;
        transition: all 0.3s ease;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    p:hover {
        background: rgba(224, 242, 254, 0.3);
        transform: translateX(5px);
    }
    
    /* Bright links */
    a {
        color: #0EA5E9;
        text-decoration: none;
        font-weight: 500;
    }
    
    a:hover {
        color: #0284C7;
        text-decoration: underline;
    }
    
    /* Super light file uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 20px;
        padding: 2rem;
        border: 2px dashed #7DD3FC;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #38BDF8;
        background: rgba(240, 249, 255, 0.9);
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.15);
    }
    
    [data-testid="stFileUploader"] label {
        color: #0EA5E9 !important;
        font-weight: 600;
        font-size: 1.15rem;
    }
    
    /* Bright, airy button */
    .stButton > button {
        background: linear-gradient(135deg, #38BDF8 0%, #0EA5E9 100%);
        color: white !important;
        border: none;
        border-radius: 14px;
        padding: 1rem 3.5rem;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.25);
        transition: all 0.3s ease;
        margin-top: 1.5rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(14, 165, 233, 0.35);
    }
    
    /* Light, fresh success message */
    .stSuccess {
        background: linear-gradient(135deg, #D1FAE5 0%, #ECFDF5 100%);
        color: #059669;
        padding: 1.3rem 2rem;
        border-radius: 14px;
        border-left: 4px solid #10B981;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.15);
    }
    
    /* Super light expander */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        border: 1px solid #E0F2FE;
        margin-top: 2rem;
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.08);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stExpander"] summary {
        color: #0EA5E9;
        font-weight: 600;
        font-size: 1.15rem;
        padding: 1.3rem 1.8rem;
        background: linear-gradient(135deg, #F0F9FF 0%, #FFFFFF 100%);
        transition: all 0.3s ease;
    }
    
    [data-testid="stExpander"] summary:hover {
        background: linear-gradient(135deg, #E0F2FE 0%, #F0F9FF 100%);
    }
    
    [data-testid="stExpander"] > div > div {
        padding: 2rem;
    }
    
    /* Very subtle horizontal rule with more spacing */
    hr {
        border: none;
        border-top: 1px solid #E0F2FE;
        margin: 4rem 0;
    }
    
    /* Image - light shadow */
    img {
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.12);
        margin: 2.5rem auto;
        display: block;
        max-width: 100%;
    }
    
    /* Container - centered with better spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 980px;
    }
    
    /* Center align main content for web app feel */
    .main .block-container {
        text-align: left;
    }
    
    /* Center headers for web app look */
    h1, h2 {
        text-align: center;
    }
            
    
    h4 {
        text-align: center;
        margin-top: 3rem;
    }

    .stMarkdown p,
    .stMarkdown li {
        text-align: left;
    }

    /* Lists */
    ul, ol {
        padding-left: 1.5rem;
    }
    
    li {
        margin-bottom: 0.8rem;
    }
    
    /* Light bold text */
    strong {
        color: #64748B;
        font-weight: 700;
    }
    
    /* Bright italic */
    em {
        color: #0EA5E9;
        font-style: italic;
    }
    
    code {
        background: rgba(240, 249, 255, 0.8);
        color: #0EA5E9;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        border: 1px solid #E0F2FE;
    }
    
    .stMarkdown {
        color: #64748B;
    }
            
    .hero {
        text-align: center;
        padding: 3rem 2rem 2.5rem 2rem;
        margin-bottom: 2.5rem;
    }

    .hero h1 {
        font-size: 2.6rem;
        margin-bottom: 0.6rem;
    }

    .hero p {
        font-size: 1.1rem;
        color: #64748B;
        max-width: 720px;
        margin: 0.5rem auto 0;
        line-height: 1.6;
    }

    .app-card {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        border: 1px solid #E0F2FE;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.06);
    }


    /* Web app style sections */
    .element-container {
        margin-bottom: 2rem;
    }
    
    /* Add breathing room around content */
    p {
        margin-bottom: 1.5rem;
    }
    
    /* Section spacing */
    .stMarkdown > div {
        margin-bottom: 2rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #38BDF8 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("Trained_Model.h5")
    return model

def model_prediction(test_image_path):
    model = load_model()
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x)
    return np.argmax(prediction)

# Sidebar
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">👁️</div>
        <h2 style="color: #0EA5E9; font-size: 1.4rem; font-weight: 700; margin: 0;">OCT Analysis</h2>
        <h2 style="color: #64748B; font-size: 0.95rem; margin-top: 0.5rem; margin-bottom: 0.3rem; font-weight: 500; text-align: center;"> AI-Powered Retinal Diagnosis </h2>
        <br>        
        <h4 style="color: #94A3B8; font-size: 0.8rem; margin: 0; padding: 0 1rem; line-height: 1.4; font-weight: 300;">Automated OCT analysis to support faster retinal diagnosis.</h4>
    </div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])

if(app_mode=="Home"):

    st.markdown("""
        <div class="hero">

        <h1>OCT Retinal Analysis Platform</h1>

        <p style="
            font-size: 1.1rem;
            color: #64748B;
            max-width: 720px;
            margin: 0.6rem auto 0;
            line-height: 1.6;
            text-align: center;">
                AI-assisted analysis of retinal OCT scans<br>
                to support fast, consistent, and clinically meaningful disease identification.
        </p>

        </div>
        """, unsafe_allow_html=True)




    st.markdown("""
        <div class="app-card">

        <h3 style="color:#0EA5E9; margin-bottom:0.8rem; text-align:center;"> What this platform does </h3>

        <p>
            This application uses a deep learning model to analyze retinal OCT scans and
            assist in identifying common retinal conditions.
        </p>

        <ul>
            <li>Normal retina</li>
            <li>Choroidal Neovascularization (CNV)</li>
            <li>Diabetic Macular Edema (DME)</li>
            <li>Drusen (Early AMD)</li>
        </ul>

        </div>

        <div class="app-card">

        <h3 style="color:#0EA5E9; margin-bottom:0.8rem; text-align:center;">How it works</h3>

        <ol>
            <li>Upload an OCT retinal image</li>
            <li>The model preprocesses and analyzes the scan</li>
            <li>The predicted condition is displayed instantly</li>
        </ol>

        </div>

        <div class="app-card">

        <h3 style="color:#0EA5E9; margin-bottom:0.8rem; text-align:center;">Who this is for</h3>

        <p>
            Designed to support clinicians, researchers, and students by reducing
            manual OCT interpretation time while maintaining consistency.
        </p>

        </div>
        """, unsafe_allow_html=True)



elif(app_mode=="About"):
    st.markdown("""
        <div class="hero">
            <h1>About the OCT Retinal Analysis Platform</h1>
            <p style="
                font-size: 1.1rem;
                color: #64748B;
                max-width: 720px;
                margin: 0.6rem auto 0;
                line-height: 1.6;
                text-align: center;">
                This platform demonstrates how deep learning can support retinal OCT interpretation
                for clinicians, researchers, and students.
            </p>
        </div>

        <div class="app-card">
            <h3 style="color:#0EA5E9; margin-bottom:0.8rem; text-align:center;">About the Dataset</h3>
            <p>
                This project is based on a large-scale retinal OCT dataset widely used in academic research.  
                OCT images capture microscopic retinal structures and are critical for diagnosing retinal diseases.
            </p>
            <ul>
                <li><strong>CNV</strong> – Subretinal neovascular membranes</li>
                <li><strong>DME</strong> – Retinal thickening with intraretinal fluid</li>
                <li><strong>Drusen</strong> – Deposits associated with early AMD</li>
                <li><strong>Normal</strong> – Healthy retinal anatomy</li>
            </ul>
        </div>

        <div class="app-card">
            <h3 style="color:#0EA5E9; margin-bottom:0.8rem; text-align:center;">Dataset Overview</h3>
            <ul>
                <li><strong>Total images:</strong> 84,495 OCT scans</li>
                <li><strong>Format:</strong> JPEG</li>
                <li><strong>Data split:</strong> Training, Validation, Testing</li>
                <li><strong>Classes:</strong> CNV, DME, DRUSEN, NORMAL</li>
            </ul>
        </div>

        <div class="app-card">
            <h3 style="color:#0EA5E9; margin-bottom:0.8rem; text-align:center;">Data Quality & Verification</h3>
            <p>All scans underwent a <strong>multi-level expert grading process</strong>:</p>
            <ol>
                <li>Initial quality control by trained graders</li>
                <li>Independent review by ophthalmologists</li>
                <li>Final verification by senior retinal specialists</li>
            </ol>
            <p>This tiered validation ensures <strong>high label accuracy</strong> and reduces grading bias.</p>
        </div>

        <div class="app-card">
            <h3 style="color:#0EA5E9; margin-bottom:0.8rem; text-align:center;">Purpose of This Platform</h3>
            <p>
                This application demonstrates how deep learning can be applied to:
            </p>
            <ul>
                <li>Assist OCT interpretation</li>
                <li>Support clinical workflows</li>
                <li>Serve as an educational and research tool</li>
            </ul>
            <p><em>This platform is intended for decision support and academic demonstration purposes.</em></p>
        </div>
    """, unsafe_allow_html=True)

    


elif(app_mode=="Disease Identification"):
    st.header("Welcome to the Retinal OCT Analysis Platform")
    test_image = st.file_uploader("Upload your OCT retinal image")
    if test_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
            tmp_file.write(test_image.read())
            temp_file_path = tmp_file.name
    
    
    if(st.button("Predict")) and test_image is not None:
        with st.spinner("Please Wait.."):
            result_index = model_prediction(temp_file_path)
            class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        st.success(f"Model is Predicting it's a {class_name[result_index]}.")

        with st.expander("Learn More"):
            if(result_index==0):
                st.write('''
                    OCT scan showing *CNV with subretinal fluid.*
                ''')
                st.image(test_image)
                st.markdown(cnv)
        
            elif(result_index==1):
                st.write('''
                    OCT scan showing *DME with retinal thickening and intraretinal fluid.*
                ''')
                st.image(test_image)
                st.markdown(dme)

            elif(result_index==2):
                st.write('''
                    OCT scan showing *drusen deposits in early AMD.*
                ''')
                st.image(test_image)
                st.markdown(drusen)
                
            elif(result_index==3):
                st.write('''
                    OCT scan showing a *normal retina with preserved foveal contour.*
                ''')
                st.image(test_image)
                st.markdown(normal)

        os.remove(temp_file_path)