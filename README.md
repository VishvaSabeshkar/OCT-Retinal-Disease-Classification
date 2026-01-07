# OCT Retinal Disease Classification using Deep Learning

## Overview
This project implements an **AI-assisted retinal OCT analysis platform** that classifies Optical Coherence Tomography (OCT) images into four clinically relevant categories: **CNV, DME, DRUSEN, and NORMAL**.

A **MobileNetV3-based deep learning model**, trained on a large, expert-labeled OCT dataset, is integrated into a **Streamlit web application**. Users can upload an OCT scan and receive an instant prediction along with clinically interpretable recommendations.  
The system is designed as a **decision-support and academic demonstration tool**, not as a standalone diagnostic system.

---

## Dataset Overview
- **Dataset**: Large Dataset of Labeled Optical Coherence Tomography (OCT) Images  
- **Source**: Mendeley Data  
- **Citation**:  
  Kermany, D., Zhang, K., Goldbaum, M. (2018).  
  *Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images*  
  DOI: `10.17632/rscbjbr9sj.3`

- **Total Images Used**: 84,495 OCT scans  
- **Format**: JPEG  
- **Classes**: CNV, DME, DRUSEN, NORMAL  

All images were validated using a **multi-tier expert grading pipeline**, ensuring high label reliability and minimal annotation bias.

---

## Model Performance (Test Set)
- **Accuracy**: 96.9%  
- **Precision**: 0.97  
- **Recall**: 0.97  
- **Weighted F1-Score**: 0.97  

The model shows strong generalization across all four retinal disease classes.

---

## Tools & Technologies
**Machine Learning**
- Python, TensorFlow/Keras
- MobileNetV3 (ImageNet pretrained)
- NumPy, Scikit-learn

**Data & Visualization**
- Pandas, Matplotlib, Seaborn

**Web Application**
- Streamlit
- Custom CSS UI

**Model Persistence**
- HDF5 (.h5), Pickle

---

## Key Features
- Automated OCT image classification  
- Four-class retinal disease prediction  
- Real-time inference via web interface  
- Clinically interpretable recommendations  
- Confusion matrix and classification report  
- Lightweight CNN optimized for efficiency  

---

## Process Flow
1. **Data Preparation**  
   OCT images split into train, validation, and test sets and resized to 224×224.

2. **Model Training**  
   MobileNetV3 with a custom dense classification head trained using categorical cross-entropy.

3. **Evaluation**  
   Model evaluated on an unseen test set with detailed performance metrics.

4. **Deployment**  
   Trained model deployed via a Streamlit web application.

---

## How to Run
```bash
# Clone repository
git clone https://github.com/your-username/oct-retinal-analysis.git
cd oct-retinal-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```
## Summary
This project demonstrates how deep learning can support retinal OCT interpretation by improving consistency and efficiency in medical image analysis. By combining a high-quality dataset, a pretrained CNN architecture, and an interactive web interface, the platform serves as a strong example of applied AI in healthcare research and education.
