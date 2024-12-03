# XAI Final Project

An interactive platform designed to introduce Explainable AI (XAI) concepts to beginners. This application demonstrates how XAI techniques enhance the transparency and interpretability of machine learning models, using sentiment analysis as a practical example.

## Application Link
Access the app here: [XAI Fundamentals](https://xai-fundamentals.streamlit.app/)

---

## Features

### **Learning Modules**
- Input any text into the **sentiment analysis** tool to classify it as **Positive**, **Neutral**, or **Negative**.
- Learn how XAI techniques, such as **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-Agnostic Explanations), explain the model's predictions by identifying the most influential words or features.

### **Case Studies**
- Explore real-world applications of XAI in domains like healthcare, finance, and customer feedback.
- Understand how interpretability methods uncover biases, enhance trust, and ensure fairness in AI systems.

---

## How to Use

### **1. Learning Modules**
1. Enter text in the input box (e.g., "This product is amazing!").
2. Click **Analyze** to get the sentiment prediction.
3. Choose an XAI method (SHAP or LIME) and click **Explain** to visualize how the prediction was made.

### **2. Case Studies**
- Read examples of XAI applied to real-world datasets.
- Explore use cases in customer feedback analysis, healthcare diagnostics, financial risk assessment, and more.

---

## Technologies Used
- **Streamlit**: For building the web application.
- **Transformers**: To leverage pre-trained models for sentiment analysis.
- **SHAP**: For providing interpretability via feature attribution.
- **LIME**: For local interpretability of individual predictions.
- **PyTorch**: For model implementation and inference.
- **Matplotlib**: For visualizations of explanations.

---

## Deployment
This app is hosted on [Streamlit Cloud](https://streamlit.io/). The platform reads the `requirements.txt` file to install the necessary dependencies and runs the application seamlessly.

---

## Local Setup
To run the app locally, follow these steps:
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Access the app in your browser at `http://localhost:8501`.

---
## Contributions
Contributions to improve or expand this project are welcome! Feel free to submit issues or pull requests to enhance the app.
