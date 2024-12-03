from scipy import special
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import shap
from lime.lime_text import LimeTextExplainer
import numpy as np
import matplotlib.pyplot as plt

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
model.eval()
if torch.cuda.is_available():
    model.cuda()


def f(x):
    # Encode the input text and process it directly to generate logit values
    encoded_inputs = [
        tokenizer.encode(v, padding="max_length", max_length=500, truncation=True)
        for v in x
    ]
    tv = torch.tensor(encoded_inputs)
    if torch.cuda.is_available():
        tv = tv.cuda()  # Only move to CUDA if available
    outputs = (
        model(tv)[0].detach().cpu().numpy()
    )  # Always detach and move to cpu for numpy conversion
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = special.logit(scores[:, 1])  # Use one vs rest logit units
    return val


def explain_with_shap(text):
    # Create an explainer using the function f
    explainer = shap.Explainer(f, tokenizer)

    # SHAP values for the provided input
    shap_values = explainer([text], fixed_context=1)  # Ensure text is passed as a list

    # Plot the SHAP values using text plot
    shap_fig, shap_ax = plt.subplots()
    shap.plots.waterfall(shap_values[0])
    plt.close(shap_fig)  # Close the figure to prevent it from displaying immediately
    return shap_fig


def predict_proba(texts):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )
    if torch.cuda.is_available():
        encoded = {k: v.cuda() for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    probabilities = softmax(outputs.logits.detach().cpu().numpy(), axis=1)
    return probabilities


def explain_with_lime(text):
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    exp = explainer.explain_instance(
        text, predict_proba, num_features=len(text.split()), labels=(1,)
    )
    return exp


def get_lime_explanation_data(exp):
    # Extract feature weights for the positive class
    weights = exp.as_list(label=1)
    features, values = zip(*weights) if weights else ([], [])
    return features, values


def plot_lime_explanation(features, values):
    indices = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(10, 5))  # Create a figure and axes object
    ax.bar(indices, values, align="center", alpha=0.7)
    ax.set_xticks(indices)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Contribution to Positive Class")
    ax.set_title("Word Contributions to Sentiment")
    plt.tight_layout()
    return fig  # Return the figure object for use in Streamlit
