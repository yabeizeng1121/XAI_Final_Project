from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_sentiment(text):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model.cuda()
        outputs = model(**inputs)
        scores = outputs.logits.softmax(dim=1).cpu().numpy()
        sentiment = "Positive" if scores[0][1] > 0.5 else "Negative"
    return sentiment
