import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load trained model
tokenizer = DistilBertTokenizer.from_pretrained("model")
model = DistilBertForSequenceClassification.from_pretrained("model")

model.eval()

device = torch.device("cpu")
model.to(device)

text = """
The customer has month to month contract.
Internet service is fiber optic.
Monthly charges are 90 dollars.
Payment method is electronic check.
"""

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

prediction = torch.argmax(outputs.logits).item()

print("Churn Prediction:", "Yes" if prediction == 1 else "No")