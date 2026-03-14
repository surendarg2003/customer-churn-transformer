from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# load model directly from HuggingFace
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

model.eval()

device = torch.device("cpu")
model.to(device)


def predict_churn(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    churn_prob = probs[0][1].item()

    prediction = "Yes" if churn_prob > 0.5 else "No"

    if churn_prob < 0.30:
        risk = "Low"
    elif churn_prob < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    return prediction, round(churn_prob * 100, 2), risk


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    risk = None

    if request.method == "POST":

        gender = request.form.get("gender")
        senior = request.form.get("senior")
        partner = request.form.get("partner")
        dependents = request.form.get("dependents")
        tenure = request.form.get("tenure")
        contract = request.form.get("contract")
        internet = request.form.get("internet")
        charges = request.form.get("charges")
        payment = request.form.get("payment")

        text = f"""
        Customer is {gender}.
        Senior citizen status is {senior}.
        Partner status is {partner}.
        Dependents status is {dependents}.
        Customer tenure is {tenure} months.
        Contract type is {contract}.
        Internet service is {internet}.
        Monthly charges are {charges} dollars.
        Payment method is {payment}.
        """

        prediction, probability, risk = predict_churn(text)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        risk=risk
    )


if __name__ == "__main__":
    app.run(debug=True)
