import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp"

from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load transformer pipeline (CPU mode)
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased",
    device=-1
)

def predict_churn(text):
    result = classifier(text)[0]

    churn_prob = result["score"]
    prediction = "Yes" if result["label"] == "LABEL_1" else "No"

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
