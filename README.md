Customer Churn Prediction using Transformers

A simple end-to-end Machine Learning web application that predicts whether a telecom customer is likely to churn using a Transformer-based NLP model.

The system converts structured customer data into natural language text and uses a DistilBERT transformer model to classify churn risk.

This project demonstrates ML model inference, API deployment, and full-stack integration.

Live Demo

Once deployed, the application allows users to input customer details and instantly receive:

Churn Prediction (Yes / No)

Probability Score

Risk Level (Low / Medium / High)

System Architecture

User Input → Flask Web App → Transformer Model → Prediction → Web Interface

Workflow:

User enters customer information through the web interface.

The backend converts the data into descriptive text.

The transformer model processes the text.

The model predicts churn probability.

Results are returned to the UI.

Features

• Web interface for customer data input
• Transformer-based classification model
• Churn probability estimation
• Risk categorization (Low / Medium / High)
• Flask backend API
• Deployable on cloud platforms (Railway / Render / etc.)

Tech Stack

Backend
Python
Flask

Machine Learning
HuggingFace Transformers
DistilBERT

Deployment
Gunicorn
Railway / Render

Frontend
HTML
CSS

Project Structure
customer-churn-transformer
│
├── main.py                # Flask application
├── requirements.txt       # Project dependencies
├── templates
│   └── index.html         # Web interface
├── README.md
How It Works

The system converts structured data into text format:

Example input:

Gender: Male
Contract: Month-to-month
Internet: Fiber optic
Monthly charges: 75

Converted to text:

Customer is male.  
Contract type is month-to-month.  
Internet service is fiber optic.  
Monthly charges are 75 dollars.

This text is then classified using a DistilBERT transformer model.

Installation

Clone the repository

git clone https://github.com/yourusername/customer-churn-transformer.git
cd customer-churn-transformer

Install dependencies

pip install -r requirements.txt

Run the application

python main.py

Open in browser

http://localhost:8000
Deployment

The project is designed to be easily deployed using:

• Railway
• Render
• Docker
• Cloud VM

Production server uses:

gunicorn main:app
Example Prediction Output

Churn: No
Probability: 4.14%
Risk Level: Low

Future Improvements

• Train a custom churn prediction model
• Use structured feature embeddings instead of text conversion
• Add model explainability (SHAP / LIME)
• Store predictions in a database
• Build REST API endpoints

Author

Surendar G

MSc Artificial Intelligence

GitHub
https://github.com/surendarg2003

LinkedIn
(Add your LinkedIn profile link)

License

This project is open-source and available under the MIT License.
