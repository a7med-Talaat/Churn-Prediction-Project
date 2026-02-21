# E-Commerce Customer Churn Prediction

## What Does This Project Do?

When customers stop buying from an online store, that's called **churn**. This project is an enterprise-grade supervised learning solution that takes raw customer purchase data and **predicts exactly which high-value customers are in danger of leaving**.

More than just a prediction engine, this dashboard provides an actionable end-to-end retention workflow:
1. **Interactive Tracking:** Calculate real-time Churn Risk and Customer Lifetime Value (CLV) based on dynamic user behavior.
2. **Targeted Interventions:** Automatically draft highly-personalized emails tailored to the exact risk factors driving a customer away.
3. **Data Actionability:** Sync directly with major CRMs (HubSpot/Salesforce) or SendGrid directly from the UI, with every single action securely logged in a permanent SQLite Outreach History database.
4. **Automated Personas:** Process thousands of users into dynamic K-Means Machine Learning clusters to isolate "VIP Whales" from "Sleeping Giants."

## How Does It Work?

### Before start :

make sure to download the dataset from this link ( https://archive.ics.uci.edu/dataset/502/online+retail+ii ) and place the downloaded file after extracting it in the data folder 

### Step 1: Clean the Data

The raw data is a big list of online store transactions (who bought what, when, and for how much).
This step cleans up the messy data:

- Removes cancelled orders
- Removes returns (negative quantities)
- Drops rows where the customer ID is missing

Then it creates **3 simple numbers** for each customer:

| Feature       | What It Means                              |
|---------------|--------------------------------------------||
| **Recency**   | How many days since their last purchase     |
| **Frequency** | How many times they bought something        |
| **Monetary**  | How much money they spent in total          |

If a customer hasn't bought anything in **more than 90 days**, we label them as **churned**.

### Step 2: Train the Model 

The model (XGBoost) learns the patterns from the data — for example:
- Customers who haven't bought in a long time are more likely to leave
- Customers who spent very little may also leave

After training, the model is saved so we can use it later without retraining.

### Step 3: Use the Model

There are **two ways** to get predictions:

- **API** (`app/main.py`) — Send customer data and get back a churn prediction. Good for connecting to other apps.
- **Dashboard** (`app/dashboard.py`) — A simple web page where you can type in a customer's info or upload a file of many customers and get predictions.

## Project Structure

```
churn-project/
├── src/
│   ├── preprocessing.py   ← Cleans data and creates features
│   ├── train.py           ← Trains the model and saves it
│   └── __init__.py
├── app/
│   ├── main.py            ← FastAPI server (API)
│   └── dashboard.py       ← Streamlit web dashboard
├── data/
│   └── online_retail_II.xlsx  ← The raw transaction data
├── models/                ← Saved model files (created after training)
├── requirements.txt       ← Python packages needed
├── Dockerfile             ← For running in a container
└── README.md
```

## How to Run

### 1. Create a Virtual Environment and Install Packages

If you are on Windows, ensure you are using a standard 64-bit Python 3.11 or 3.12 installation (avoiding MSYS2/MinGW environments which can cause package build errors).

**Windows (PowerShell):**
```powershell
# Create the environment using the explicit Windows Python path
& "your python.exe location" -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install the packages
pip install --upgrade pip setuptools wheel
cd (folder location)
pip install -r requirements.txt
```
*(Note: If you get an "Execution Policy" error when activating, run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser` first, then try activating again).*

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python -m src.train
```

You should see output like this:

```
🔄 Loading Data...
🛠 Preparing Features...
🚀 Training Model...

📊 Model Performance:
              precision    recall  f1-score   support
           0       0.99      1.00      0.99       576
           1       1.00      0.98      0.99       287
    accuracy                           0.99       863

ROC-AUC Score: 1.0000
```

### 3. Start the API

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The API will be running at: **http://127.0.0.1:8000**

- Open **http://127.0.0.1:8000/docs** in your browser to see and test the API.

**Test it with a sample request:**

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"CustomerID\": 12345, \"Recency\": 30, \"Frequency\": 5, \"Monetary\": 500.0}"
```

Or in **PowerShell**:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{"CustomerID": 12345, "Recency": 30, "Frequency": 5, "Monetary": 500.0}'
```

Example response:

```json
{
  "CustomerID": 12345,
  "Churn": false,
  "Churn_Probability": 0.0009,
  "Risk_Level": "Low"
}
```

### 4. Start the Dashboard

Open a **new terminal** (keep the API running), access the project folder with ( cd 'folder location' ), activate the venv again, then:

```bash
python -m streamlit run app/dashboard.py --server.port 8501
```

Open **http://127.0.0.1:8501** in your browser to view the enhanced dashboard featuring interactive Plotly visualizations and AI retention recommendations.

### Dashboard Showcases

**Home Page / Project Overview**
![Home Page](./assets/Home%20page.png)

**Single Customer Analysis**
![Single Customer Analysis Page](./assets/Single%20customer%20analysis%20page.png)

**AI Retention & AI Email Drafter**
![AI Retention and Drafter](./assets/AI%20Retenation%20,%20AI%20Email%20Drafter.png)

**Batch Analytics & Persona Segmentation**
![Batch Analysis Page](./assets/Batch%20analysis%20page.png)

**Outreach History Database**
![Outreach History Database](./assets/Outreach%20history%20database.png)

From the dashboard, you can:

- **Single Customer Analysis:** Real-Time Dashboard calculating precise Churn Probability, Estimated Customer Lifetime Value (CLV), and SHAP Feature impact.
- **Automated Retention & Outreach:** Instantly generated, highly personalized email drafts tailored to the customer's specific risk factors with CRM integration buttons.
- **Batch Processing & Personas:** Process thousands of rows simultaneously to classify customers into 4 K-Means Machine Learning Personas.
- **Priority Threat Matrix:** Visualize the entire customer base in a responsive 4-quadrant bubble map to instantly identify high-value users at high risk of churn.
- **Audit & Compliance:** Monitor all executed actions (Sent Emails, CRM pushes) through a permanent SQLite log on the Outreach History tab.

### 5. Run with Docker (Optional)

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

This starts only the API inside a container.

## Tech Used

| Tool         | What It Does                          |
|--------------|---------------------------------------|
| Python       | The programming language              |
| Pandas       | Works with data (tables, cleaning)    |
| XGBoost      | The machine learning model            |
| Scikit-Learn | Splits data, measures model accuracy  |
| FastAPI      | Creates the API                       |
| Streamlit    | Creates the web dashboard             |
| Docker       | Packages everything into a container  |
