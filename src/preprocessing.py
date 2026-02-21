import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    """Load and perform basic cleaning on raw data."""
    df = pd.read_excel(path)
    
    # Remove cancellations (Invoice starts with 'C')
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    
    # Remove negative quantities (returns)
    df = df[df['Quantity'] > 0]
    
    # Calculate Total Amount
    df['TotalAmount'] = df['Quantity'] * df['Price']
    
    # Drop rows with missing Customer ID
    df = df.dropna(subset=['Customer ID'])
    
    # Convert Customer ID to int
    df['Customer ID'] = df['Customer ID'].astype(int)
    
    return df

def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Recency, Frequency, Monetary features."""
    # Reference date is the day after the last transaction in the dataset
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days, # Recency
        'Invoice': 'count',                                       # Frequency
        'TotalAmount': 'sum'                                      # Monetary
    }).reset_index()
    
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    
    # Define Churn: If Recency > 90 days, label as 1 (Churned), else 0
    # Note: In a real business case, this threshold is decided by stakeholders
    rfm['Churn'] = (rfm['Recency'] > 90).astype(int)
    
    return rfm

def assign_rfm_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a customer segment based on RFM quartile scores."""
    df = df.copy()

    # Score each feature from 1-4 using quartiles (4 = best)
    # For Recency, lower is better so we reverse the labels
    df['R_Score'] = pd.qcut(df['Recency'], q=4, labels=[4, 3, 2, 1]).astype(int)
    df['F_Score'] = pd.qcut(df['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
    df['M_Score'] = pd.qcut(df['Monetary'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)

    df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']

    def _label(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 4 and f >= 4:
            return 'Champions'
        if r >= 3 and f >= 3:
            return 'Loyal'
        if r >= 3 and f <= 2:
            return 'New Customers'
        if r == 2 and f >= 2:
            return 'At Risk'
        if r <= 1 and f >= 3:
            return 'Can\'t Lose Them'
        return 'Lost'

    df['Segment'] = df.apply(_label, axis=1)
    return df


def get_retention_tip(probability: float, segment: str = "") -> str:
    """Return a plain-English retention recommendation."""
    if probability < 0.3:
        return "Customer looks healthy. Keep up regular engagement."
    if probability < 0.6:
        if segment in ("At Risk", "Can't Lose Them"):
            return "Send a personalized win-back offer or loyalty reward soon."
        return "Consider a check-in email or small incentive to re-engage."
    if probability < 0.8:
        return "High risk — reach out with a strong discount or exclusive deal now."
    return "Very high risk — escalate to retention team for a personal outreach."


def prepare_model_data(df: pd.DataFrame):
    """Split features and target."""
    features = ['Recency', 'Frequency', 'Monetary']
    X = df[features]
    y = df['Churn']
    return X, y, features
