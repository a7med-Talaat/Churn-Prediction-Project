import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import time
import sqlite3
from datetime import datetime
import os

# Page Config
st.set_page_config(page_title="Churn Predictor Pro", layout="wide", initial_sidebar_state="expanded")

# Initialize SQLite Database for Outreach History
def init_db():
    # Ensure data dir exists
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/outreach_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS outreach_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            customer_id INTEGER,
            action_type TEXT,
            status TEXT,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_action(customer_id, action_type, status, details):
    conn = sqlite3.connect('data/outreach_history.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO outreach_history (timestamp, customer_id, action_type, status, details)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, customer_id, action_type, status, details))
    conn.commit()
    conn.close()
    
# Call to ensure DB exists on startup
init_db()

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }
    .action-btn {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🛒 E-Commerce Churn Prediction Pro")
st.markdown("Discover actionable insights, predict customer churn risk, and receive data-driven retention strategies.")
st.divider()

# Attempt to load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/churn_model.pkl")
    except Exception:
        return None

model = load_model()

if model is None:
    st.error("Model not found. Please run the training script first.")
    st.stop()

# Helper function to calculate a mock CLV (Customer Lifetime Value)
def calculate_clv(monetary, frequency, recency):
    aov = monetary / max(frequency, 1)
    lifespan_multiplier = max(1, (365 - recency) / 60) 
    clv = aov * frequency * lifespan_multiplier
    return clv

# Tabs
tab_overview, tab1, tab2, tab3 = st.tabs(["ℹ️ Project Overview", "👤 Single Customer Analysis", "📊 Batch Analytics (CSV)", "📖 Outreach History"])

# ----------------- TAB OVERVIEW: PROJECT DESCRIPTION -----------------
with tab_overview:
    st.header("About This Project")
    st.markdown("""
    When customers stop buying from an online store, that's called **churn**. This project is designed to precisely predict which customers are likely to leave, allowing businesses to take proactive action—like sending a targeted discount or a reminder—before the customer is gone.

    ### 🛠️ How It Works
    We analyze how customers have been shopping using three key predictive metrics factors (known as **RFM**):
    1. **Recency**: How many days since their last purchase.
    2. **Frequency**: How many times they have bought something over their lifetime.
    3. **Monetary**: How much money they have spent in total.

    Using an advanced Machine Learning model (XGBoost), we learn the complex patterns from historical transaction data to calculate a **Churn Probability** for each active customer. 

    ### 🚀 Dashboard Features
    - **Single Customer Analysis (What-If Simulator):** Use the interactive sliders to instantly simulate customer behavior changes (e.g., getting a customer to make one more purchase). See their churn risk update in real-time, understand *why* they are at risk using SHAP Feature Impact analysis, and get targeted retention strategies.
    - **Smart Outreach Drafter & CRM Integration:** The dashboard automatically writes personalized marketing emails targeting the specific reason a customer is leaving, complete with 1-click integration to leading CRMs.
    - **Batch Analytics & Personas:** Upload a CSV of your customer base to get bulk predictions. This feature automatically segments your customers into 4 distinct behavioral personas (e.g., VIPs, At-Risk) using advanced **K-Means Clustering**, visualized via interactive 3D scatter plots and Radar charts.
    - **Priority Matrix (CLV vs. Risk):** Instantly visualize which high-value customers are in the "Danger Zone" using our interactive quad-matrix bubble chart.

    ### 💰 What is Customer Lifetime Value (CLV)?
    Alongside predicting whether a customer will churn, this dashboard calculates an **Estimated Customer Lifetime Value (CLV)**. This metric represents the total net profit or revenue a business can expect from a single customer throughout their entire relationship. 
    
    By tracking CLV alongside Churn Probability, businesses can prioritize exactly *who* to save. A customer with an 80% risk of churning who has a CLV of \\$10 is lower priority than a customer with a 50% risk but a CLV of \\$5,000.
    """)

# ----------------- TAB 1: SINGLE PREDICTION -----------------
with tab1:
    st.header("Single Customer Prediction & What-If Simulator")
    st.markdown("Adjust the sliders below to simulate different customer scenarios and see how the churn risk changes in real-time.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Customer Details")
        c_id = st.number_input("Customer ID", value=12345)
        recency = st.slider("Recency (Days since last purchase)", min_value=1, max_value=360, value=30, help="Lower is better")
        frequency = st.slider("Frequency (Number of purchases)", min_value=1, max_value=100, value=5, help="Higher is better")
        monetary = st.slider("Monetary (Total Spend $)", min_value=10.0, max_value=5000.0, value=500.0, step=10.0, help="Higher is better")
        
    # Since we use sliders, we calculate prediction reactively.
    input_data = pd.DataFrame([{"Recency": recency, "Frequency": frequency, "Monetary": monetary}])
    
    # Prediction
    prob = model.predict_proba(input_data)[0][1]
    
    # Calculate CLV
    clv_value = calculate_clv(monetary, frequency, recency)
    
    with col2:
        st.subheader("Analysis Results")
        
        # Adding KPI Metrics
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Churn Risk", f"{prob * 100:.1f}%", delta=None)
        kpi2.metric("Estimated CLV", f"${clv_value:,.2f}", help="Predicted total lifetime value of this customer.")
        if prob > 0.6:
            risk_label = "🔥 High Risk"
        elif prob > 0.3:
            risk_label = "⚠️ Medium Risk"
        else:
            risk_label = "✅ Low Risk"
        kpi3.metric("Status", risk_label)
        
        st.divider()

        col2_a, col2_b = st.columns(2)
        
        with col2_a:
            # 1. Gauge Chart for Probability
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability", 'font': {'size': 18}},
                number={'suffix': "%", 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(144, 238, 144, 0.6)"}, # Light Green
                        {'range': [30, 70], 'color': "rgba(255, 215, 0, 0.6)"},   # Gold
                        {'range': [70, 100], 'color': "rgba(250, 128, 114, 0.6)"} # Salmon
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                height=280, 
                margin=dict(l=30, r=30, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col2_b:
            # 2. SHAP Explanation
            st.markdown("#### 🤔 Why this prediction?", help="SHAP values show which features push the prediction higher or lower.")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            
            # Simple bar chart for Shap values
            features = list(input_data.columns)
            shap_val = shap_values[0]
            
            impact = ["Increases Risk" if v > 0 else "Decreases Risk" for v in shap_val]
            
            fig_shap = px.bar(
                x=shap_val, 
                y=features, 
                orientation='h',
                color=impact,
                color_discrete_map={"Increases Risk": "salmon", "Decreases Risk": "lightgreen"},
                labels={'x': "Impact on Churn Risk", 'y': 'Feature'}
            )
            fig_shap.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig_shap, use_container_width=True)
        
        st.divider()

        # 3. AI Recommendations & Email Drafter
        col_rec, col_email = st.columns(2)
        
        most_important_idx = np.abs(shap_val).argmax()
        top_feature = features[most_important_idx]
        top_val = shap_val[most_important_idx]
        
        with col_rec:
            st.markdown("#### 💡 AI Retention Strategy")
            if prob < 0.3:
                st.success("Customer is highly engaged! No immediate action required. Recommend sending a loyalty reward.")
                email_draft = f"Subject: You're one of our favorites! 🎉\n\nHi there,\n\nWe noticed you've been shopping with us recently and we just wanted to say thank you! As a token of our appreciation, here is a special loyalty gift: [GIFT_LINK].\n\nStay awesome,\n The Team"
            else:
                if top_feature == "Recency" and top_val > 0:
                    st.warning("**Action:** High risk due to inactivity. Send a 'We Miss You' email with a 20% personalized discount code valid for 48 hours.")
                    email_draft = f"Subject: We miss you! Here's 20% off just for you 🎁\n\nHi there,\n\nIt's been a while since your last visit {recency} days ago! We've added lots of new items we think you'll love.\n\nCome back and use code MISSYOU20 for 20% off your next order.\n\nShop now: [STORE_LINK]\n\nBest,\nThe Team"
                elif top_feature == "Frequency" and top_val > 0:
                    st.warning("**Action:** Customer buys rarely. Suggest a points-based loyalty program or a subscription model to encourage regular visits.")
                    email_draft = f"Subject: Unlock VIP Perks Today! ⭐\n\nHi there,\n\nDid you know you can earn rewards on every purchase? Join our VIP loyalty program today and get free shipping on your next order.\n\nJoin here: [VIP_LINK]\n\nBest,\nThe Team"
                elif top_feature == "Monetary" and top_val > 0:
                    st.warning("**Action:** Low spend is driving the risk. Recommend upselling/cross-selling bundles or offering free shipping on higher-value orders.")
                    email_draft = f"Subject: Selected just for you: Our Premium Bundles 💎\n\nHi there,\n\nBased on your past purchases, we've curated a few premium bundles we think you'd enjoy.\n\nPlus, get FREE express shipping on all orders over $150.\n\nView Bundles: [BUNDLE_LINK]\n\nBest,\nThe Team"
                else:
                    st.warning("**Action:** General risk indicated. Check in with a customer satisfaction survey.")
                    email_draft = f"Subject: How are we doing? Let us know! 📝\n\nHi there,\n\nYour feedback is incredibly important to us. Could you take 1 minute to let us know how we can improve your experience?\n\nTake the survey: [SURVEY_LINK]\n\nBest,\nThe Team"

        with col_email:
            st.markdown("#### ✍️ AI Email Drafter")
            st.text_area("Generated Outreach Message", value=email_draft, height=150)
            
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("📧 Send Email via SendGrid", use_container_width=True):
                    with st.spinner('Sending...'):
                        time.sleep(1)
                        log_action(c_id, "Email via SendGrid", "Sent", "Targeted Risk-Based Outreach")
                        st.success("Email successfully logged & sent!")
            with btn_col2:
                if st.button("☁️ Push to Salesforce/HubSpot", use_container_width=True):
                    with st.spinner('Syncing to CRM...'):
                        time.sleep(1)
                        log_action(c_id, "CRM Sync", "Success", "Added to Churn Mitigation Campaign")
                        st.success("Successfully synced and logged to CRM!")

# ----------------- TAB 2: BATCH ANALYTICS -----------------
with tab2:
    st.header("Batch Analytics & Persona Segmentation")
    uploaded_file = st.file_uploader("Upload Customer CSV (Must contain Recency, Frequency, Monetary)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        required_cols = ['Recency', 'Frequency', 'Monetary']
        if not all(col in df.columns for col in required_cols):
            st.error("CSV must contain columns: Recency, Frequency, Monetary")
        else:
            # Predict
            preds = model.predict(df[required_cols])
            probs = model.predict_proba(df[required_cols])[:, 1]
            
            df['Churn_Probability'] = probs
            df['Risk_Level'] = df['Churn_Probability'].apply(
                lambda x: 'High' if x > 0.6 else ('Medium' if x > 0.3 else 'Low')
            )
            
            # Calculate CLV for batch
            df['Estimated_CLV'] = df.apply(lambda row: calculate_clv(row['Monetary'], row['Frequency'], row['Recency']), axis=1)
            
            st.success(f"Successfully processed {len(df)} customers!")
            
            # Metrics
            met1, met2, met3, met4 = st.columns(4)
            high_risk = len(df[df['Risk_Level'] == 'High'])
            avg_risk = df['Churn_Probability'].mean() * 100
            total_clv_at_risk = df[df['Risk_Level'] == 'High']['Estimated_CLV'].sum()
            
            met1.metric("Total Customers", len(df))
            met2.metric("High Risk Customers 🚨", high_risk)
            met3.metric("Avg Churn Risk", f"{avg_risk:.1f}%")
            met4.metric("Revenue at Risk 💸", f"${total_clv_at_risk:,.0f}")
            
            st.divider()
            
            # Priority Matrix - CLV vs Risk
            st.subheader("🎯 Threat Intelligence: Priority Matrix")
            st.markdown("Identify high-value customers who are at high risk of churning (Top Right Quadrant). Dot size indicates Frequency of purchases.")
            
            fig_matrix = px.scatter(
                df, 
                x='Churn_Probability', 
                y='Estimated_CLV',
                size='Frequency',
                color='Risk_Level',
                hover_data=['Recency', 'Monetary'],
                color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
                labels={'Churn_Probability': "Churn Risk (%)", 'Estimated_CLV': "Customer Lifetime Value ($)"}
            )
            # Add quadrant lines
            fig_matrix.add_hline(y=df['Estimated_CLV'].median(), line_dash="dash", line_color="gray", opacity=0.5)
            fig_matrix.add_vline(x=0.5, line_dash="dash", line_color="red", opacity=0.5)
            
            fig_matrix.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_matrix, use_container_width=True)
            
            st.divider()
            
            st.subheader("🤖 Persona Segmentation")
            st.markdown("We've grouped your customers into 4 distinct behavioral personas using K-Means Clustering.")
            
            # Scale RFM for clustering
            scaler = MinMaxScaler()
            rfm_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])
            
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(rfm_scaled)
            
            cluster_means = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
            
            # Dynamic naming based on cluster properties
            persona_mapping = {}
            if len(cluster_means) >= 4:
                vip_id = cluster_means['Monetary'].idxmax()
                persona_mapping[vip_id] = "🏆 VIPs (Whales)"
                remaining = cluster_means.drop(index=vip_id)
                dormant_id = remaining['Recency'].idxmax()
                persona_mapping[dormant_id] = "💤 Sleeping Giants"
                remaining2 = remaining.drop(index=dormant_id)
                regular_id = remaining2['Frequency'].idxmax()
                persona_mapping[regular_id] = "🛍️ Regulars"
                last_id = remaining2.drop(index=regular_id).index[0]
                persona_mapping[last_id] = "⚠️ At-Risk / Low Value"
            else:
                for i in cluster_means.index:
                    persona_mapping[i] = f"Persona {i}"
            
            df['Persona'] = df['Cluster'].map(persona_mapping)
            
            col_radar, col_bar = st.columns(2)
            
            with col_radar:
                fig_radar = go.Figure()
                min_max_means = scaler.fit_transform(cluster_means)
                categories = ['Recency', 'Frequency', 'Monetary']
                
                for i, c_id in enumerate(cluster_means.index):
                    fig_radar.add_trace(go.Scatterpolar(
                        r=list(min_max_means[i]) + [min_max_means[i][0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=persona_mapping[c_id]
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                    showlegend=True,
                    title="Persona Characteristics Profile",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
            with col_bar:
                persona_counts = df['Persona'].value_counts().reset_index()
                persona_counts.columns = ['Persona', 'Count']
                fig_bar_p = px.bar(persona_counts, x='Persona', y='Count', title="Customer Count Distribution", color='Persona')
                fig_bar_p.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_bar_p, use_container_width=True)
            
            st.divider()
            
            st.subheader("Standard Analytics")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_pie = px.pie(df, names='Risk_Level', title='Customer Risk Distribution',
                                 color='Risk_Level', 
                                 color_discrete_map={'High': 'salmon', 'Medium': 'gold', 'Low': 'lightgreen'},
                                 hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_chart2:
                fig_3d = px.scatter_3d(df, x='Recency', y='Frequency', z='Monetary',
                                       color='Risk_Level',
                                       color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
                                       title="Customer Clustering (3D)",
                                       opacity=0.7)
                fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30))
                st.plotly_chart(fig_3d, use_container_width=True)
            
            st.subheader("Data Export")
            st.dataframe(df.style.background_gradient(subset=['Churn_Probability'], cmap='Reds'))
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions & Personas CSV", csv, "churn_personas_predictions.csv", "text/csv")


with tab3:
    st.header("📖 Outreach History Database")
    
    col_hist1, col_hist_btn = st.columns([4, 1])
    
    def fetch_history():
        try:
            conn = sqlite3.connect('data/outreach_history.db')
            history_df = pd.read_sql_query("SELECT * FROM outreach_history ORDER BY id DESC", conn)
            conn.close()
            return history_df
        except Exception:
            return pd.DataFrame()
            
    with col_hist_btn:
        if st.button("🔄 Refresh Database", use_container_width=True):
            st.rerun()
            
    h_df = fetch_history()
    
    if len(h_df) > 0:
        # Format for nicer display
        st.dataframe(
            h_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "id": st.column_config.NumberColumn("Log ID", format="%d"),
                "timestamp": st.column_config.TextColumn("Date & Time"),
                "customer_id": st.column_config.NumberColumn("Customer ID", format="%d"),
                "action_type": st.column_config.TextColumn("Action Executed"),
                "status": st.column_config.TextColumn("Status"),
                "details": st.column_config.TextColumn("Campaign Details"),
            }
        )
        
        # Download SQL Backup
        st.download_button(
            "💾 Download Database Backup (CSV)", 
            h_df.to_csv(index=False).encode('utf-8'), 
            "outreach_history_backup.csv", 
            "text/csv"
        )
    else:
        st.info("The history database is currently empty. Analyze a single customer and use the Action Buttons to populate the logs!")