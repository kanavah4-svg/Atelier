import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Atelier 8 - Dashboard",
    page_icon="👜",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e1e1e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">👜 ATELIER 8 Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Circular Luxury Restoration & Authentication Studio - UAE Market Analysis</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Dashboard Navigation")
    page = st.radio(
        "Select Analysis:",
        ["Overview", "Customer Insights", "Predictive Analytics", "Service Analysis", "Financial Projections"]
    )
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_raw_data = st.checkbox("Show Raw Data Tables", value=False)

@st.cache_data
def load_data():
    df = pd.read_csv("atelier8_customer_survey_data.csv")
    return df

df = load_data()

if page == "Overview":
    st.markdown("## 📈 Business Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="📊 Survey Respondents", value="600", delta="UAE Market")
    with col2:
        adoption_rate = (df['Adoption_Likelihood'].isin(['Likely', 'Very Likely']).sum() / len(df)) * 100
        st.metric(label="✅ Adoption Rate", value=f"{adoption_rate:.1f}%", delta="High Intent")
    with col3:
        avg_wtp = df['WTP_Basic_Restoration_AED'].mean()
        st.metric(label="💰 Avg WTP (Restoration)", value=f"AED {avg_wtp:.0f}", delta="Per Service")
    with col4:
        sustainability = (df['Sustainability_Importance'].isin(['Important', 'Very Important']).sum() / len(df)) * 100
        st.metric(label="🌱 Sustainability Focus", value=f"{sustainability:.1f}%", delta="Customer Base")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Core Services")
        services = pd.DataFrame({
            'Service': ['Restoration', 'Authentication', 'Resale Concierge', 'Membership Plans'],
            'Price Range (AED)': ['800 - 5,000', '250 - 500', '20-30% Commission', '5,000 - 25,000/year'],
            'Target Segment': ['All', 'Collectors', 'Rotation', 'Loyal']
        })
        st.dataframe(services, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### 📊 Brand Distribution")
        brand_counts = df['Primary_Brand_Owned'].value_counts().head(6)
        fig = px.pie(values=brand_counts.values, names=brand_counts.index, title="Customer Brand Ownership")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Customer Insights":
    st.markdown("## 👥 Customer Insights & Segmentation")
    df_cluster = df.copy()
    df_cluster['Income_Numeric'] = df_cluster['Income_Level'].map({
        'Low (<100K AED)': 1, 'Medium (100K-250K AED)': 2, 
        'High (250K-500K AED)': 3, 'Very High (>500K AED)': 4
    })
    df_cluster['Adoption_Numeric'] = df_cluster['Adoption_Likelihood'].map({
        'Very Unlikely': 1, 'Unlikely': 2, 'Neutral': 3, 'Likely': 4, 'Very Likely': 5
    })
    df_cluster['Sustainability_Numeric'] = df_cluster['Sustainability_Importance'].map({
        'Not Important': 1, 'Slightly Important': 2, 'Moderately Important': 3, 
        'Important': 4, 'Very Important': 5
    })
    features = ['Age', 'Income_Numeric', 'WTP_Basic_Restoration_AED', 'Num_Luxury_Items', 'Adoption_Numeric', 'Sustainability_Numeric']
    X_cluster = df_cluster[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_cluster['Segment'] = kmeans.fit_predict(X_scaled)
    segment_names = {0: 'Investment Collectors', 1: 'Rotation Enthusiasts',
                    2: 'Conscious Curators', 3: 'Hype Sneaker Owners'}
    df_cluster['Segment_Name'] = df_cluster['Segment'].map(segment_names)
    col1, col2 = st.columns(2)
    with col1:
        segment_counts = df_cluster['Segment_Name'].value_counts()
        fig = px.bar(x=segment_counts.index, y=segment_counts.values, title="Customer Segment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x='Adoption_Likelihood', y='Age', color='Adoption_Likelihood', title="Age vs Adoption")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Predictive Analytics":
    st.markdown("## 🤖 Predictive Analytics & Machine Learning")
    st.info("Classification Analysis: Predicting customer purchase likelihood")
    df_model = df.copy()
    df_model['Income_Numeric'] = df_model['Income_Level'].map({
        'Low (<100K AED)': 1, 'Medium (100K-250K AED)': 2,
        'High (250K-500K AED)': 3, 'Very High (>500K AED)': 4
    })
    df_model['Adoption_Numeric'] = df_model['Adoption_Likelihood'].map({
        'Very Unlikely': 1, 'Unlikely': 2, 'Neutral': 3, 'Likely': 4, 'Very Likely': 5
    })
    df_model['Sustainability_Numeric'] = df_model['Sustainability_Importance'].map({
        'Not Important': 1, 'Slightly Important': 2, 'Moderately Important': 3,
        'Important': 4, 'Very Important': 5
    })
    features = ['Age', 'Income_Numeric', 'WTP_Basic_Restoration_AED', 'Num_Luxury_Items', 'Adoption_Numeric', 'Sustainability_Numeric']
    X = df_model[features].fillna(0)
    y = df_model['Has_Purchased']
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(feature_importance, x='Coefficient', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        accuracy = model.score(X, y) * 100
        st.metric("Model Accuracy", f"{accuracy:.1f}%")

elif page == "Service Analysis":
    st.markdown("## 🛠️ Service Analysis & Pricing Strategy")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 💰 Willingness to Pay by Brand")
        wtp_by_brand = df.groupby('Primary_Brand_Owned')['WTP_Basic_Restoration_AED'].mean().sort_values(ascending=False).head(8)
        fig = px.bar(x=wtp_by_brand.index, y=wtp_by_brand.values, title="Average WTP for Restoration by Brand")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### 📊 Services Interested In")
        services = df['Services_Interested_In'].value_counts()
        fig = px.pie(values=services.values, names=services.index, title="Service Interest Distribution")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Financial Projections":
    st.markdown("## 💰 Financial Projections & Business Sustainability")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    base_revenue = 120000
    growth_factor = 1.12
    monthly_revenue = [base_revenue * (growth_factor ** i) for i in range(12)]
    df_revenue = pd.DataFrame({'Month': months, 'Revenue': monthly_revenue})
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.line(df_revenue, x='Month', y='Revenue', title='Monthly Revenue Projection (Year 1)', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        total_annual = sum(monthly_revenue)
        st.metric("Total Annual Revenue", f"AED {total_annual:,.0f}")
        st.metric("Monthly Growth Rate", f"{(growth_factor - 1) * 100:.1f}%")
        st.metric("Break-even Month", "Month 4")

if show_raw_data:
    st.markdown("---")
    st.markdown("### 📋 Raw Data")
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><p><strong>ATELIER 8</strong> | Circular Luxury Analytics Dashboard | 2025</p></div>", unsafe_allow_html=True)
