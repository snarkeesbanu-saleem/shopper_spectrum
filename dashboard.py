import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page Config
st.set_page_config(
    page_title="Shopper Spectrum Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 Shopper Spectrum - Customer Analytics Dashboard")
st.markdown("RFM Analysis • Segmentation • 12 Visualizations")
st.markdown("---")

# ───────────────────────────────────────────────
# Load & Clean Data
# ───────────────────────────────────────────────
@st.cache_data
def load_data():
    candidates = [
        "online_retail.csv",
        "OnlineRetail.csv.csv",
        "cleaned_onlineReatil.csv",
        "data/online_retail.csv",
        "data/cleaned_onlineReatil.csv"
    ]

    df = None
    used_file = None

    for file in candidates:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file, encoding="ISO-8859-1", on_bad_lines='warn')
                used_file = file
                break
            except:
                pass

    if df is None:
        st.error("No valid CSV found. Please place online_retail.csv in the project folder.")
        st.stop()

    # Cleaning
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df[df['CustomerID'].notna()]
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df[df['InvoiceDate'].notna()]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)

    return df

with st.spinner("Loading data..."):
    raw_df = load_data()

# ───────────────────────────────────────────────
# Sidebar Filters
# ───────────────────────────────────────────────
st.sidebar.header("Filters")

date_min = raw_df['InvoiceDate'].min().date()
date_max = raw_df['InvoiceDate'].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

countries = sorted(raw_df['Country'].unique())
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries,
    default=["United Kingdom"]
)

# Apply filters
df = raw_df.copy()
if len(date_range) == 2:
    start, end = date_range
    df = df[(df['InvoiceDate'].dt.date >= start) & (df['InvoiceDate'].dt.date <= end)]

if selected_countries:
    df = df[df['Country'].isin(selected_countries)]

# ───────────────────────────────────────────────
# RFM Calculation
# ───────────────────────────────────────────────
@st.cache_data
def compute_rfm(_df):
    if _df.empty:
        return pd.DataFrame()

    snapshot = _df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = _df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1], duplicates='drop')
    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4], duplicates='drop')
    rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4], duplicates='drop')
    
    rfm['RFM'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
    
    def segment(r):
        if r in ['444','443','434','433']:
            return 'Champion'
        if r.startswith('4') and int(r[1]) >= 2:
            return 'Loyal'
        if r.startswith('3'):
            return 'Potential'
        if r in ['211','221','212','111','112','121']:
            return 'At Risk'
        return 'Other'
    
    rfm['Segment'] = rfm['RFM'].apply(segment)
    return rfm

rfm = compute_rfm(df)

# ───────────────────────────────────────────────
# Safe size helper
# ───────────────────────────────────────────────
def safe_size_col(series, min_size=5, buffer=10, max_size=60):
    s = series.copy()
    if s.min() < 0:
        s = s - s.min() + buffer
    return s.clip(lower=min_size, upper=max_size)

# ───────────────────────────────────────────────
# KPIs
# ───────────────────────────────────────────────
cols = st.columns(4)
cols[0].metric("Customers", f"{len(rfm):,}")
cols[1].metric("Revenue", f"₹{df['TotalPrice'].sum():,.0f}")
cols[2].metric("Avg Order", f"₹{df.groupby('InvoiceNo')['TotalPrice'].sum().mean():,.0f}" if not df.empty else "₹0")
cols[3].metric("Active (90d)", f"{len(rfm[rfm['Recency'] <= 90]):,}" if 'Recency' in rfm.columns else "0")

st.markdown("---")

# ───────────────────────────────────────────────
# Tabs
# ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 RFM & Segments", "📈 Visualizations", "🔍 Tables"])

with tab1:
    st.subheader("Customer Segments")
    c1, c2 = st.columns([3, 2])
    
    with c1:
        seg_count = rfm['Segment'].value_counts().reset_index()
        seg_count.columns = ['Segment', 'Count']
        fig_pie = px.pie(seg_count, values='Count', names='Segment',
                         title='Segment Distribution', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with c2:
        st.subheader("Segment Profile")
        profile = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(1)
        profile['Customers'] = rfm['Segment'].value_counts()
        st.dataframe(profile.sort_values('Customers', ascending=False))

with tab2:
    st.subheader("Visualizations")
    
    if df.empty:
        st.warning("No transactions after filtering")
    elif rfm.empty:
        st.warning("No customers after filtering")
    else:
        viz_choice = st.selectbox("Choose visualization", [
            "1. Revenue by Country (Top 10)",
            "2. Recency vs Frequency (size = Monetary)",
            "3. Monetary Distribution (Histogram)",
            "4. Top 10 Products by Revenue",
            "5. RFM per Segment (Box)",
            "6. Average RFM Radar Chart",
            "7. Monthly Revenue Trend",
            "8. Customers per Segment (Bar)",
            "9. Frequency vs Monetary (size = Recency)",
            "10. Recency by Segment (Histogram)",
            "11. Top 30 Customers (Treemap)",
            "12. R vs F Heatmap"
        ])
        
        try:
            if viz_choice == "1. Revenue by Country (Top 10)":
                top = df.groupby('Country')['TotalPrice'].sum().nlargest(10).reset_index()
                fig = px.bar(top, x='TotalPrice', y='Country', orientation='h', color='TotalPrice')
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_choice == "2. Recency vs Frequency (size = Monetary)":
                fig = px.scatter(rfm, x='Recency', y='Frequency', color='Segment',
                                 size=safe_size_col(rfm['Monetary']),
                                 hover_name='CustomerID', opacity=0.7,
                                 title="Recency vs Frequency – size = Monetary (adjusted)")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_choice == "3. Monetary Distribution (Histogram)":
                fig = px.histogram(rfm, x='Monetary', color='Segment', nbins=50,
                                   title="Monetary Value Distribution", marginal="box")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_choice == "4. Top 10 Products by Revenue":
                top_prod = df.groupby('Description')['TotalPrice'].sum().nlargest(10).reset_index()
                fig = px.bar(top_prod, x='TotalPrice', y='Description', orientation='h',
                             title="Top 10 Products", color='TotalPrice')
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_choice == "5. RFM per Segment (Box)":
                melted = rfm.melt(id_vars='Segment', value_vars=['Recency','Frequency','Monetary'],
                                  var_name='Metric', value_name='Value')
                fig = px.box(melted, x='Segment', y='Value', color='Metric',
                             title="RFM Box Plots by Segment")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_choice == "6. Average RFM Radar Chart":
                radar = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().reset_index()
                if len(radar) < 1:
                    st.info("Not enough segments for radar chart")
                else:
                    radar['Recency'] = 1 - (radar['Recency'] - radar['Recency'].min()) / (radar['Recency'].max() - radar['Recency'].min() + 1e-6)
                    for col in ['Frequency','Monetary']:
                        radar[col] = (radar[col] - radar[col].min()) / (radar[col].max() - radar[col].min() + 1e-6)
                    fig = go.Figure()
                    for _, row in radar.iterrows():
                        vals = row[['Recency','Frequency','Monetary']].tolist()
                        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=['Recency','Frequency','Monetary','Recency'],
                                                      fill='toself', name=row['Segment']))
                    fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_choice == "7. Monthly Revenue Trend":
                monthly = df.groupby('Month')['TotalPrice'].sum().reset_index()
                fig = px.line(monthly, x='Month', y='TotalPrice', markers=True,
                              title="Revenue Trend by Month")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_choice == "8. Customers per Segment (Bar)":
                cnt = rfm['Segment'].value_counts().reset_index(name='Count')
                fig = px.bar(cnt, x='Segment', y='Count', color='Segment',
                             title="Customer Count by Segment")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_choice == "9. Frequency vs Monetary (size = Recency)":
                fig = px.scatter(rfm, x='Frequency', y='Monetary', color='Segment',
                                 size=safe_size_col(rfm['Recency']),
                                 hover_name='CustomerID',
                                 title="Frequency vs Monetary – size = Recency (adjusted)")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_choice == "10. Recency by Segment (Histogram)":
                fig = px.histogram(rfm, x='Recency', color='Segment', barmode='overlay',
                                   title="Recency Distribution by Segment", nbins=40)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_choice == "11. Top 30 Customers (Treemap)":
                top30 = rfm.nlargest(30, 'Monetary').copy()
                top30['label'] = top30['CustomerID'] + "<br>₹" + top30['Monetary'].round(0).astype(str)
                fig = px.treemap(top30, path=['Segment','label'], values='Monetary',
                                 color='Monetary', title="Top 30 Customers by Value")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_choice == "12. R vs F Heatmap":
                if len(rfm) < 20:
                    st.info("Too few customers for heatmap")
                else:
                    temp = rfm.copy()
                    temp['R_bin'] = pd.qcut(temp['Recency'], 5, labels=['1','2','3','4','5'], duplicates='drop')
                    temp['F_bin'] = pd.qcut(temp['Frequency'], 5, labels=['1','2','3','4','5'], duplicates='drop')
                    hm = temp.pivot_table(index='R_bin', columns='F_bin', values='CustomerID', aggfunc='count', fill_value=0)
                    fig = px.imshow(hm.values, x=hm.columns, y=hm.index,
                                    color_continuous_scale='Blues', text_auto=True,
                                    title="Customer Count: Recency vs Frequency Bins")
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in visualization: {str(e)}")
            st.info("Try a different visualization or adjust filters")

with tab3:
    st.subheader("RFM Table")
    st.dataframe(rfm.sort_values('Monetary', ascending=False))
    
    st.subheader("Sample Transactions")
    st.dataframe(df.sample(min(1000, len(df))))

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')} • Shopper Spectrum")