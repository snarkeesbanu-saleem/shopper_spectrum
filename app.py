# app.py – Shopper Spectrum Dashboard (Fully Corrected)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Shopper Spectrum", layout="wide", page_icon="🛒")

# ------------------- THEME & SESSION STATE -------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Default"

def get_theme_style(theme):
    if theme == "Neon":
        bg = "#0a0f0a"
        card_gradient = "linear-gradient(135deg, #00ff88, #00bcd4)"
        accent = "#00ff88"
        plotly_template = "plotly_dark"
    elif theme == "Cyan":
        bg = "#0a1a2a"
        card_gradient = "linear-gradient(135deg, #00e5ff, #00796b)"
        accent = "#00e5ff"
        plotly_template = "plotly_dark"
    elif theme == "Sunset":
        bg = "#1f0f1a"
        card_gradient = "linear-gradient(135deg, #ff6b6b, #ff8e53)"
        accent = "#ffaa66"
        plotly_template = "plotly_dark"
    else:  # Default
        bg = "#f8fafc"
        card_gradient = "linear-gradient(135deg, #3b82f6, #8b5cf6)"
        accent = "#3b82f6"
        plotly_template = "plotly_white"
    return bg, card_gradient, accent, plotly_template

def apply_theme():
    bg, grad, accent, template = get_theme_style(st.session_state.theme)
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg}; }}
        .metric-card {{
            background: {grad};
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            color: white;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        }}
        .metric-card:hover {{ transform: translateY(-5px); }}
        .metric-value {{ font-size: 2rem; font-weight: bold; }}
        .metric-label {{ font-size: 0.9rem; opacity: 0.9; }}
        .dashboard-title {{
            font-size: 2.5rem;
            font-weight: 800;
            background: {grad};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stSelectbox label, .stSlider label, .stRadio label {{ color: {accent} !important; }}
        [data-testid="stSidebar"] {{ background: {bg}; border-right: 2px solid {accent}; }}
        .sidebar-section {{
            background: rgba(255,255,255,0.08);
            border-radius: 15px;
            padding: 0.8rem;
            margin-bottom: 1.2rem;
        }}
        .sidebar-section-title {{ color: {accent}; }}
        .stButton button {{
            background: {grad};
            color: white;
            border: none;
            border-radius: 25px;
        }}
    </style>
    """, unsafe_allow_html=True)
    import plotly.io as pio
    pio.templates.default = template

apply_theme()

# ------------------- DATA LOADING -------------------
@st.cache_data
def load_raw_data():
    retail_path = "online_retail.csv"
    if not os.path.exists(retail_path):
        st.error("online_retail.csv not found. Please upload the dataset.")
        st.stop()
    retail = pd.read_csv(retail_path, encoding='latin1')
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])
    retail = retail[retail['Quantity'] > 0]
    retail = retail[retail['CustomerID'].notna()]
    retail['CustomerID'] = retail['CustomerID'].astype(int)
    retail['TotalPrice'] = retail['Quantity'] * retail['UnitPrice']
    return retail

@st.cache_data
def compute_rfm(retail):
    snapshot_date = retail['InvoiceDate'].max() + pd.DateOffset(days=1)
    rfm = retail.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})
    rfm = rfm[rfm['Monetary'] > 0]
    try:
        rfm['R_quartile'] = pd.qcut(rfm['Recency'], 4, labels=['4','3','2','1'], duplicates='drop')
        rfm['F_quartile'] = pd.qcut(rfm['Frequency'], 4, labels=['1','2','3','4'], duplicates='drop')
        rfm['M_quartile'] = pd.qcut(rfm['Monetary'], 4, labels=['1','2','3','4'], duplicates='drop')
    except:
        rfm['R_quartile'] = pd.cut(rfm['Recency'], bins=4, labels=['4','3','2','1'])
        rfm['F_quartile'] = pd.cut(rfm['Frequency'], bins=4, labels=['1','2','3','4'])
        rfm['M_quartile'] = pd.cut(rfm['Monetary'], bins=4, labels=['1','2','3','4'])
    rfm['RFM_Score'] = rfm['R_quartile'].astype(str) + rfm['F_quartile'].astype(str) + rfm['M_quartile'].astype(str)
    seg_map = {
        '444': 'Champions', '443': 'Loyal', '434': 'Loyal', '344': 'Potential',
        '433': 'New', '333': 'Promising', '322': 'Need Attention', '311': 'About to Sleep',
        '211': 'At Risk', '111': 'Hibernating', '144': 'Can\'t Lose'
    }
    rfm['Segment'] = rfm['RFM_Score'].map(seg_map).fillna('Others')
    rfm = rfm.reset_index()
    return rfm

@st.cache_data
def compute_cohorts(retail):
    first_purchase = retail.groupby('CustomerID')['InvoiceDate'].min().reset_index()
    first_purchase.columns = ['CustomerID', 'FirstPurchase']
    first_purchase['CohortMonth'] = first_purchase['FirstPurchase'].dt.to_period('M').astype(str)  # convert to string
    retail_cohort = retail.merge(first_purchase[['CustomerID','CohortMonth']], on='CustomerID')
    retail_cohort['InvoiceMonth'] = retail_cohort['InvoiceDate'].dt.to_period('M')
    retail_cohort['CohortIndex'] = (retail_cohort['InvoiceMonth'].dt.year - retail_cohort['CohortMonth'].str.split('-').str[0].astype(int))*12 + (retail_cohort['InvoiceMonth'].dt.month - retail_cohort['CohortMonth'].str.split('-').str[1].astype(int))
    # Retention
    retention = retail_cohort.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    cohort_size = retention[retention['CohortIndex']==0][['CohortMonth','CustomerID']].rename(columns={'CustomerID':'CohortSize'})
    retention = retention.merge(cohort_size, on='CohortMonth')
    retention['RetentionRate'] = retention['CustomerID'] / retention['CohortSize']
    # Revenue
    revenue = retail_cohort.groupby(['CohortMonth', 'CohortIndex'])['TotalPrice'].sum().reset_index()
    revenue = revenue.merge(cohort_size, on='CohortMonth')
    revenue['AvgRevenue'] = revenue['TotalPrice'] / revenue['CohortSize']
    return retention, revenue

@st.cache_data
def compute_product_affinity(retail):
    basket = retail.groupby('InvoiceNo')['StockCode'].apply(list).reset_index()
    all_items = retail['StockCode'].value_counts().head(100).index.tolist()
    pair_counts = {}
    for items in basket['StockCode']:
        items_clean = [i for i in items if i in all_items]
        for i in range(len(items_clean)):
            for j in range(i+1, len(items_clean)):
                pair = tuple(sorted([items_clean[i], items_clean[j]]))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    desc_map = retail[['StockCode','Description']].drop_duplicates().set_index('StockCode')['Description'].to_dict()
    result = []
    for (a,b), cnt in top_pairs:
        result.append({'Item A': a, 'Item A Desc': desc_map.get(a, a),
                       'Item B': b, 'Item B Desc': desc_map.get(b, b), 'Count': cnt})
    return pd.DataFrame(result)

@st.cache_data
def predict_clv(rfm):
    X = rfm[['Frequency']].values
    y = rfm['Monetary'].values
    model = LinearRegression()
    model.fit(X, y)
    rfm['Predicted_CLV'] = np.clip(model.predict(X), 0, None)
    return rfm

def predict_next_purchase(rfm):
    rfm['NextPurchaseDays'] = rfm['Recency'] / (rfm['Frequency'] + 0.1)
    rfm['NextPurchaseDays'] = rfm['NextPurchaseDays'].clip(lower=1, upper=180)
    return rfm

def assign_churn_risk(rfm):
    conditions = [
        (rfm['Recency'] > 90) & (rfm['Frequency'] < 3),
        (rfm['Recency'] > 60) | (rfm['Frequency'] < 5),
        True
    ]
    choices = ['High', 'Medium', 'Low']
    rfm['ChurnRisk'] = np.select(conditions, choices, default='Low')
    return rfm

# Load all data
with st.spinner("Loading data and computing analytics..."):
    retail = load_raw_data()
    rfm_full = compute_rfm(retail)
    retention, revenue_cohort = compute_cohorts(retail)
    affinity_df = compute_product_affinity(retail)
    rfm_full = predict_clv(rfm_full)
    rfm_full = predict_next_purchase(rfm_full)
    rfm_full = assign_churn_risk(rfm_full)

# ------------------- QUERY PARAMS & SIDEBAR -------------------
def sync_filters_from_url():
    params = st.query_params
    segment = params.get("segment", "All")
    recency_min = int(params.get("recency_min", 0))
    recency_max = int(params.get("recency_max", 365))
    freq_min = int(params.get("freq_min", 1))
    freq_max = int(params.get("freq_max", 100))
    monetary_min = float(params.get("monetary_min", 0))
    monetary_max = float(params.get("monetary_max", 10000))
    return segment, recency_min, recency_max, freq_min, freq_max, monetary_min, monetary_max

segment_default, rmin, rmax, fmin, fmax, mmin, mmax = sync_filters_from_url()

st.sidebar.markdown('<div class="sidebar-section-title">🛒 Shopper Spectrum</div>', unsafe_allow_html=True)
theme_options = ["Default", "Neon", "Cyan", "Sunset"]
selected_theme = st.sidebar.selectbox("Theme", theme_options, index=theme_options.index(st.session_state.theme))
if selected_theme != st.session_state.theme:
    st.session_state.theme = selected_theme
    st.rerun()

st.sidebar.markdown("---")
segments = ["All"] + sorted(rfm_full['Segment'].dropna().unique())
selected_segment = st.sidebar.selectbox("Customer Segment", segments, index=segments.index(segment_default) if segment_default in segments else 0)
recency_range = st.sidebar.slider("Recency (days)", 0, 365, (rmin, rmax), step=1)
freq_range = st.sidebar.slider("Frequency (orders)", 1, 200, (fmin, fmax), step=1)
monetary_range = st.sidebar.slider("Monetary (total spend)", 0, 50000, (int(mmin), int(mmax)), step=1000)

st.query_params["segment"] = selected_segment
st.query_params["recency_min"] = recency_range[0]
st.query_params["recency_max"] = recency_range[1]
st.query_params["freq_min"] = freq_range[0]
st.query_params["freq_max"] = freq_range[1]
st.query_params["monetary_min"] = monetary_range[0]
st.query_params["monetary_max"] = monetary_range[1]

if st.sidebar.button("🔄 Auto-refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
search_id = st.sidebar.text_input("🔍 Search Customer by ID")
if search_id:
    try:
        cust_id = int(search_id)
        cust_data = rfm_full[rfm_full['CustomerID']==cust_id]
        if not cust_data.empty:
            st.sidebar.success(f"Found customer {cust_id}")
            st.sidebar.write(f"Segment: {cust_data['Segment'].iloc[0]}")
            st.sidebar.write(f"Recency: {cust_data['Recency'].iloc[0]} days")
            st.sidebar.write(f"Frequency: {cust_data['Frequency'].iloc[0]} orders")
            st.sidebar.write(f"Monetary: ₹{cust_data['Monetary'].iloc[0]:,.2f}")
            st.sidebar.write(f"Churn Risk: {cust_data['ChurnRisk'].iloc[0]}")
            st.sidebar.write(f"Predicted CLV: ₹{cust_data['Predicted_CLV'].iloc[0]:,.2f}")
    except:
        st.sidebar.error("Invalid ID")

# ------------------- FILTER MAIN DATA -------------------
mask = (rfm_full['Recency'].between(recency_range[0], recency_range[1])) & \
       (rfm_full['Frequency'].between(freq_range[0], freq_range[1])) & \
       (rfm_full['Monetary'].between(monetary_range[0], monetary_range[1]))
if selected_segment != "All":
    mask &= (rfm_full['Segment'] == selected_segment)
filtered_df = rfm_full[mask]

if filtered_df.empty:
    st.warning("No customers match the filters.")
    st.stop()

# ------------------- KPI CARDS -------------------
total_customers = len(filtered_df)
avg_recency = filtered_df['Recency'].mean()
avg_frequency = filtered_df['Frequency'].mean()
avg_monetary = filtered_df['Monetary'].mean()
avg_clv = filtered_df['Predicted_CLV'].mean()

st.markdown('<div class="dashboard-title">Shopper Spectrum</div>', unsafe_allow_html=True)
st.caption("Customer segmentation & advanced analytics dashboard")

col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown(f'<div class="metric-card"><div class="metric-label">👥 Total Customers</div><div class="metric-value">{total_customers:,}</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="metric-card"><div class="metric-label">📅 Avg Recency (days)</div><div class="metric-value">{avg_recency:.1f}</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="metric-card"><div class="metric-label">🔄 Avg Frequency</div><div class="metric-value">{avg_frequency:.1f}</div></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="metric-card"><div class="metric-label">💰 Avg Monetary (₹)</div><div class="metric-value">{avg_monetary:,.0f}</div></div>', unsafe_allow_html=True)
col5.markdown(f'<div class="metric-card"><div class="metric-label">📈 Predicted CLV</div><div class="metric-value">₹{avg_clv:,.0f}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ------------------- CHARTS (All features) -------------------
# Churn risk
st.subheader("⚠️ Churn Risk Distribution")
risk_counts = filtered_df['ChurnRisk'].value_counts().reset_index()
risk_counts.columns = ['Risk', 'Count']
fig_risk = px.pie(risk_counts, values='Count', names='Risk', hole=0.3, color_discrete_sequence=['#ef4444','#f59e0b','#10b981'])
st.plotly_chart(fig_risk, use_container_width=True, key="risk")

# Cohort Analysis (now with string months)
st.subheader("📆 Cohort Analysis – Retention Heatmap")
retention_pivot = retention.pivot(index='CohortMonth', columns='CohortIndex', values='RetentionRate')
fig_cohort = px.imshow(retention_pivot, text_auto='.0%', aspect='auto', color_continuous_scale='Blues',
                       labels=dict(x="Months after first purchase", y="Cohort Month", color="Retention Rate"))
st.plotly_chart(fig_cohort, use_container_width=True, key="cohort")

st.subheader("💰 Revenue per Customer by Cohort")
revenue_pivot = revenue_cohort.pivot(index='CohortMonth', columns='CohortIndex', values='AvgRevenue')
fig_rev_cohort = px.imshow(revenue_pivot, text_auto='.0f', aspect='auto', color_continuous_scale='Greens',
                           labels=dict(x="Months after first purchase", y="Cohort Month", color="Avg Revenue"))
st.plotly_chart(fig_rev_cohort, use_container_width=True, key="rev_cohort")

# Product affinity
st.subheader("🛍️ Product Affinity (Top 10 Frequently Bought Together)")
if not affinity_df.empty:
    st.dataframe(affinity_df, use_container_width=True)
else:
    st.info("Not enough data for product affinity.")

# CLV distribution
st.subheader("💰 Customer Lifetime Value (CLV) Distribution")
fig_clv = px.histogram(filtered_df, x='Predicted_CLV', nbins=30, log_x=True, title="")
st.plotly_chart(fig_clv, use_container_width=True, key="clv")

# Next purchase prediction
st.subheader("📅 Next Purchase Prediction (Days)")
fig_next = px.histogram(filtered_df, x='NextPurchaseDays', nbins=30, title="")
st.plotly_chart(fig_next, use_container_width=True, key="next_purchase")

# Segment comparison
st.subheader("⚖️ Segment Comparison Tool")
seg_list = filtered_df['Segment'].unique()
if len(seg_list) >= 2:
    col_a, col_b = st.columns(2)
    with col_a:
        seg_a = st.selectbox("Compare Segment A", seg_list, index=0, key="seg_a")
    with col_b:
        seg_b = st.selectbox("Compare Segment B", seg_list, index=min(1, len(seg_list)-1), key="seg_b")
    if seg_a and seg_b:
        comp_df = filtered_df[filtered_df['Segment'].isin([seg_a, seg_b])]
        fig_comp = px.box(comp_df, x='Segment', y='Monetary', color='Segment', title=f"Monetary: {seg_a} vs {seg_b}")
        st.plotly_chart(fig_comp, use_container_width=True, key="comp_box")
        profile_a = comp_df[comp_df['Segment']==seg_a][['Recency','Frequency','Monetary']].mean()
        profile_b = comp_df[comp_df['Segment']==seg_b][['Recency','Frequency','Monetary']].mean()
        comp_radar = pd.DataFrame({
            'Metric': ['Recency', 'Frequency', 'Monetary'],
            seg_a: [profile_a['Recency'], profile_a['Frequency'], profile_a['Monetary']],
            seg_b: [profile_b['Recency'], profile_b['Frequency'], profile_b['Monetary']]
        })
        fig_radar_comp = px.line_polar(comp_radar.melt(id_vars='Metric'), r='value', theta='Metric', color='variable', line_close=True)
        st.plotly_chart(fig_radar_comp, use_container_width=True, key="comp_radar")
else:
    st.info("Need at least two segments to compare.")

# What-If Simulator
st.subheader("🎛️ What-If Simulator – Segment Prediction")
sim_rec = st.number_input("Recency (days)", 0, 365, 30)
sim_freq = st.number_input("Frequency (orders)", 1, 200, 10)
sim_mon = st.number_input("Monetary (₹)", 0, 50000, 1000)
thresholds = {}
for metric in ['Recency','Frequency','Monetary']:
    q = rfm_full[metric].quantile([0.25,0.5,0.75]).values
    thresholds[metric] = q
r_quart = '4' if sim_rec <= thresholds['Recency'][0] else '3' if sim_rec <= thresholds['Recency'][1] else '2' if sim_rec <= thresholds['Recency'][2] else '1'
f_quart = '1' if sim_freq <= thresholds['Frequency'][0] else '2' if sim_freq <= thresholds['Frequency'][1] else '3' if sim_freq <= thresholds['Frequency'][2] else '4'
m_quart = '1' if sim_mon <= thresholds['Monetary'][0] else '2' if sim_mon <= thresholds['Monetary'][1] else '3' if sim_mon <= thresholds['Monetary'][2] else '4'
rfm_score = r_quart + f_quart + m_quart
pred_seg = {'444':'Champions','443':'Loyal','434':'Loyal','344':'Potential','433':'New','333':'Promising','322':'Need Attention','311':'About to Sleep','211':'At Risk','111':'Hibernating','144':'Can\'t Lose'}.get(rfm_score, 'Others')
st.success(f"Predicted Segment: **{pred_seg}** (RFM Score: {rfm_score})")

# Customer Health Gauge
st.subheader("📊 Overall Customer Health Index")
norm_rec = 1 - (avg_recency / 365) if avg_recency else 0.5
norm_freq = avg_frequency / 200 if avg_frequency else 0.5
norm_mon = avg_monetary / 50000 if avg_monetary else 0.5
health_score = (norm_rec + norm_freq + norm_mon) / 3 * 100
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health_score,
    title={'text': "Customer Health Index (%)"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "darkblue"},
           'steps': [{'range': [0, 30], 'color': "red"},
                     {'range': [30, 70], 'color': "yellow"},
                     {'range': [70, 100], 'color': "green"}],
           'threshold': {'value': 70, 'line': {'color': "orange", 'width': 2}}}))
st.plotly_chart(fig_gauge, use_container_width=True, key="gauge")

# Current segment mix
st.subheader("📈 Current Segment Mix")
segment_counts = filtered_df['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']
fig_current = px.pie(segment_counts, values='Count', names='Segment', hole=0.3)
st.plotly_chart(fig_current, use_container_width=True, key="current_seg")

# Top products bar chart
st.subheader("☁️ Top Product Descriptions")
top_products = retail['Description'].value_counts().head(20).reset_index()
top_products.columns = ['Description', 'Frequency']
fig_cloud = px.bar(top_products, x='Frequency', y='Description', orientation='h', title="Most Frequent Products")
st.plotly_chart(fig_cloud, use_container_width=True, key="cloud")

# Scatter plot
st.subheader("📉 Recency vs Monetary (Scatter)")
fig_scatter = px.scatter(filtered_df, x='Recency', y='Monetary', color='Segment', size='Frequency',
                         hover_data=['CustomerID'], log_y=True)
st.plotly_chart(fig_scatter, use_container_width=True, key="scatter")

# Heatmap
st.subheader("🔥 Heatmap: Avg Monetary (Recency vs Frequency)")
filtered_df['Recency_bin'] = pd.cut(filtered_df['Recency'], bins=5, labels=['Very Recent','Recent','Moderate','Old','Very Old'])
filtered_df['Frequency_bin'] = pd.cut(filtered_df['Frequency'], bins=5, labels=['Very Low','Low','Medium','High','Very High'])
heat_data = filtered_df.pivot_table(index='Recency_bin', columns='Frequency_bin', values='Monetary', aggfunc='mean')
fig_heat = px.imshow(heat_data, text_auto='.0f', aspect='auto', color_continuous_scale='Viridis')
st.plotly_chart(fig_heat, use_container_width=True, key="heat")

# Boxplot
st.subheader("📦 Monetary by Segment (Boxplot)")
fig_box = px.box(filtered_df, x='Segment', y='Monetary', color='Segment')
st.plotly_chart(fig_box, use_container_width=True, key="box")

# Treemap
st.subheader("🌳 Treemap of Segments")
fig_treemap = px.treemap(segment_counts, path=['Segment'], values='Count')
st.plotly_chart(fig_treemap, use_container_width=True, key="treemap")

# 3D Scatter
st.subheader("🎯 3D RFM Space")
fig_3d = px.scatter_3d(filtered_df, x='Recency', y='Frequency', z='Monetary', color='Segment', size_max=8)
st.plotly_chart(fig_3d, use_container_width=True, key="3d")

# Radar
st.subheader("📡 Segment Profile Radar (Normalized)")
segment_profile = filtered_df.groupby('Segment')[['Recency','Frequency','Monetary']].mean()
for col in ['Recency','Frequency','Monetary']:
    seg_min, seg_max = segment_profile[col].min(), segment_profile[col].max()
    if seg_max != seg_min:
        segment_profile[col+'_norm'] = (segment_profile[col] - seg_min) / (seg_max - seg_min)
    else:
        segment_profile[col+'_norm'] = 0.5
radar_df = segment_profile.reset_index().melt(id_vars='Segment', value_vars=['Recency_norm','Frequency_norm','Monetary_norm'],
                                              var_name='Metric', value_name='Value')
fig_radar = px.line_polar(radar_df, r='Value', theta='Metric', color='Segment', line_close=True)
st.plotly_chart(fig_radar, use_container_width=True, key="radar")

# Data table & download
st.subheader("📋 Customer Data (Filtered)")
st.dataframe(filtered_df, use_container_width=True, height=400)
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Filtered Data", data=csv, file_name="shopper_segments.csv", mime='text/csv')

# PDF export
if st.button("🖨️ Print / Save as PDF"):
    st.markdown("<script>window.print();</script>", unsafe_allow_html=True)

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")