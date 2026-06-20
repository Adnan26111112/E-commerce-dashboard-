"""
E-Commerce Sales Performance & Business Insights Dashboard
==========================================================
Run with:   streamlit run app.py
Deploys to: https://streamlit.io/cloud  (free)

All data files must be in the same folder as this script:
  /Users/adnan/Downloads/archive/
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths — relative paths so this works both locally AND when deployed ───────
# Locally: run this script from inside the project folder (where app.py lives)
# On Streamlit Cloud: the repo root becomes the working directory automatically
DATA_DIR   = './'
MODELS_DIR = './models/'

# ── Dark, moody, high-contrast color palette ───────────────────────────────────
BG          = '#0e1117'   # page background (matches Streamlit dark theme)
CARD_BG     = '#1a1d29'   # card / chart background
GRID        = 'rgba(255,255,255,0.08)'
TEXT_LIGHT  = '#e6e6e6'
TEXT_MUTED  = '#9ca3af'

ACCENT  = '#3b82f6'   # electric blue
WARM    = '#fb5607'   # vivid orange
GREEN   = '#10d97a'   # neon green
PURPLE  = '#a855f7'   # vivid purple
GOLD    = '#fbbf24'   # amber
RED     = '#ef4444'   # alert red
GREY    = '#6b7280'

SEGMENT_COLORS = {
    'Champions': ACCENT, 'Loyal': GREEN, 'Recent': GOLD,
    'At Risk': WARM, 'Lost': GREY, 'Potential': PURPLE
}

# Apply a dark Plotly template globally — every chart inherits this automatically
pio.templates.default = "plotly_dark"

# ── Custom CSS — dark, moody, high-contrast ────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    .metric-card {{
        background: {CARD_BG};
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid {ACCENT};
    }}
    .metric-value {{ font-size: 28px; font-weight: 700; color: {ACCENT}; }}
    .metric-label {{ font-size: 13px; color: {TEXT_MUTED}; margin-top: 2px; }}
    .section-header {{
        font-size: 20px; font-weight: 600; color: {TEXT_LIGHT};
        border-bottom: 2px solid {ACCENT};
        padding-bottom: 8px; margin-bottom: 20px;
    }}
    [data-testid="stMetricValue"] {{ font-size: 26px !important; color: {TEXT_LIGHT}; }}
    h1, h2, h3, h4, p, span, label {{ color: {TEXT_LIGHT}; }}
    .stDataFrame {{ background-color: {CARD_BG}; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — cached so it only runs once per session
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_master():
    df = pd.read_csv(
        os.path.join(DATA_DIR, 'master.csv'),
        parse_dates=['order_purchase_timestamp']
    )
    return df

@st.cache_data
def load_forecast():
    df = pd.read_csv(os.path.join(DATA_DIR, 'forecast.csv'), parse_dates=['ds'])
    return df

@st.cache_data
def load_monthly():
    df = pd.read_csv(os.path.join(DATA_DIR, 'monthly_revenue.csv'), parse_dates=['ds'])
    return df

@st.cache_data
def load_rfm():
    df = pd.read_csv(os.path.join(DATA_DIR, 'rfm.csv'))
    return df

@st.cache_data
def load_feature_importance():
    df = pd.read_csv(os.path.join(DATA_DIR, 'feature_importance.csv'))
    return df

@st.cache_data
def load_predictions():
    df = pd.read_csv(os.path.join(DATA_DIR, 'model_predictions.csv'))
    return df

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, 'churn_model.pkl'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))
    return model, feature_cols

# ── Load everything ───────────────────────────────────────────────────────────
try:
    master     = load_master()
    forecast   = load_forecast()
    monthly    = load_monthly()
    rfm        = load_rfm()
    feat_imp   = load_feature_importance()
    preds      = load_predictions()
    rf_model, feature_cols = load_model()
    data_loaded = True
except Exception as e:
    data_loaded = False
    load_error  = str(e)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🛒 E-Commerce Dashboard")
    st.markdown("**Olist Brazilian E-Commerce**")
    st.markdown("---")

    tab_selected = st.radio(
        "Navigate",
        ["📊 Executive Summary",
         "📈 Sales Performance",
         "👥 Customer Analytics",
         "📦 Product Insights",
         "🤖 ML Predictions",
         "🔮 Revenue Forecast"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    if data_loaded:
        st.success("✅ All data loaded")
        st.caption(f"Master: {master.shape[0]:,} rows")
        st.caption(f"Customers: {master['customer_unique_id'].nunique():,}")
        st.caption(f"Orders: {master['order_id'].nunique():,}")
    else:
        st.error(f"❌ Data load failed:\n{load_error}")

    st.markdown("---")
    st.caption("Built by Adnan | Data Analysis Portfolio Project")

# ── Stop here if data didn't load ─────────────────────────────────────────────
if not data_loaded:
    st.error("Could not load data files. Make sure all CSVs and model files are in the archive folder.")
    st.code(f"Expected path: {DATA_DIR}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if tab_selected == "📊 Executive Summary":
    st.title("📊 Executive Summary")
    st.markdown("Top-line business KPIs at a glance.")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    total_revenue    = master['price'].sum()
    total_orders     = master['order_id'].nunique()
    unique_customers = master['customer_unique_id'].nunique()
    avg_order_value  = master.groupby('order_id')['price'].sum().mean()
    avg_review       = master['review_score'].mean()
    repeat_rate      = (master.groupby('customer_unique_id')['order_id'].nunique() > 1).mean() * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Revenue",       f"R${total_revenue/1e6:.2f}M")
    c2.metric("Total Orders",        f"{total_orders:,}")
    c3.metric("Unique Customers",    f"{unique_customers:,}")
    c4.metric("Avg Order Value",     f"R${avg_order_value:.2f}")
    c5.metric("Avg Review Score",    f"{avg_review:.2f} / 5.0")
    c6.metric("Repeat Purchase Rate",f"{repeat_rate:.1f}%")

    st.markdown("---")

    # ── Revenue trend ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Monthly Revenue Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly['ds'], y=monthly['y'],
            fill='tozeroy', fillcolor='rgba(26,108,245,0.1)',
            line=dict(color=ACCENT, width=2.5),
            mode='lines+markers', marker=dict(size=5),
            name='Revenue'
        ))
        fig.update_yaxes(tickprefix='R$', tickformat=',.0f')
        fig.update_layout(
            height=300, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor=GRID)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Review Score Distribution")
        rev_counts = master['review_score'].value_counts().sort_index()
        colors = [RED, WARM, GOLD, GREEN, ACCENT]
        fig = go.Figure(go.Bar(
            x=rev_counts.index.astype(str),
            y=rev_counts.values,
            marker_color=colors,
            text=rev_counts.values,
            textposition='outside'
        ))
        fig.update_layout(
            height=300, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
            xaxis_title='Score', yaxis=dict(showgrid=False, showticklabels=False)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── RFM segments ──────────────────────────────────────────────────────────
    st.markdown("#### Customer Segments (RFM)")
    seg_counts = rfm['segment'].value_counts().reset_index()
    seg_counts.columns = ['segment', 'count']
    color_map = SEGMENT_COLORS
    fig = px.bar(
        seg_counts, x='segment', y='count',
        color='segment', color_discrete_map=color_map,
        text='count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=300, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
        xaxis_title='', yaxis=dict(showgrid=False, showticklabels=False)
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SALES PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_selected == "📈 Sales Performance":
    st.title("📈 Sales Performance")

    # ── Year filter ───────────────────────────────────────────────────────────
    years = sorted(master['year'].dropna().unique().astype(int))
    selected_years = st.multiselect("Filter by year", years, default=years)
    filtered = master[master['year'].isin(selected_years)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue (filtered)", f"R${filtered['price'].sum()/1e6:.2f}M")
    col2.metric("Orders (filtered)",  f"{filtered['order_id'].nunique():,}")
    col3.metric("Avg Order Value",    f"R${filtered.groupby('order_id')['price'].sum().mean():.2f}")

    st.markdown("---")

    # ── Monthly trend ─────────────────────────────────────────────────────────
    st.markdown("#### Monthly Revenue")
    monthly_filtered = (
        filtered.groupby('month_label')['price'].sum()
        .reset_index().sort_values('month_label')
    )
    fig = px.area(monthly_filtered, x='month_label', y='price',
                  color_discrete_sequence=[ACCENT])
    fig.update_yaxes(tickprefix='R$', tickformat=',.0f')
    fig.update_layout(
        height=320, xaxis_title='', yaxis_title='Revenue',
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT)
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Orders by Day of Week")
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        daily = filtered.groupby('weekday')['order_id'].nunique().reindex(day_order).reset_index()
        daily.columns = ['day', 'orders']
        daily['color'] = daily['day'].apply(lambda d: WARM if d in ['Saturday','Sunday'] else ACCENT)
        fig = go.Figure(go.Bar(
            x=daily['day'], y=daily['orders'],
            marker_color=daily['color'], opacity=0.85
        ))
        fig.update_layout(
            height=300, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
            xaxis_title='', yaxis_title='Orders'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Orders by Hour of Day")
        hourly = filtered.groupby('hour')['order_id'].nunique().reset_index()
        hourly.columns = ['hour', 'orders']
        fig = px.bar(hourly, x='hour', y='orders', color_discrete_sequence=[ACCENT])
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
            xaxis_title='Hour', yaxis_title='Orders'
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Geographic ────────────────────────────────────────────────────────────
    st.markdown("#### Revenue by State")
    state_rev = (
        filtered.groupby('customer_state')['price'].sum()
        .reset_index().sort_values('price', ascending=True)
    )
    fig = px.bar(
        state_rev, x='price', y='customer_state',
        orientation='h', color_discrete_sequence=[ACCENT]
    )
    fig.update_xaxes(tickprefix='R$', tickformat=',.0f')
    fig.update_layout(
        height=500, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
        xaxis_title='Revenue', yaxis_title=''
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CUSTOMER ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_selected == "👥 Customer Analytics":
    st.title("👥 Customer Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Purchase Frequency Distribution")
        purchase_counts = (
            master.groupby('customer_unique_id')['order_id'].nunique()
        )
        freq = purchase_counts.clip(upper=6).value_counts().sort_index().reset_index()
        freq.columns = ['orders', 'customers']
        freq['orders'] = freq['orders'].astype(str)
        freq.loc[freq['orders'] == '6', 'orders'] = '6+'
        colors = ['#3b5b8c'] + [ACCENT] * (len(freq) - 1)
        fig = go.Figure(go.Bar(
            x=freq['orders'], y=freq['customers'],
            marker_color=colors, opacity=0.85,
            text=freq['customers'], textposition='outside'
        ))
        fig.update_layout(
            height=320, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
            xaxis_title='Number of orders', yaxis=dict(showgrid=False, showticklabels=False)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### RFM Segment Distribution")
        seg_counts = rfm['segment'].value_counts().reset_index()
        seg_counts.columns = ['segment', 'count']
        color_map = SEGMENT_COLORS
        fig = px.pie(
            seg_counts, values='count', names='segment',
            color='segment', color_discrete_map=color_map,
            hole=0.4
        )
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_LIGHT)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Delivery Time vs Review Score")
    del_review = master[master['delivery_days'].between(1, 60)].copy()
    del_review['delivery_bucket'] = pd.cut(
        del_review['delivery_days'],
        bins=[0, 5, 10, 15, 20, 30, 60],
        labels=['1–5d','6–10d','11–15d','16–20d','21–30d','30d+']
    )
    bucket_stats = del_review.groupby('delivery_bucket', observed=True).agg(
        avg_review=('review_score', 'mean'),
        orders=('order_id', 'nunique')
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=bucket_stats['delivery_bucket'], y=bucket_stats['orders'],
               name='Order volume', marker_color='#2d3a52', opacity=0.9),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=bucket_stats['delivery_bucket'], y=bucket_stats['avg_review'],
                   name='Avg review score', line=dict(color=WARM, width=2.5),
                   mode='lines+markers', marker=dict(size=8, symbol='diamond')),
        secondary_y=True
    )
    fig.update_yaxes(title_text="Orders", secondary_y=False)
    fig.update_yaxes(title_text="Avg Review Score", range=[1, 5], secondary_y=True)
    fig.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Business insight:** Review score drops sharply after 10 days — a delivery SLA of ≤10 days is a high-ROI operational target.")

    # ── RFM table ─────────────────────────────────────────────────────────────
    st.markdown("#### RFM Segment Summary")
    rfm_summary = rfm.groupby('segment').agg(
        customers=('customer_unique_id', 'count'),
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary', 'mean')
    ).round(1).reset_index().sort_values('customers', ascending=False)
    rfm_summary.columns = ['Segment','Customers','Avg Recency (days)','Avg Orders','Avg Spend (R$)']
    st.dataframe(rfm_summary, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRODUCT INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_selected == "📦 Product Insights":
    st.title("📦 Product Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 15 Categories by Revenue")
        cat_rev = (
            master.groupby('category')['price'].sum()
            .sort_values(ascending=True).tail(15).reset_index()
        )
        fig = px.bar(cat_rev, x='price', y='category', orientation='h',
                     color_discrete_sequence=[ACCENT])
        fig.update_xaxes(tickprefix='R$', tickformat=',.0f')
        fig.update_layout(
            height=480, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
            xaxis_title='Revenue', yaxis_title=''
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Payment Type Mix")
        pay_type = master.groupby('payment_type')['order_id'].nunique().reset_index()
        pay_type.columns = ['type', 'orders']
        fig = px.pie(pay_type, values='orders', names='type',
                     color_discrete_sequence=[ACCENT,GREEN,WARM,PURPLE],
                     hole=0.4)
        fig.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_LIGHT)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Installments vs Avg Order Value")
        inst_aov = (
            master[master['payment_installments'].between(1, 12)]
            .groupby('payment_installments')['price'].mean().reset_index()
        )
        inst_aov.columns = ['installments', 'avg_price']
        fig = px.bar(inst_aov, x='installments', y='avg_price',
                     color_discrete_sequence=[ACCENT])
        fig.update_yaxes(tickprefix='R$', tickformat=',.0f')
        fig.update_layout(
            height=200, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
            xaxis_title='Installments', yaxis_title='Avg Price'
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Category bubble chart ─────────────────────────────────────────────────
    st.markdown("#### Category Map — Volume vs Price vs Review Score")
    cat_stats = master.groupby('category').agg(
        revenue=('price', 'sum'),
        orders=('order_id', 'nunique'),
        avg_price=('price', 'mean'),
        avg_review=('review_score', 'mean')
    ).reset_index()
    cat_stats = cat_stats[cat_stats['orders'] >= 200]

    fig = px.scatter(
        cat_stats, x='orders', y='avg_price',
        size='revenue', color='avg_review',
        hover_name='category',
        color_continuous_scale='RdYlGn',
        range_color=[3.5, 5.0],
        size_max=50,
        labels={'orders':'Number of Orders','avg_price':'Avg Price (R$)','avg_review':'Avg Review'}
    )
    fig.update_layout(
        height=480, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **How to read:** Bubble size = total revenue. Color = review score (green=good, red=poor). High-volume + red bubbles are service risk areas.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_selected == "🤖 ML Predictions":
    st.title("🤖 ML Predictions — Repeat Purchase Model")
    st.markdown("Random Forest model predicting whether a customer will make a second purchase.")

    # ── Model metrics ─────────────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC",  f"{roc_auc_score(preds['actual'], preds['prob_repeat']):.4f}")
    col2.metric("F1 Score",  f"{f1_score(preds['actual'], preds['predicted']):.4f}")
    col3.metric("Accuracy",  f"{accuracy_score(preds['actual'], preds['predicted']):.4f}")
    col4.metric("Test Samples", f"{len(preds):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(preds['actual'], preds['predicted'])
        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Predicted: One-time','Predicted: Repeat'],
            y=['Actual: One-time','Actual: Repeat'],
            colorscale=[[0, CARD_BG], [1, ACCENT]], showscale=False
        )
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_LIGHT)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Score Distribution by Actual Class")
        df0 = preds[preds['actual']==0]['prob_repeat']
        df1 = preds[preds['actual']==1]['prob_repeat']
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df0, name='One-time buyer', opacity=0.65,
                                   marker_color=WARM, nbinsx=40))
        fig.add_trace(go.Histogram(x=df1, name='Repeat buyer', opacity=0.65,
                                   marker_color=ACCENT, nbinsx=40))
        fig.add_vline(x=0.5, line_dash='dash', line_color='gray')
        fig.update_layout(
            barmode='overlay', height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
            xaxis_title='Predicted probability of repeat purchase',
            legend=dict(x=0.6, y=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("#### Top 15 Features — What Drives Repeat Purchase?")
    top_feat = feat_imp.head(15).sort_values('importance', ascending=True)
    fig = px.bar(top_feat, x='importance', y='feature', orientation='h',
                 color_discrete_sequence=[ACCENT])
    fig.update_layout(
        height=450, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
        xaxis_title='Feature Importance', yaxis_title=''
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Business translation:** The features at the top are what you should focus on operationally to improve customer retention.")

    # ── Live predictor ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Live Customer Predictor")
    st.markdown("Enter a new customer's first order details to get a retention prediction.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        order_value    = c1.number_input("Order Value (R$)",    min_value=0.0,  value=150.0, step=10.0)
        freight_value  = c1.number_input("Freight Value (R$)",  min_value=0.0,  value=20.0,  step=5.0)
        review_score   = c2.slider("Review Score",              min_value=1,    max_value=5, value=4)
        delivery_days  = c2.number_input("Delivery Days",       min_value=1,    max_value=60, value=8)
        delivery_delta = c3.number_input("Delivery Delta (+ early / - late)", min_value=-30, max_value=30, value=2)
        installments   = c3.number_input("Payment Installments", min_value=1,   max_value=24, value=1)
        submitted = st.form_submit_button("Predict →", use_container_width=True)

    if submitted:
        # Build input row matching training feature order
        freight_ratio  = freight_value / (order_value + 1)
        was_early      = int(delivery_delta > 0)
        was_very_late  = int(delivery_delta < -7)
        is_high_value  = int(order_value >= master['price'].quantile(0.75))

        input_dict = {col: 0 for col in feature_cols}
        input_dict.update({
            'first_order_value':    order_value,
            'first_order_total':    order_value,
            'first_freight':        freight_value,
            'first_review_score':   review_score,
            'first_delivery_days':  delivery_days,
            'first_delivery_delta': delivery_delta,
            'first_payment_inst':   installments,
            'freight_ratio':        freight_ratio,
            'was_early':            was_early,
            'was_very_late':        was_very_late,
            'is_high_value':        is_high_value,
        })
        input_df = pd.DataFrame([input_dict])[feature_cols]
        prob = rf_model.predict_proba(input_df)[0][1]

        col1, col2 = st.columns([1, 2])
        with col1:
            if prob >= 0.6:
                st.success(f"✅ **Likely to return**\n\nProbability: **{prob:.1%}**")
            elif prob >= 0.35:
                st.warning(f"⚠️ **Uncertain**\n\nProbability: **{prob:.1%}**")
            else:
                st.error(f"❌ **At risk of churning**\n\nProbability: **{prob:.1%}**")
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%', 'font': {'size': 36, 'color': TEXT_LIGHT}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': TEXT_MUTED},
                    'bar': {'color': ACCENT},
                    'bgcolor': CARD_BG,
                    'steps': [
                        {'range': [0, 35],  'color': '#3a1f1f'},
                        {'range': [35, 60], 'color': '#3a331a'},
                        {'range': [60, 100],'color': '#1a3a2a'}
                    ],
                    'threshold': {'line': {'color': TEXT_LIGHT, 'width': 2}, 'value': 50}
                },
                title={'text': 'Repeat Purchase Probability', 'font': {'color': TEXT_LIGHT}}
            ))
            fig.update_layout(
                height=250, margin=dict(l=20, r=20, t=30, b=0),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TEXT_LIGHT)
            )
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REVENUE FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_selected == "🔮 Revenue Forecast":
    st.title("🔮 Revenue Forecast")
    st.markdown("6-month revenue projection using Facebook Prophet time series model.")

    future_only = forecast[forecast['is_forecast'] == True]
    historical  = forecast[forecast['is_forecast'] == False]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Projected 6-Month Revenue",
                f"R${future_only['yhat'].sum()/1e6:.2f}M")
    col2.metric("Peak Forecast Month",
                str(future_only.loc[future_only['yhat'].idxmax(), 'ds'].date()))
    col3.metric("Avg Monthly Forecast",
                f"R${future_only['yhat'].mean():,.0f}")

    st.markdown("---")

    # ── Main forecast chart ───────────────────────────────────────────────────
    st.markdown("#### Revenue Forecast — Historical + 6-Month Projection")
    fig = go.Figure()

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=monthly['ds'], y=monthly['y'],
        mode='lines+markers', name='Actual Revenue',
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=5)
    ))

    # Future forecast
    fig.add_trace(go.Scatter(
        x=future_only['ds'], y=future_only['yhat'],
        mode='lines+markers', name='Forecast',
        line=dict(color=WARM, width=2.5),
        marker=dict(size=7, symbol='diamond')
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([future_only['ds'], future_only['ds'][::-1]]),
        y=pd.concat([future_only['yhat_upper'], future_only['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(232,89,60,0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))

    # Divider line
    split = monthly['ds'].max()
    fig.add_vline(x=str(split), line_dash='dot', line_color='gray', opacity=0.6)
    fig.add_annotation(x=str(split), y=1, yref='paper', text='  forecast →',
                       showarrow=False, font=dict(color='gray', size=11))

    fig.update_yaxes(tickprefix='R$', tickformat=',.0f')
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT),
        legend=dict(x=0, y=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor=GRID)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast table ────────────────────────────────────────────────────────
    st.markdown("#### Month-by-Month Forecast")
    forecast_table = future_only[['ds','yhat','yhat_lower','yhat_upper']].copy()
    forecast_table.columns = ['Month','Forecast (R$)','Lower Bound (R$)','Upper Bound (R$)']
    forecast_table['Month'] = forecast_table['Month'].dt.strftime('%B %Y')
    for col in ['Forecast (R$)','Lower Bound (R$)','Upper Bound (R$)']:
        forecast_table[col] = forecast_table[col].apply(lambda x: f"R${x:,.0f}")
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    st.info("💡 **How to use:** Use the lower bound for conservative budget planning. Use the forecast for target-setting. If actuals fall below the lower bound, investigate anomalies immediately.")