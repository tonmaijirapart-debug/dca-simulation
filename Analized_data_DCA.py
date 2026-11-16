import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# --- 0. Page Configuration & Styling ---
st.set_page_config(
    page_title="DCA Rolling Window Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 16px 20px;
        border-radius: 6px;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.85rem;
        color: #495057;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.6rem;
        font-weight: 600;
        color: #212529;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Dollar-Cost Averaging (DCA) Historical Simulation")
st.caption("Rolling-window backtest evaluating annualized effective returns and terminal portfolio values across historical market cycles.")

# --- 1. Sidebar Controls ---
st.sidebar.subheader("Simulation Parameters")

asset_options = {
    "S&P 500 Index (^GSPC)": "^GSPC",
    "Nasdaq 100 ETF (QQQ)": "QQQ",
    "Gold Futures (GC=F)": "GC=F",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Custom Ticker": "CUSTOM"
}

selected_label = st.sidebar.selectbox("Asset", list(asset_options.keys()))

if asset_options[selected_label] == "CUSTOM":
    ticker = st.sidebar.text_input("Ticker Symbol (Yahoo Finance)", value="NVDA").upper().strip()
    asset_name = ticker
else:
    ticker = asset_options[selected_label]
    asset_name = selected_label.split(" (")[0]

investment_per_month = st.sidebar.number_input(
    "Monthly Contribution ($)",
    min_value=100,
    max_value=1000000,
    value=1000,
    step=500
)

years_duration = st.sidebar.slider(
    "Investment Horizon (Years)",
    min_value=1,
    max_value=30,
    value=15
)


# --- 2. Data Fetching & Financial Functions ---
@st.cache_data(show_spinner=False)
def load_price_data(ticker_symbol):
    df_raw = yf.download(ticker_symbol, period='max', interval='1mo', progress=False)
    if df_raw.empty:
        return pd.Series(dtype=float)

    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    if 'Adj Close' in df_raw.columns:
        return df_raw['Adj Close'].dropna()
    elif 'Close' in df_raw.columns:
        return df_raw['Close'].dropna()
    return pd.Series(dtype=float)


def get_annual_return(n_months, pmt, fv):
    # Annuity Due equation: FV = PMT * [((1 + r)^n - 1) / r] * (1 + r)
    def func(r):
        if r <= -1:
            return -fv
        return pmt * (((1 + r)**n_months - 1) / r) * (1 + r) - fv

    try:
        monthly_r = optimize.newton(func, 0.005)
        annual_r = ((1 + monthly_r)**12) - 1
        return annual_r * 100
    except Exception:
        return np.nan


# --- 3. Load Data & Validate ---
with st.spinner(f"Loading historical data for {asset_name} ({ticker})..."):
    data = load_price_data(ticker)

months_duration = years_duration * 12

if data.empty:
    st.error(f"No historical price data found for ticker '{ticker}'. Please verify the symbol.")
    st.stop()

if len(data) <= months_duration:
    st.warning(
        f"Insufficient historical data for {asset_name}. Available history is {len(data)/12:.1f} years "
        f"(since {data.index[0].strftime('%Y-%m')}), which is shorter than the selected {years_duration}-year horizon."
    )
    st.stop()


# --- 4. Rolling Window Simulation ---
results = []
start_date_range = data.index[:-months_duration]

for start_date in start_date_range:
    start_idx = data.index.get_loc(start_date)

    if start_idx + months_duration < len(data):
        window_prices = data.iloc[start_idx : start_idx + months_duration]

        # DCA accumulation
        units_bought = investment_per_month / window_prices
        total_units = units_bought.sum()
        final_price = data.iloc[start_idx + months_duration]
        final_value = total_units * final_price

        # Solve for Effective Annual Return (%)
        annual_return_pct = get_annual_return(months_duration, investment_per_month, final_value)

        results.append({
            'Start_Date': start_date,
            'End_Date': window_prices.index[-1],
            'Final_Value': final_value,
            'Annual_Return_Pct': annual_return_pct
        })

df_results = pd.DataFrame(results)


# --- 5. Dashboard Output: Metrics, Charts, and Summary Table ---
if not df_results.empty:
    total_principal = investment_per_month * months_duration
    mean_fv = df_results['Final_Value'].mean()
    median_fv = df_results['Final_Value'].median()
    mean_return = df_results['Annual_Return_Pct'].mean()
    win_rate = (df_results['Final_Value'] >= total_principal).mean() * 100
    multiple = mean_fv / total_principal

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**Dataset Range:** {data.index[0].strftime('%Y-%m')} to {data.index[-1].strftime('%Y-%m')}  \n"
        f"**Rolling Windows:** {len(df_results):,} paths"
    )

    # Top KPI Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Principal Invested", f"${total_principal:,.0f}")
    m2.metric("Mean Terminal Value", f"${mean_fv:,.0f}", f"{multiple:.2f}x Principal")
    m3.metric("Mean Annualized Return", f"{mean_return:.2f}%")
    m4.metric("Positive Return Frequency", f"{win_rate:.1f}%", f"{len(df_results):,} Rolling Periods")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    col1, col2 = st.columns(2)

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8

    # Chart 1: Annualized Return Distribution
    with col1:
        st.markdown("##### Distribution of Annualized Returns (%)")
        fig1, ax1 = plt.subplots(figsize=(8, 4.8), dpi=120)
        ax1.hist(
            df_results['Annual_Return_Pct'],
            bins=45,
            color='#2e6b5e',
            edgecolor='white',
            linewidth=0.6,
            alpha=0.85
        )
        ax1.axvline(mean_return, color='#b22222', linestyle='--', linewidth=1.5, label=f'Mean: {mean_return:.2f}%')
        ax1.set_title(
            f'{asset_name} | {years_duration}-Year Rolling DCA ({data.index[0].strftime("%Y-%m")} – {data.index[-1].strftime("%Y-%m")})',
            fontsize=10,
            color='#495057',
            pad=12
        )
        ax1.set_xlabel('Annualized Effective Return (%)', fontsize=9, labelpad=8)
        ax1.set_ylabel('Frequency (Rolling Periods)', fontsize=9, labelpad=8)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(axis='y', linestyle=':', alpha=0.5)
        ax1.legend(frameon=True, facecolor='white', edgecolor='#e9ecef', fontsize=9)
        fig1.tight_layout()
        st.pyplot(fig1)

    # Chart 2: Final Portfolio Value Distribution
    with col2:
        st.markdown("##### Distribution of Terminal Portfolio Value ($)")
        fig2, ax2 = plt.subplots(figsize=(8, 4.8), dpi=120)
        ax2.hist(
            df_results['Final_Value'],
            bins=45,
            color='#2b5c8f',
            edgecolor='white',
            linewidth=0.6,
            alpha=0.85
        )
        ax2.axvline(total_principal, color='#212529', linestyle='-', linewidth=1.5, label=f'Principal: ${total_principal:,.0f}')
        ax2.axvline(mean_fv, color='#b22222', linestyle='--', linewidth=1.5, label=f'Mean FV: ${mean_fv:,.0f}')
        ax2.xaxis.set_major_formatter(
            lambda x, pos: f'${x/1e6:.1f}M' if mean_fv >= 1e6 else f'${x/1e3:,.0f}K'
        )
        ax2.set_title(
            f'${investment_per_month:,}/Month Contribution | {years_duration}-Year Horizon',
            fontsize=10,
            color='#495057',
            pad=12
        )
        ax2.set_xlabel('Terminal Portfolio Value ($)', fontsize=9, labelpad=8)
        ax2.set_ylabel('Frequency (Rolling Periods)', fontsize=9, labelpad=8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(axis='y', linestyle=':', alpha=0.5)
        ax2.legend(frameon=True, facecolor='white', edgecolor='#e9ecef', fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2)

    # Summary Statistics Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Summary Statistics & Percentiles")

    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    stats_ret = df_results['Annual_Return_Pct'].describe(percentiles=percentiles)
    stats_fv = df_results['Final_Value'].describe(percentiles=percentiles)

    summary_df = pd.DataFrame({
        'Metric': [
            'Count (Simulations)',
            'Mean',
            'Standard Deviation',
            'Minimum',
            '10th Percentile',
            '25th Percentile',
            '50th Percentile (Median)',
            '75th Percentile',
            '90th Percentile',
            'Maximum'
        ],
        'Annualized Return (%)': [
            f"{stats_ret['count']:,.0f}",
            f"{stats_ret['mean']:.4f}%",
            f"{stats_ret['std']:.4f}%",
            f"{stats_ret['min']:.4f}%",
            f"{stats_ret['10%']:.4f}%",
            f"{stats_ret['25%']:.4f}%",
            f"{stats_ret['50%']:.4f}%",
            f"{stats_ret['75%']:.4f}%",
            f"{stats_ret['90%']:.4f}%",
            f"{stats_ret['max']:.4f}%"
        ],
        'Terminal Portfolio Value ($)': [
            f"{stats_fv['count']:,.0f}",
            f"${stats_fv['mean']:,.2f}",
            f"${stats_fv['std']:,.2f}",
            f"${stats_fv['min']:,.2f}",
            f"${stats_fv['10%']:,.2f}",
            f"${stats_fv['25%']:,.2f}",
            f"${stats_fv['50%']:,.2f}",
            f"${stats_fv['75%']:,.2f}",
            f"${stats_fv['90%']:,.2f}",
            f"${stats_fv['max']:,.2f}"
        ]
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)
