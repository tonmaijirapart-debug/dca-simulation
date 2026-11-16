import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# --- 0. Page Configuration ---
st.set_page_config(
    page_title="DCA Rolling Window Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    mean_return = df_results['Annual_Return_Pct'].mean()
    win_rate = (df_results['Final_Value'] >= total_principal).mean() * 100
    multiple = mean_fv / total_principal

    st.sidebar.divider()
    st.sidebar.caption(
        f"**Dataset Range:** {data.index[0].strftime('%Y-%m')} to {data.index[-1].strftime('%Y-%m')}  \n"
        f"**Rolling Windows:** {len(df_results):,} paths"
    )

    # Top KPI Row (native Streamlit containers adapt automatically to Dark & Light mode)
    m1, m2, m3, m4 = st.columns(4)
    with m1.container(border=True):
        st.metric("Total Principal Invested", f"${total_principal:,.0f}")
    with m2.container(border=True):
        st.metric("Mean Terminal Value", f"${mean_fv:,.0f}", f"{multiple:.2f}x Principal")
    with m3.container(border=True):
        st.metric("Mean Annualized Return", f"{mean_return:.2f}%")
    with m4.container(border=True):
        st.metric("Positive Return Frequency", f"{win_rate:.1f}%", f"{len(df_results):,} Rolling Periods")

    st.divider()

    # Charts Row (Classic raw Matplotlib style)
    col1, col2 = st.columns(2)

    # Chart 1: Annualized Return Distribution
    with col1:
        st.subheader("Distribution of Annualized Returns (%)")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.hist(df_results['Annual_Return_Pct'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        ax1.axvline(mean_return, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_return:.2f}%')
        ax1.set_title(
            f'Distribution of Annualized Returns (DCA {years_duration} Years in {asset_name})\n'
            f'Start Years: {data.index[0].strftime("%Y-%m")} to {data.index[-1].strftime("%Y-%m")}'
        )
        ax1.set_xlabel('Annual Effective Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    # Chart 2: Final Portfolio Value Distribution
    with col2:
        st.subheader("Distribution of Final Value ($)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(df_results['Final_Value'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(total_principal, color='black', linestyle='solid', linewidth=1, label=f'Principal: ${total_principal:,.0f}')
        ax2.axvline(mean_fv, color='red', linestyle='dashed', linewidth=1, label=f'Mean FV: ${mean_fv:,.2f}')
        ax2.xaxis.set_major_formatter(
            lambda x, pos: f'${x/1e6:.1f}M' if mean_fv >= 1e6 else f'${x/1e3:,.0f}K'
        )
        ax2.set_title(
            f'Distribution of Final Value (DCA ${investment_per_month:,}/mo, {years_duration} Years in {asset_name})\n'
            f'Start Years: {data.index[0].strftime("%Y-%m")} to {data.index[-1].strftime("%Y-%m")}'
        )
        ax2.set_xlabel('Final Value ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    # Summary Statistics Table
    st.divider()
    st.subheader("Summary Statistics & Percentiles")

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
            f"{stats_ret['mean']:.5f}%",
            f"{stats_ret['std']:.5f}%",
            f"{stats_ret['min']:.5f}%",
            f"{stats_ret['10%']:.5f}%",
            f"{stats_ret['25%']:.5f}%",
            f"{stats_ret['50%']:.5f}%",
            f"{stats_ret['75%']:.5f}%",
            f"{stats_ret['90%']:.5f}%",
            f"{stats_ret['max']:.5f}%"
        ],
        'Final Value ($)': [
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
