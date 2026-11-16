import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# --- 0. ตั้งค่าหน้าเว็บ (Page Configuration) ---
st.set_page_config(
    page_title="DCA Simulation Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 DCA Rolling Window Simulation Dashboard")
st.markdown("จำลองผลลัพธ์การลงทุนแบบถัวเฉลี่ยต้นทุน (**Dollar-Cost Averaging**) จากข้อมูลจริงในอดีตทุกช่วงเวลา")

# --- 1. แถบเมนูด้านซ้าย (Sidebar Controls) สำหรับเปลี่ยนโจทย์ ---
st.sidebar.header("⚙️ ตั้งค่าโจทย์การลงทุน")

asset_options = {
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Gold ทองคำ (GC=F)": "GC=F",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "กำหนดเอง (Custom Ticker)": "CUSTOM"
}

selected_label = st.sidebar.selectbox("เลือกสินทรัพย์ (Asset)", list(asset_options.keys()))

if asset_options[selected_label] == "CUSTOM":
    ticker = st.sidebar.text_input("กรอก Yahoo Finance Ticker (เช่น NVDA, TSLA)", value="NVDA").upper().strip()
    asset_name = ticker
else:
    ticker = asset_options[selected_label]
    asset_name = selected_label.split(" (")[0]

investment_per_month = st.sidebar.number_input(
    "เงินลงทุนต่อเดือน - PMT ($)",
    min_value=100,
    max_value=1000000,
    value=10000,
    step=1000
)

years_duration = st.sidebar.slider(
    "ระยะเวลาลงทุน (ปี)",
    min_value=1,
    max_value=30,
    value=15
)

# --- 2. ฟังก์ชันโหลดข้อมูลและคำนวณ Annual Return ---
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
    # สมการ Annuity Due (ลงทุนต้นงวด): FV = PMT * [((1+r)^n - 1) / r] * (1+r)
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


# --- 3. โหลดข้อมูลและตรวจสอบความพร้อม ---
with st.spinner(f"กำลังดึงข้อมูลราคา {asset_name} ({ticker})..."):
    data = load_price_data(ticker)

months_duration = years_duration * 12

if data.empty:
    st.error(f"ไม่พบข้อมูลสำหรับ Ticker: `{ticker}` กรุณาตรวจสอบชื่อย่อสินทรัพย์อีกครั้ง")
    st.stop()

if len(data) <= months_duration:
    st.warning(
        f"⚠️ ข้อมูลของ **{asset_name}** มีเพียง **{len(data)/12:.1f} ปี** "
        f"(ตั้งแต่ {data.index[0].strftime('%Y-%m')}) ซึ่งไม่พอสำหรับการจำลอง **{years_duration} ปี** "
        f"กรุณาลดจำนวนปีที่แถบด้านซ้าย"
    )
    st.stop()

# --- 4. ลูปคำนวณ Rolling Window ---
results = []
start_date_range = data.index[:-months_duration]

for start_date in start_date_range:
    start_idx = data.index.get_loc(start_date)

    if start_idx + months_duration < len(data):
        window_prices = data.iloc[start_idx : start_idx + months_duration]

        # คำนวณ DCA
        units_bought = investment_per_month / window_prices
        total_units = units_bought.sum()
        final_price = data.iloc[start_idx + months_duration]
        final_value = total_units * final_price

        # คำนวณ Annual Return (%) ย้อนกลับ
        annual_return_pct = get_annual_return(months_duration, investment_per_month, final_value)

        results.append({
            'Start_Date': start_date,
            'End_Date': window_prices.index[-1],
            'Final_Value': final_value,
            'Annual_Return_Pct': annual_return_pct
        })

df_results = pd.DataFrame(results)

# --- 5 & 6. แสดงผล Summary Cards, กราฟ และตารางสถิติ ---
if not df_results.empty:
    total_principal = investment_per_month * months_duration
    mean_fv = df_results['Final_Value'].mean()
    median_fv = df_results['Final_Value'].median()
    mean_return = df_results['Annual_Return_Pct'].mean()
    win_rate = (df_results['Final_Value'] >= total_principal).mean() * 100
    profit_pct = ((mean_fv - total_principal) / total_principal) * 100

    st.caption(
        f"📅 ช่วงข้อมูลที่ใช้: **{data.index[0].strftime('%Y-%m')}** ถึง **{data.index[-1].strftime('%Y-%m')}** "
        f"| จำนวนการจำลองทั้งหมด: **{len(df_results):,} รอบ (Paths)**"
    )

    # กล่องสรุปตัวเลขสำคัญ 4 กล่องด้านบน
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💰 เงินต้นรวม (Principal)", f"${total_principal:,.0f}")
    m2.metric("🏆 เงินก้อนสุดท้ายเฉลี่ย (Mean FV)", f"${mean_fv:,.0f}", f"{profit_pct:+.1f}% จากเงินต้น")
    m3.metric("📊 ผลตอบแทนทบต้นเฉลี่ย (Mean Return)", f"{mean_return:.2f}% / ปี")
    m4.metric("✅ โอกาสไม่ขาดทุน (Win Rate)", f"{win_rate:.1f}%", f"{len(df_results)} Simulations")

    st.divider()

    # แบ่งหน้าจอเป็น 2 คอลัมน์สำหรับแสดงกราฟคู่กัน
    col1, col2 = st.columns(2)

    # กราฟที่ 1: Distribution of Annual Return (%)
    with col1:
        st.subheader("1. การกระจายตัวของผลตอบแทนต่อปี (%)")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.hist(df_results['Annual_Return_Pct'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        ax1.axvline(mean_return, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_return:.2f}%')
        ax1.set_title(
            f'Distribution of Annualized Returns (DCA {years_duration} Years in {asset_name})\n'
            f'Data: {data.index[0].strftime("%Y-%m")} to {data.index[-1].strftime("%Y-%m")}'
        )
        ax1.set_xlabel('Annual Effective Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        with st.expander("📋 ดูตารางสถิติ Annual Return (%)"):
            stats_ret = df_results['Annual_Return_Pct'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            st.dataframe(stats_ret.apply(lambda x: f"{x:.5f}%"), use_container_width=True)

    # กราฟที่ 2: Distribution of Final Value ($)
    with col2:
        st.subheader("2. การกระจายตัวของเงินก้อนสุดท้าย ($)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(df_results['Final_Value'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(total_principal, color='black', linestyle='solid', linewidth=1.5, label=f'Principal: ${total_principal:,.0f}')
        ax2.axvline(mean_fv, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean FV: ${mean_fv:,.0f}')
        ax2.xaxis.set_major_formatter(
            lambda x, pos: f'${x/1e6:.1f}M' if mean_fv >= 1e6 else f'${x/1e3:,.0f}K'
        )
        ax2.set_title(
            f'Distribution of Final Value (DCA ${investment_per_month:,}/mo, {years_duration} Years in {asset_name})\n'
            f'Data: {data.index[0].strftime("%Y-%m")} to {data.index[-1].strftime("%Y-%m")}'
        )
        ax2.set_xlabel('Final Value ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        with st.expander("📋 ดูตารางสถิติ Final Value ($)"):
            stats_fv = df_results['Final_Value'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            st.dataframe(stats_fv.apply(lambda x: f"${x:,.2f}"), use_container_width=True)
