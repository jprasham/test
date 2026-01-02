import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Portfolio Intelligence Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.4);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid rgba(71, 85, 105, 0.5);
    }
    .summary-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1));
        border: 2px solid rgba(59, 130, 246, 0.3);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.title("🎯 Portfolio Intelligence Dashboard")
st.markdown(f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    default_tickers = ['SPY', 'GLD', 'SMH', 'COPX', 'TLT', 'FXI', 'UUP', 'USO']
    tickers = st.multiselect(
        "Select ETFs",
        options=['SPY', 'GLD', 'SMH', 'COPX', 'TLT', 'FXI', 'UUP', 'USO', 
                'QQQ', 'IWM', 'EEM', 'VNQ', 'DBC', 'GDX', 'SLV'],
        default=default_tickers,
        help="SPY is required as the market benchmark"
    )
    
    window = st.slider("Rolling Window (days)", 10, 60, 30, 
                      help="Window for correlation calculation")
    
    period = st.selectbox("Analysis Period", ['6mo', '1y', '2y'], index=1,
                         help="Longer period provides more reliable trend signals")
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("**ETF Guide:**")
    st.caption("SPY = S&P 500")
    st.caption("GLD = Gold")
    st.caption("SMH = Semiconductors")
    st.caption("COPX = Copper")
    st.caption("TLT = 20Y+ Treasuries")
    st.caption("FXI = China Large-Cap")
    st.caption("UUP = US Dollar Index")
    st.caption("USO = Oil")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_data(tickers, period):
    """Fetch historical price data from Yahoo Finance"""
    try:
        data = yf.download(tickers, period=period, progress=False)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient"""
    if len(x) < 2 or len(y) < 2:
        return 0
    return np.corrcoef(x, y)[0, 1]

def calculate_beta(asset_returns, market_returns):
    """Calculate beta (market sensitivity)"""
    if len(asset_returns) < 2 or len(market_returns) < 2:
        return 0
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    variance = np.var(market_returns)
    return covariance / variance if variance > 0 else 0

def calculate_dual_beta(asset_returns, market_returns):
    """
    Calculate up beta (on up days) and down beta (on down days)
    Asymmetry > 0.2 indicates convex payoff (good hedge quality)
    """
    up_mask = market_returns > 0
    down_mask = market_returns < 0
    
    beta_up = calculate_beta(
        asset_returns[up_mask], 
        market_returns[up_mask]
    ) if up_mask.sum() > 1 else 0
    
    beta_down = calculate_beta(
        asset_returns[down_mask], 
        market_returns[down_mask]
    ) if down_mask.sum() > 1 else 0
    
    return beta_up, beta_down

def calculate_trend_signal(prices):
    """
    Calculate trend based on 50 and 200 day moving averages
    
    Returns:
        - "Positive ↑": Price above BOTH 50D and 200D MA
        - "Negative ↓": Price below BOTH 50D and 200D MA
        - "No Signal": Mixed condition (between MAs)
    """
    if len(prices) < 200:
        return "No Signal", 0
    
    ma_50 = prices.rolling(window=50).mean()
    ma_200 = prices.rolling(window=200).mean()
    
    current_price = prices.iloc[-1]
    current_ma_50 = ma_50.iloc[-1]
    current_ma_200 = ma_200.iloc[-1]
    
    if current_price > current_ma_50 and current_price > current_ma_200:
        return "Positive ↑", 1
    elif current_price < current_ma_50 and current_price < current_ma_200:
        return "Negative ↓", -1
    else:
        return "No Signal", 0

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

if len(tickers) < 2:
    st.warning("⚠️ Please select at least 2 tickers (including SPY)")
elif 'SPY' not in tickers:
    st.warning("⚠️ SPY is required as the market benchmark")
else:
    with st.spinner("📊 Fetching data and running analysis..."):
        # Fetch data
        data = fetch_data(tickers, period)
        
        if data is not None and not data.empty:
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Rolling correlation with SPY
            rolling_corr = {}
            for ticker in tickers:
                if ticker != 'SPY':
                    rolling_corr[ticker] = returns[ticker].rolling(window=window).corr(returns['SPY'])
            
            rolling_corr_df = pd.DataFrame(rolling_corr)
            recent_returns = returns.tail(window)
            
            # ================================================================
            # CALCULATE METRICS FOR EACH ASSET
            # ================================================================
            metrics = []
            for ticker in tickers:
                if ticker == 'SPY':
                    continue
                
                # Current correlation with SPY
                corr = calculate_correlation(
                    recent_returns[ticker].values, 
                    recent_returns['SPY'].values
                )
                
                # Up/Down Beta
                beta_up, beta_down = calculate_dual_beta(
                    recent_returns[ticker].values,
                    recent_returns['SPY'].values
                )
                
                asymmetry = beta_up - beta_down
                
                # Volatility (annualized)
                vol = recent_returns[ticker].std() * np.sqrt(252) * 100
                
                # 30-day cumulative return
                cum_return = (1 + recent_returns[ticker]).prod() - 1
                
                # Trend signal using 50/200 day MA
                trend_signal, trend_value = calculate_trend_signal(data[ticker])
                
                # Correlation trend (falling = improving diversification)
                if len(rolling_corr_df) > 60:
                    recent_corr = rolling_corr_df[ticker].iloc[-1]
                    earlier_corr = rolling_corr_df[ticker].iloc[-60]
                    corr_change = recent_corr - earlier_corr
                    corr_trend = "↓" if corr_change < -0.05 else "↑" if corr_change > 0.05 else "→"
                else:
                    corr_trend = "→"
                    corr_change = 0
                
                metrics.append({
                    'Ticker': ticker,
                    'Correlation': corr,
                    'Corr_Change': corr_change,
                    'Corr_Trend': corr_trend,
                    'Volatility': vol,
                    'Up Beta': beta_up,
                    'Down Beta': beta_down,
                    'Asymmetry': asymmetry,
                    '30D Return': cum_return * 100,
                    'Trend': trend_signal,
                    'Trend_Value': trend_value
                })
            
            metrics_df = pd.DataFrame(metrics)
            
            # Create correlation matrix for all assets
            corr_matrix = recent_returns[tickers].corr()
            
            # ================================================================
            # GENERATE AI-READY EXECUTIVE SUMMARY
            # ================================================================
            
            # Calculate summary statistics
            avg_corr = metrics_df['Correlation'].mean()
            
            # Assets with positive trend AND low correlation (ideal for allocation)
            trend_diversifiers = metrics_df[
                (metrics_df['Trend_Value'] == 1) & 
                (metrics_df['Correlation'].abs() < 0.5)
            ]['Ticker'].tolist()
            
            # Assets to avoid: high correlation (>0.7) with no hedge benefit
            avoid_assets = metrics_df[
                (metrics_df['Correlation'] > 0.7)
            ]['Ticker'].tolist()
            
            # Good hedges: negative correlation + positive asymmetry
            good_hedges = metrics_df[
                (metrics_df['Correlation'] < -0.3) & 
                (metrics_df['Asymmetry'] > 0.15)
            ].sort_values('Asymmetry', ascending=False)['Ticker'].tolist()
            
            # Assets with positive trend but high correlation
            high_corr_trends = metrics_df[
                (metrics_df['Trend_Value'] == 1) & 
                (metrics_df['Correlation'] > 0.7)
            ]['Ticker'].tolist()
            
            # Assets with negative trend
            negative_trends = metrics_df[
                (metrics_df['Trend_Value'] == -1)
            ]['Ticker'].tolist()
            
            # ================================================================
            # DISPLAY EXECUTIVE SUMMARY
            # ================================================================
            
            st.markdown("""
            <div class="summary-box">
                <h3 style="margin-bottom: 1rem; color: #60a5fa;">📊 Executive Summary</h3>
            </div>
            """, unsafe_allow_html=True)
            
            summary_text = f"""
**Best Opportunities (Trend + Diversification):** {f"**{', '.join(trend_diversifiers)}** show positive technical momentum (above 50D/200D MA) with correlation below 0.5, offering optimal risk-adjusted allocation in a top-down framework." if trend_diversifiers else "No assets currently meet both trend and diversification criteria. Consider hedges or wait for technical setups."}

**Recommended Hedges:** {f"**{', '.join(good_hedges[:3])}** exhibit negative correlation (<-0.3) and convex payoff profiles (asymmetry >0.15), providing effective downside protection. Allocate 10-20% for tail risk hedging." if good_hedges else "Limited hedge candidates available. Consider increasing cash or defensive positioning."}

**Assets to Avoid/Reduce:** {f"**{', '.join(avoid_assets)}** show correlation >0.7 with SPY, offering minimal diversification benefit and concentrated equity beta exposure." if avoid_assets else "No highly correlated assets identified."}{f" Additionally, **{', '.join(high_corr_trends)}** have positive trends but high correlation—suitable only for tactical overweight with tight risk management." if high_corr_trends else ""}{f" **{', '.join(negative_trends)}** in negative trends (below 50D/200D MA)—avoid until technical improvement." if negative_trends else ""}

**Market Regime:** {"Risk-on environment with elevated correlations (avg {avg_corr:.2f}). Focus on hedges and await better diversification entry points." if avg_corr > 0.5 else f"Balanced regime (avg correlation {avg_corr:.2f}) with {len(trend_diversifiers)} assets offering momentum and diversification. Favorable for global asset allocation." if avg_corr > 0.2 else f"Defensive regime (avg correlation {avg_corr:.2f}) with strong hedge candidates. Prioritize capital preservation and convex positions."}
            """
            
            st.markdown(summary_text)
            
            # ================================================================
            # LAYOUT: THREE COLUMNS
            # ================================================================
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            # ================================================================
            # COLUMN 1: CORRELATION MATRIX HEATMAP
            # ================================================================
            
            with col1:
                st.subheader("Correlation Matrix")
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale=[
                        [0, '#10b981'],      # Green for negative correlation
                        [0.5, '#1e293b'],    # Dark for zero
                        [1, '#ef4444']       # Red for positive correlation
                    ],
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10, "color": "white"},
                    colorbar=dict(
                        title="Correlation",
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
                    ),
                    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
                ))
                
                fig_heatmap.update_layout(
                    template="plotly_dark",
                    height=400,
                    xaxis_title="",
                    yaxis_title="",
                    font=dict(size=10)
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                st.markdown("""
                **Color Guide:** 🟢 Green = Hedge | 🔴 Red = Correlated | ⚫ Dark = Diversifier
                """)
            
            # ================================================================
            # COLUMN 2: ROLLING CORRELATION TRENDS
            # ================================================================
            
            with col2:
                st.subheader(f"Rolling {window}D Correlation vs SPY")
                
                fig_corr = go.Figure()
                
                colors = ['#fbbf24', '#3b82f6', '#ef4444', '#10b981', 
                         '#a78bfa', '#f472b6', '#fb923c', '#22d3ee']
                
                for idx, ticker in enumerate(rolling_corr_df.columns):
                    fig_corr.add_trace(go.Scatter(
                        x=rolling_corr_df.index,
                        y=rolling_corr_df[ticker],
                        mode='lines',
                        name=ticker,
                        line=dict(width=2.5, color=colors[idx % len(colors)])
                    ))
                
                fig_corr.add_hline(y=0, line_dash="dash", line_color="gray", 
                                  opacity=0.5, line_width=1)
                fig_corr.add_hline(y=0.8, line_dash="dot", line_color="red", 
                                  opacity=0.3, annotation_text="High Corr (0.8)", 
                                  annotation_position="right")
                fig_corr.add_hline(y=-0.3, line_dash="dot", line_color="green", 
                                  opacity=0.3, annotation_text="Hedge Zone (-0.3)", 
                                  annotation_position="right")
                
                fig_corr.update_layout(
                    template="plotly_dark",
                    height=400,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", 
                        y=-0.3, 
                        xanchor="center", 
                        x=0.5
                    ),
                    xaxis_title="",
                    yaxis_title="Correlation",
                    yaxis=dict(range=[-1, 1])
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # ================================================================
            # COLUMN 3: MARKET REGIME INDICATORS
            # ================================================================
            
            with col3:
                st.subheader("Market Regime")
                
                high_corr_count = (metrics_df['Correlation'].abs() > 0.7).sum()
                hedge_count = (metrics_df['Correlation'] < -0.3).sum()
                positive_trend_count = (metrics_df['Trend_Value'] == 1).sum()
                
                st.metric("Avg Correlation", f"{avg_corr:.2f}")
                st.metric("High Corr Assets", f"{high_corr_count}/{len(metrics_df)}")
                st.metric("Hedge Candidates", f"{hedge_count}")
                st.metric("Positive Trends", f"{positive_trend_count}/{len(metrics_df)}")
                
                # Regime indicator
                if avg_corr > 0.5:
                    st.error("⚠️ **Risk-On**\nHigh correlation regime")
                elif avg_corr < 0.2:
                    st.success("✅ **Diversified**\nLow correlation regime")
                else:
                    st.info("ℹ️ **Mixed**\nModerate correlation")
            
            # ================================================================
            # BETA ASYMMETRY ANALYSIS
            # ================================================================
            
            st.subheader("Beta Asymmetry Analysis")
            
            col_beta1, col_beta2 = st.columns([3, 1])
            
            with col_beta1:
                fig_beta = go.Figure()
                
                fig_beta.add_trace(go.Bar(
                    name='Up Beta (Greed)',
                    x=metrics_df['Ticker'],
                    y=metrics_df['Up Beta'],
                    marker_color='rgb(16, 185, 129)',
                    text=metrics_df['Up Beta'].round(2),
                    textposition='outside'
                ))
                
                fig_beta.add_trace(go.Bar(
                    name='Down Beta (Fear)',
                    x=metrics_df['Ticker'],
                    y=metrics_df['Down Beta'],
                    marker_color='rgb(239, 68, 68)',
                    text=metrics_df['Down Beta'].round(2),
                    textposition='outside'
                ))
                
                fig_beta.update_layout(
                    template="plotly_dark",
                    height=300,
                    barmode='group',
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", 
                        y=1.02, 
                        xanchor="right", 
                        x=1
                    ),
                    xaxis_title="",
                    yaxis_title="Beta",
                    showlegend=True
                )
                
                st.plotly_chart(fig_beta, use_container_width=True)
            
            with col_beta2:
                st.markdown("**Asymmetry Guide:**")
                st.markdown("**>0.2** = Convex (Good hedge)")
                st.markdown("**0-0.2** = Neutral")
                st.markdown("**<0** = Concave (Risk)")
                
                good_asymmetry = metrics_df[metrics_df['Asymmetry'] > 0.2]
                if len(good_asymmetry) > 0:
                    st.markdown("**Best Asymmetry:**")
                    for _, row in good_asymmetry.iterrows():
                        st.markdown(f"✅ **{row['Ticker']}**: {row['Asymmetry']:.2f}")
                else:
                    st.markdown("*No assets with asymmetry >0.2*")
            
            # ================================================================
            # DETAILED METRICS TABLE
            # ================================================================
            
            st.subheader("Detailed Metrics")
            
            display_df = metrics_df.copy()
            display_df['Correlation'] = display_df['Correlation'].map('{:.2f}'.format)
            display_df['Volatility'] = display_df['Volatility'].map('{:.1f}%'.format)
            display_df['Up Beta'] = display_df['Up Beta'].map('{:.2f}'.format)
            display_df['Down Beta'] = display_df['Down Beta'].map('{:.2f}'.format)
            display_df['Asymmetry'] = display_df['Asymmetry'].map('{:.2f}'.format)
            display_df['30D Return'] = display_df['30D Return'].map('{:.1f}%'.format)
            
            # Add status column
            def get_status(row):
                corr = float(row['Correlation'])
                if corr < -0.3:
                    return "🟢 Hedge"
                elif abs(corr) < 0.3:
                    return "🟡 Diversifier"
                elif corr > 0.7:
                    return "🔴 Concentrated"
                else:
                    return "🔵 Growth"
            
            display_df['Status'] = display_df.apply(get_status, axis=1)
            
            st.dataframe(
                display_df[['Ticker', 'Status', 'Correlation', 'Corr_Trend', 'Trend', 
                           'Volatility', 'Up Beta', 'Down Beta', 'Asymmetry', '30D Return']],
                use_container_width=True,
                height=350
            )
            
            # ================================================================
            # FOOTER
            # ================================================================
            
            st.markdown("---")
            st.caption(f"""
            📊 **Analysis Period:** {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')} | 
            **Rolling Window:** {window} days | **Tickers:** {', '.join(tickers)} | 
            **Trend:** 50D/200D MA | **Source:** Yahoo Finance | **Update:** Hourly cache
            """)
            
        else:
            st.error("❌ Failed to fetch data. Please check your internet connection and try again.")
            st.info("💡 Tip: Try selecting fewer tickers or a shorter time period.")
