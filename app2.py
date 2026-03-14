import matplotlib
matplotlib.use("Agg")
import streamlit as st
import math, warnings, datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import minimize
import yfinance as yf
from zoneinfo import ZoneInfo
warnings.filterwarnings("ignore")

st.set_page_config(page_title="âš¡ Quant Terminal", layout="wide", page_icon="âš¡", initial_sidebar_state="collapsed")
NY  = ZoneInfo("America/New_York")
NOW = datetime.datetime.now(NY)

BS  = dict(primary="#A855F7",light="#D8B4FE",dark="#1E0A2E",mid="#2D1B4E",border="#7C3AED",glow="rgba(168,85,247,0.15)",emoji="ðŸ”®",label="BLACK-SCHOLES")
MC  = dict(primary="#06B6D4",light="#A5F3FC",dark="#021B2E",mid="#0C2D3E",border="#0891B2",glow="rgba(6,182,212,0.15)",emoji="ðŸŽ²",label="MONTE CARLO")
BTH = dict(primary="#10B981",light="#A7F3D0",dark="#022C22",mid="#064E3B",border="#059669",glow="rgba(16,185,129,0.15)",emoji="ðŸŒ³",label="BINOMIAL TREE")
CF  = dict(primary="#F59E0B",light="#FDE68A",dark="#1C1400",mid="#292000",border="#D97706",glow="rgba(245,158,11,0.15)")
QT  = dict(primary="#EC4899",light="#FBCFE8",dark="#1F0010",mid="#3B0023",border="#DB2777",glow="rgba(236,72,153,0.15)")

PLOT_CFG = dict(template="plotly_dark", paper_bgcolor="#020202", plot_bgcolor="#0D0D0D",
                font=dict(family="Courier New", color="#666"), margin=dict(l=50,r=20,t=55,b=30))

st.markdown("""
<style>
html,body,.stApp{background:#020202!important;color:#E5E7EB;}
section[data-testid="stSidebar"]{display:none;}
.block-container{padding-top:0.3rem!important;max-width:100%!important;}
.stTabs [data-baseweb="tab-list"]{background:#020202;border-bottom:1px solid #222;gap:2px;padding:0 4px;}
.stTabs [data-baseweb="tab"]{background:#0a0a0a;color:#555;border:1px solid #222;
  border-radius:6px 6px 0 0;font-family:monospace;font-weight:bold;padding:8px 16px;font-size:12px;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#A855F7,#06B6D4)!important;
  color:#fff!important;border-color:transparent!important;}
.stButton>button{font-family:monospace;font-weight:bold;border:none;border-radius:6px;
  padding:7px 18px;font-size:13px;transition:all 0.2s;}
.bs-btn>button{background:linear-gradient(135deg,#7C3AED,#A855F7);color:#fff;}
.mc-btn>button{background:linear-gradient(135deg,#0891B2,#06B6D4);color:#fff;}
.bt-btn>button{background:linear-gradient(135deg,#059669,#10B981);color:#fff;}
.cf-btn>button{background:linear-gradient(135deg,#D97706,#F59E0B);color:#000;}
.qt-btn>button{background:linear-gradient(135deg,#DB2777,#EC4899);color:#fff;}
.all-btn>button{background:linear-gradient(135deg,#A855F7,#06B6D4,#10B981);color:#fff;font-size:14px;}
label,.stSlider label,.stNumberInput label,.stSelectbox label,.stTextInput label{
  font-family:monospace!important;font-weight:bold!important;font-size:11px!important;color:#888!important;}
input,select{background:#0D0D0D!important;color:#E5E7EB!important;
  border:1px solid #333!important;font-family:monospace!important;border-radius:5px!important;}
h1,h2,h3{font-family:monospace!important;color:#E5E7EB!important;}
</style>""", unsafe_allow_html=True)

# â”€â”€ HELPERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def fp(p):
    if p > 10000: return f"{p:,.0f}"
    elif p > 100: return f"{p:,.2f}"
    elif p > 1:   return f"{p:.4f}"
    elif p > 0:   return f"{p:.6f}"
    return "â€”"

def fc(c):
    a = abs(c)
    if a > 100:  return f"{a:,.2f}"
    elif a > 0.1: return f"{a:.4f}"
    return f"{a:.6f}"

def card(label, value, color="#E5E7EB", sub=""):
    sub_html = f"<div style='color:#444;font-size:10px;margin-top:3px'>{sub}</div>" if sub else ""
    return (
        "<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:8px;"
        "padding:12px 16px;font-family:Courier New,monospace'>"
        f"<div style='color:#555;font-size:10px;margin-bottom:4px'>{label}</div>"
        f"<div style='color:{color};font-size:19px;font-weight:bold'>{value}</div>"
        f"{sub_html}</div>"
    )

def stat_box(col, label, value, color="#E5E7EB", sub=""):
    col.markdown(card(label, value, color, sub), unsafe_allow_html=True)

def themed_card(label, value, theme, hint=""):
    hint_html = f"<div style='color:#333;font-size:9px;margin-top:3px'>{hint}</div>" if hint else ""
    return (
        f"<div style='background:{theme['mid']};border:1px solid {theme['border']};"
        "border-radius:6px;padding:10px 14px;text-align:center;font-family:Courier New,monospace'>"
        f"<div style='color:{theme['primary']};font-size:10px;letter-spacing:1px'>{label}</div>"
        f"<div style='color:{theme['light']};font-size:18px;font-weight:bold'>{value:.5f}</div>"
        f"{hint_html}</div>"
    )

# â”€â”€ DATA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_data(ttl=300)
def batch_quotes(tickers_tuple):
    tickers = list(tickers_tuple)
    result  = {t: dict(price=0.0, chg=0.0, pct=0.0, vol=0) for t in tickers}
    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
        for tk in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    cl = raw["Close"][tk].dropna()
                    vo = raw["Volume"][tk].dropna() if "Volume" in raw else pd.Series(dtype=float)
                else:
                    cl = raw["Close"].dropna()
                    vo = raw["Volume"].dropna() if "Volume" in raw else pd.Series(dtype=float)
                if len(cl) < 2: continue
                p = float(cl.iloc[-1]); pr = float(cl.iloc[-2])
                chg = p - pr; pct = chg / pr * 100
                vol = int(vo.iloc[-1]) if len(vo) > 0 else 0
                result[tk] = dict(price=p, chg=chg, pct=pct, vol=vol)
            except: pass
    except: pass
    return result

@st.cache_data(ttl=300)
def fetch_ohlcv(ticker, period="3mo"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df[df["Close"].notna()].copy()
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_range(ticker, start, end):
    try:
        df = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df[df["Close"].notna()].copy()
    except: return pd.DataFrame()

@st.cache_data(ttl=200)
def get_news(ticker):
    try: return yf.Ticker(ticker).news[:10] or []
    except: return []

@st.cache_data(ttl=300)
def get_info(ticker):
    try: return yf.Ticker(ticker).info or {}
    except: return {}

# â”€â”€ OPTIONS MATH â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def bs_params(St, K, r, sigma, T, q=0):
    d1 = (math.log(St/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return d1, d1 - sigma*math.sqrt(T)

def run_bs(St, K, sigma, T, r, q):
    d1, d2 = bs_params(St, K, r, sigma, T, q)
    sc = St*math.exp(-q*T)*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    sp = K*math.exp(-r*T)*norm.cdf(-d2) - St*math.exp(-q*T)*norm.cdf(-d1)
    dc = math.exp(-r*T)*norm.cdf(d2)
    dp = math.exp(-r*T)*norm.cdf(-d2)
    delta_c = math.exp(-q*T)*norm.cdf(d1)
    delta_p = -math.exp(-q*T)*norm.cdf(-d1)
    gamma   = math.exp(-q*T)*norm.pdf(d1)/(St*sigma*math.sqrt(T))
    vega    = St*math.exp(-q*T)*norm.pdf(d1)*math.sqrt(T)/100
    theta_c = (-(St*norm.pdf(d1)*sigma*math.exp(-q*T))/(2*math.sqrt(T))
               - r*K*math.exp(-r*T)*norm.cdf(d2)
               + q*St*math.exp(-q*T)*norm.cdf(d1))/365
    theta_p = (-(St*norm.pdf(d1)*sigma*math.exp(-q*T))/(2*math.sqrt(T))
               + r*K*math.exp(-r*T)*norm.cdf(-d2)
               - q*St*math.exp(-q*T)*norm.cdf(-d1))/365
    return dict(d1=d1, d2=d2, dc=dc, sc=sc, dp=dp, sp=sp,
                delta_c=delta_c, delta_p=delta_p, gamma=gamma, vega=vega,
                theta_c=theta_c, theta_p=theta_p)

def run_mc(St, K, sigma, T, r, reps):
    np.random.seed(42)
    Z  = np.random.standard_normal(reps)
    ST = St * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    d1, d2 = bs_params(St, K, r, sigma, T)
    return dict(d1=d1, d2=d2,
                dc=np.exp(-r*T)*np.mean(ST>K),
                sc=np.exp(-r*T)*np.mean(np.maximum(ST-K,0)),
                dp=np.exp(-r*T)*np.mean(ST<K),
                sp=np.exp(-r*T)*np.mean(np.maximum(K-ST,0)),
                ST=ST, reps=reps)

def run_bt(St, K, sigma, T, r, N):
    dt = T/N; u = math.exp(sigma*math.sqrt(dt)); d = 1/u
    p  = (math.exp(r*dt)-d)/(u-d)
    ST = np.array([St*(u**j)*(d**(N-j)) for j in range(N+1)])
    cp = np.maximum(ST-K,0); pp = np.maximum(K-ST,0)
    dcp = (ST>K).astype(float); dpp = (ST<K).astype(float)
    for _ in range(N, 0, -1):
        cp  = np.exp(-r*dt)*(p*cp[1:] +(1-p)*cp[:-1])
        pp  = np.exp(-r*dt)*(p*pp[1:] +(1-p)*pp[:-1])
        dcp = np.exp(-r*dt)*(p*dcp[1:]+(1-p)*dcp[:-1])
        dpp = np.exp(-r*dt)*(p*dpp[1:]+(1-p)*dpp[:-1])
    d1, d2 = bs_params(St, K, r, sigma, T)
    return dict(d1=d1, d2=d2, dc=dcp[0], sc=cp[0], dp=dpp[0], sp=pp[0])

# â”€â”€ PORTFOLIO MATH â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def port_stats(w, mu, cov):
    ret = float(np.dot(w, mu))
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol

def max_sharpe_w(mu, cov, rf):
    n = len(mu)
    def neg_sh(w):
        r, v = port_stats(w, mu, cov)
        return -(r-rf)/v if v > 1e-10 else 0
    res = minimize(neg_sh, np.ones(n)/n, bounds=[(0,1)]*n,
                   constraints={'type':'eq','fun':lambda x: sum(x)-1},
                   method='SLSQP', options={'ftol':1e-12,'maxiter':1000})
    return res.x if res.success else np.ones(n)/n

def min_var_w(mu, cov):
    n = len(mu)
    res = minimize(lambda w: port_stats(w, mu, cov)[1], np.ones(n)/n,
                   bounds=[(0,1)]*n,
                   constraints={'type':'eq','fun':lambda x: sum(x)-1},
                   method='SLSQP', options={'ftol':1e-12,'maxiter':1000})
    return res.x if res.success else np.ones(n)/n

def calc_frontier(mu, cov, n_pts=60):
    mv = min_var_w(mu, cov)
    mv_r, _ = port_stats(mv, mu, cov)
    max_r = float(np.max(mu)) * 0.92
    targets = np.linspace(mv_r, max_r, n_pts)
    vols, rets = [], []
    x0 = mv.copy()
    for t in targets:
        try:
            res = minimize(
                lambda w: port_stats(w, mu, cov)[1], x0,
                bounds=[(0,1)]*len(mu),
                constraints=[
                    {'type':'eq','fun':lambda x: sum(x)-1},
                    {'type':'eq','fun':lambda x,tgt=t: port_stats(x,mu,cov)[0]-tgt}
                ],
                method='SLSQP', options={'ftol':1e-12,'maxiter':500})
            if res.success and res.fun > 1e-6:
                r, v = port_stats(res.x, mu, cov)
                if abs(r-t) < 0.01:
                    vols.append(v); rets.append(r); x0 = res.x.copy()
        except: pass
    return np.array(vols), np.array(rets)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  HEADER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown(
    "<div style='background:linear-gradient(90deg,#0A0010,#001020,#001A0E);"
    "padding:14px 24px;border-bottom:1px solid #222;"
    "display:flex;justify-content:space-between;align-items:center;margin-bottom:0'>"
    "<div>"
    "<span style='font-size:24px;font-weight:bold;font-family:monospace;"
    "background:linear-gradient(90deg,#A855F7,#06B6D4,#10B981,#F59E0B);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:3px'>"
    "âš¡ QUANT TERMINAL"
    "</span>"
    "<span style='color:#333;font-family:monospace;font-size:11px;margin-left:12px'>"
    "Market Intelligence Â· Options Â· Quant Â· CFO Suite"
    "</span>"
    "</div>"
    f"<div style='font-family:monospace;font-size:11px;color:#444'>"
    f"{NOW.strftime('%A %b %d, %Y  |  %I:%M %p ET')} &nbsp;|&nbsp;"
    "<span style='color:#10B981'>â— LIVE</span>"
    "</div></div>",
    unsafe_allow_html=True
)

#  MAIN TABS
T_MKT, T_SIG, T_CHART, T_CALC, T_CFO, T_QUANT = st.tabs([
    "🌍 MARKET OVERVIEW",
    "📡 STOCK SIGNALS",
    "📊 CHART ANALYZER",
    "📐 FINANCE CALCULATOR",
    "💼 CFO CALCULATORS",
    "🔬 QUANT LAB",
])


#  TAB 1 â€” MARKET OVERVIEW

with T_MKT:
    if st.button("ðŸ”„ Refresh", key="ref_mkt"): st.cache_data.clear()
    INDICES = {
        "S&P 500":"^GSPC","NASDAQ":"^IXIC","Dow Jones":"^DJI","Russell 2K":"^RUT",
        "VIX":"^VIX","10Y Yield":"^TNX","Gold":"GC=F","WTI Oil":"CL=F",
        "BTC":"BTC-USD","EUR/USD":"EURUSD=X","USD/JPY":"USDJPY=X","Silver":"SI=F"
    }
    with st.spinner("Loading market data..."):
        q = batch_quotes(tuple(INDICES.values()))

    bar_html = "<div style='display:flex;gap:8px;flex-wrap:wrap;padding:10px 0;margin-bottom:8px'>"
    for name, tk in INDICES.items():
        d = q.get(tk, dict(price=0, chg=0, pct=0))
        cc = "#00FF41" if d["chg"] >= 0 else "#FF4444"
        sg = "â–²" if d["chg"] >= 0 else "â–¼"
        bar_html += (
            "<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:6px;"
            "padding:7px 12px;font-family:monospace;min-width:110px'>"
            f"<div style='color:#777;font-size:9px'>{name}</div>"
            f"<div style='color:#E5E7EB;font-size:13px;font-weight:bold'>{fp(d['price'])}</div>"
            f"<div style='color:{cc};font-size:11px'>{sg}{abs(d['pct']):.2f}%</div></div>"
        )
    st.markdown(bar_html + "</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        advancing = 2007; declining = 3315; total = advancing + declining
        adv_pct = advancing / total * 100
        st.markdown(
            "<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:8px;"
            "padding:14px;margin-bottom:10px'>"
            "<div style='color:#A855F7;font-family:monospace;font-weight:bold;"
            "font-size:12px;margin-bottom:10px'>📊 MARKET BREADTH</div>"
            "<div style='display:flex;gap:20px;flex-wrap:wrap'>"
            "<div><div style='color:#555;font-size:10px;font-family:monospace'>ADVANCING</div>"
            f"<div style='color:#00FF41;font-size:18px;font-weight:bold;font-family:monospace'>{advancing:,}</div></div>"
            "<div><div style='color:#555;font-size:10px;font-family:monospace'>DECLINING</div>"
            f"<div style='color:#FF4444;font-size:18px;font-weight:bold;font-family:monospace'>{declining:,}</div></div>"
            "<div><div style='color:#555;font-size:10px;font-family:monospace'>ABOVE SMA50</div>"
            "<div style='color:#06B6D4;font-size:18px;font-weight:bold;font-family:monospace'>29.0%</div></div>"
            "<div><div style='color:#555;font-size:10px;font-family:monospace'>ABOVE SMA200</div>"
            "<div style='color:#06B6D4;font-size:18px;font-weight:bold;font-family:monospace'>40.8%</div></div>"
            "<div><div style='color:#555;font-size:10px;font-family:monospace'>NEW HIGH</div>"
            "<div style='color:#F59E0B;font-size:18px;font-weight:bold;font-family:monospace'>73</div></div>"
            "<div><div style='color:#555;font-size:10px;font-family:monospace'>NEW LOW</div>"
            "<div style='color:#FF4444;font-size:18px;font-weight:bold;font-family:monospace'>267</div></div>"
            "</div>"
            "<div style='margin-top:10px;background:#111;border-radius:4px;height:10px;overflow:hidden'>"
            f"<div style='background:linear-gradient(90deg,#00FF41,#10B981);"
            f"height:100%;width:{adv_pct:.1f}%'></div></div></div>",
            unsafe_allow_html=True
        )
        g2a, g2b = st.columns(2)
        GAINERS = [("BIAF",2.12,98.13),("AIFF",2.75,58.05),("SVCO",5.03,52.42),
                   ("ELPW",5.16,43.73),("PLYX",6.36,36.77),("APEI",57.66,21.19)]
        LOSERS  = [("IMMP",0.48,-81.64),("IBG",1.10,-55.28),("CMCT",0.63,-44.31),
                   ("BHAT",0.68,-43.75),("KLC",1.95,-42.65),("CDIO",2.80,-38.60)]
        def sig_tbl(title, data, color, icon):
            rows = "".join(
                f"<tr><td style='color:{color};padding:5px 8px;font-weight:bold'>{t}</td>"
                f"<td style='color:#E5E7EB;padding:5px 8px'>{px}</td>"
                f"<td style='color:{color};padding:5px 8px'>{pc:+.2f}%</td></tr>"
                for t, px, pc in data
            )
            return (
                f"<div style='background:#0D0D0D;border:1px solid #1a1a1a;"
                f"border-radius:8px;padding:12px;margin-bottom:8px'>"
                f"<div style='color:{color};font-family:monospace;font-weight:bold;"
                f"font-size:12px;margin-bottom:8px'>{icon} {title}</div>"
                f"<table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:12px'>"
                f"<thead><tr>"
                f"<td style='color:#555;padding:4px 8px;font-size:10px'>TICKER</td>"
                f"<td style='color:#555;padding:4px 8px;font-size:10px'>PRICE</td>"
                f"<td style='color:#555;padding:4px 8px;font-size:10px'>CHG%</td>"
                f"</tr></thead><tbody>{rows}</tbody></table></div>"
            )
        with g2a: st.markdown(sig_tbl("TOP GAINERS", GAINERS, "#00FF41", "ðŸš€"), unsafe_allow_html=True)
        with g2b: st.markdown(sig_tbl("TOP LOSERS",  LOSERS,  "#FF4444", "ðŸ“‰"), unsafe_allow_html=True)

    with col2:
        FUTURES = [("Crude Oil","99.31","+3.74%","#00FF41"),("Natural Gas","3.13","-3.12%","#FF4444"),
                   ("Gold","5,023","-2.00%","#FF4444"),("S&P 500 Fut","6,625","-1.52%","#FF4444"),
                   ("Nasdaq Fut","24,335","-1.76%","#FF4444"),("Dow Fut","46,511","-1.07%","#FF4444")]
        rows = "".join(
            f"<tr><td style='color:#E5E7EB;padding:6px 10px;font-weight:bold'>{n}</td>"
            f"<td style='color:#F59E0B;padding:6px 10px'>{px}</td>"
            f"<td style='color:{c};padding:6px 10px'>{pc}</td></tr>"
            for n, px, pc, c in FUTURES
        )
        st.markdown(
            "<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:8px;padding:12px;margin-bottom:10px'>"
           "<div style='color:#F59E0B;font-family:monospace;font-weight:bold;font-size:12px;margin-bottom:8px'>⚡ FUTURES</div>"
            f"<table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:12px'><tbody>{rows}</tbody></table></div>",
            unsafe_allow_html=True
        )
        ECON = [("GDP Price Index QoQ","3.8%","3.7%","#F59E0B"),
                ("PCE Price Index YoY","2.8%","2.9%","#00FF41"),
                ("Personal Income MoM","0.4%","0.5%","#FF4444"),
                ("Michigan Sentiment","55.5","55.0","#F59E0B"),
                ("JOLTs Job Openings","6.95M","6.70M","#00FF41")]
        rows = "".join(
            f"<tr><td style='color:#aaa;padding:5px 8px;font-size:11px'>{ev}</td>"
            f"<td style='color:{c};padding:5px 8px;font-weight:bold'>{act}</td>"
            f"<td style='color:#555;padding:5px 8px'>{exp}</td></tr>"
            for ev, act, exp, c in ECON
        )
        st.markdown(
            "<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:8px;padding:12px;margin-bottom:10px'>"
            "<div style='color:#06B6D4;font-family:monospace;font-weight:bold;font-size:12px;margin-bottom:8px'>📅 ECONOMIC CALENDAR</div>"
            "<table style='width:100%;border-collapse:collapse;font-family:monospace'>"
            "<thead><tr>"
            "<td style='color:#555;padding:4px 8px;font-size:10px'>EVENT</td>"
            "<td style='color:#555;padding:4px 8px;font-size:10px'>ACTUAL</td>"
            "<td style='color:#555;padding:4px 8px;font-size:10px'>EXPECTED</td>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>",
            unsafe_allow_html=True
        )
        INSIDERS = [("AMR","Buy","$1.87M","#00FF41"),("XENE","Sale","$403K","#FF4444"),
                    ("VRT","Prop. Sale","$263M","#FF4444"),("SVRE","Buy","$688M","#00FF41")]
        rows = "".join(
            f"<tr><td style='color:#F59E0B;padding:5px 8px;font-weight:bold'>{t}</td>"
            f"<td style='color:{c};padding:5px 8px'>{tx}</td>"
            f"<td style='color:#aaa;padding:5px 8px'>{v}</td></tr>"
            for t, tx, v, c in INSIDERS
        )
        st.markdown(
            "<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:8px;padding:12px'>"
            "<div style='color:#EC4899;font-family:monospace;font-weight:bold;font-size:12px;margin-bottom:8px'>🕵️ INSIDER ACTIVITY</div>"
            f"<table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:12px'><tbody>{rows}</tbody></table></div>",
            unsafe_allow_html=True
        )

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  TAB 2 â€” STOCK SIGNALS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with T_SIG:
    st.markdown("<h4 style='color:#A855F7;font-family:monospace;margin-bottom:10px'>📡 LIVE STOCK SIGNALS</h4>", unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns([2, 2, 8])
    with sc1: screen_tk = st.text_input("Ticker", value="AAPL", key="sc_tk").upper().strip()
    with sc2: screen_per = st.selectbox("Period", ["1mo","3mo","6mo","1y"], index=2, key="sc_per")
    with sc3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_screen = st.button("ðŸ” Analyze Stock", key="run_screen")

    if (run_screen or screen_tk) and screen_tk:
        with st.spinner(f"Analyzing {screen_tk}..."):
            df   = fetch_ohlcv(screen_tk, screen_per)
            info = get_info(screen_tk)
            qd   = batch_quotes((screen_tk,)).get(screen_tk, dict(price=0,chg=0,pct=0,vol=0))
            news = get_news(screen_tk)

        if not df.empty and len(df) > 30:
            c_s = df["Close"]
            sma20 = float(c_s.rolling(20).mean().iloc[-1])
            sma50 = float(c_s.rolling(50).mean().iloc[-1]) if len(c_s) > 50 else sma20
            ema12 = c_s.ewm(span=12).mean(); ema26 = c_s.ewm(span=26).mean()
            macd_l = float((ema12 - ema26).iloc[-1])
            sig_l  = float(((ema12 - ema26).ewm(span=9).mean()).iloc[-1])
            dd = c_s.diff(); g = dd.clip(lower=0).rolling(14).mean()
            lo = (-dd.clip(upper=0)).rolling(14).mean()
            rsi_val = float((100 - 100/(1 + g/lo.replace(0, np.nan))).iloc[-1])
            bb_mid = float(c_s.rolling(20).mean().iloc[-1])
            bb_std = float(c_s.rolling(20).std().iloc[-1])
            bb_up = bb_mid + 2*bb_std; bb_lo = bb_mid - 2*bb_std
            hi52 = float(c_s.tail(252).max()) if len(c_s) > 252 else float(c_s.max())
            lo52 = float(c_s.tail(252).min()) if len(c_s) > 252 else float(c_s.min())
            price = qd["price"] if qd["price"] > 0 else float(c_s.iloc[-1])

            signals = []
            signals.append(("🔺 Above SMA20","#00FF41") if price > sma20 else ("🔻 Below SMA20","#FF4444"))
            signals.append(("🔺 Above SMA50","#00FF41") if price > sma50 else ("🔻 Below SMA50","#FF4444"))
            signals.append(("📈 MACD Bullish","#00FF41") if macd_l > sig_l else ("BEARISH 🐻","#FF4444"))
            if rsi_val > 70:
            signals.append(("🔴 RSI Overbought","#FF4444"))
            elif rsi_val < 30:
            signals.append(("🚨 RSI Oversold","#00FF41"))
            else:
            signals.append((f"⚪ RSI Neutral {rsi_val:.1f}","#F59E0B"))

if price > bb_up:
    signals.append(("📊 Above BB Upper","#FF4444"))
elif price < bb_lo:
    signals.append(("📊 Below BB Lower","#00FF41"))
else:
    signals.append(("📊 Inside BB","#06B6D4"))


            bull = sum(1 for _, c2 in signals if "#00FF41" in c2)
            bear = sum(1 for _, c2 in signals if "#FF4444" in c2)
            sc_col = "#00FF41" if bull > bear else ("#FF4444" if bear > bull else "#F59E0B")
            sc_lbl = "BULLISH ðŸ‚" if bull > bear else ("BEARISH 🐻" if bear > bull else "NEUTRAL âš–ï¸")
            chg_c  = "#00FF41" if qd["chg"] >= 0 else "#FF4444"
            chg_sg = "â–²" if qd["chg"] >= 0 else "â–¼"
            rsi_c  = "#FF4444" if rsi_val > 70 else ("#00FF41" if rsi_val < 30 else "#F59E0B")

            sa, sb, sc2x, sd = st.columns(4)
            stat_box(sa, "PRICE",    fp(price),                                    "#F59E0B")
            stat_box(sb, "CHANGE",   f"{chg_sg}{fc(qd['chg'])} ({qd['pct']:+.2f}%)", chg_c)
            stat_box(sc2x, "RSI (14)", f"{rsi_val:.1f}",                              rsi_c)
            stat_box(sd, "SIGNAL",   sc_lbl,                                        sc_col, f"{bull} bull Â· {bear} bear")

            sig_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin:12px 0'>"
            for s, c2 in signals:
                sig_html += (
                    f"<span style='background:#0D0D0D;color:{c2};"
                    f"border:1px solid {c2}44;padding:5px 12px;"
                    f"border-radius:20px;font-family:monospace;font-size:11px'>{s}</span>"
                )
            st.markdown(sig_html + "</div>", unsafe_allow_html=True)

            fundamentals = {
                "Market Cap":    f"${info.get('marketCap',0)/1e9:.1f}B" if info.get('marketCap') else "â€”",
                "P/E Ratio":     f"{info.get('trailingPE',0):.1f}x"     if info.get('trailingPE') else "â€”",
                "EPS (TTM)":     f"${info.get('trailingEps',0):.2f}"    if info.get('trailingEps') else "â€”",
                "Revenue":       f"${info.get('totalRevenue',0)/1e9:.1f}B" if info.get('totalRevenue') else "â€”",
                "Profit Margin": f"{info.get('profitMargins',0)*100:.1f}%" if info.get('profitMargins') else "â€”",
                "52W High": fp(hi52), "52W Low": fp(lo52),
                "Beta":      f"{info.get('beta',0):.2f}"           if info.get('beta') else "â€”",
                "Div Yield": f"{info.get('dividendYield',0)*100:.2f}%" if info.get('dividendYield') else "0%",
                "Volume":    f"{qd['vol']:,.0f}",
            }
            fund_html = "<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:12px'>"
            for k, v in fundamentals.items():
                fund_html += (
                    "<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:6px;padding:10px 12px'>"
                    f"<div style='color:#555;font-size:10px;font-family:monospace'>{k}</div>"
                    f"<div style='color:#E5E7EB;font-size:14px;font-weight:bold;font-family:monospace'>{v}</div>"
                    "</div>"
                )
            st.markdown(fund_html + "</div>", unsafe_allow_html=True)

            d1v, _, d3v = st.columns([4, 1, 2])
            with d1v:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[.75,.25], vertical_spacing=0.02)
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                    increasing_line_color="#00FF41", decreasing_line_color="#FF4444", name=screen_tk
                ), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=c_s.rolling(20).mean(),
                    line=dict(color="#F59E0B", width=1.2), name="SMA20"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=c_s.rolling(50).mean(),
                    line=dict(color="#06B6D4", width=1.2), name="SMA50"), row=1, col=1)
                bb_m = c_s.rolling(20).mean(); bb_sd = c_s.rolling(20).std()
                fig.add_trace(go.Scatter(x=df.index, y=bb_m+2*bb_sd,
                    line=dict(color="rgba(168,85,247,0.4)", width=1, dash="dash"), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=bb_m-2*bb_sd,
                    line=dict(color="rgba(168,85,247,0.4)", width=1, dash="dash"),
                    fill="tonexty", fillcolor="rgba(168,85,247,0.03)", showlegend=False), row=1, col=1)
                vcols = ["#00FF41" if float(cl) >= float(op) else "#FF4444"
                         for cl, op in zip(df["Close"], df["Open"])]
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                    marker_color=vcols, showlegend=False, opacity=0.6), row=2, col=1)
                fig.update_layout(**PLOT_CFG, height=420, xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", x=0, y=1.05,
                                font=dict(family="Courier New", size=9), bgcolor="rgba(0,0,0,0)"))
                fig.update_xaxes(gridcolor="#111"); fig.update_yaxes(gridcolor="#111")
                st.plotly_chart(fig, use_container_width=True)

            with d3v:
                if news:
                    st.markdown(
                        f"<div style='color:#A855F7;font-family:monospace;font-weight:bold;"
                        f"font-size:11px;margin-bottom:6px'>ðŸ“° {screen_tk} NEWS</div>",
                        unsafe_allow_html=True
                    )
                    for item in news[:6]:
                        try:
                            ct    = item.get("content", {}); 
                            title = ct.get("title", item.get("title",""))[:70]
                            pub   = ct.get("pubDate","")[:10]
                            st.markdown(
                                "<div style='background:#0D0D0D;border-left:2px solid #A855F7;"
                                "padding:6px 10px;margin-bottom:6px;border-radius:0 4px 4px 0;"
                                "font-family:monospace'>"
                                f"<div style='color:#ccc;font-size:11px;line-height:1.4'>{title}</div>"
                                f"<div style='color:#444;font-size:9px;margin-top:3px'>{pub}</div>"
                                "</div>",
                                unsafe_allow_html=True
                            )
                        except: continue
        else:
            st.warning(f"Not enough data for {screen_tk}. Try a different ticker or period.")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  TAB 3 â€” CHART ANALYZER (up to 3 tickers)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with T_CHART:
    st.markdown("<h4 style='color:#06B6D4;font-family:monospace;margin-bottom:10px'>ðŸ“Š TECHNICAL CHART ANALYZER</h4>", unsafe_allow_html=True)
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 4])
    with cc1: tk1 = st.text_input("Ticker 1", value="SPY",  key="ch_tk1").upper().strip()
    with cc2: tk2 = st.text_input("Ticker 2", value="QQQ",  key="ch_tk2").upper().strip()
    with cc3: tk3 = st.text_input("Ticker 3 (optional)", value="", key="ch_tk3").upper().strip()
    with cc4: chart_per = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=2, key="ch_per")

    chart_tickers = [t for t in [tk1, tk2, tk3] if t]

    def build_chart(ticker, period):
        df2 = fetch_ohlcv(ticker, period)
        if df2.empty or len(df2) < 10:
            st.warning(f"No data for {ticker}"); return
        c2 = df2["Close"]
        dd = c2.diff(); g = dd.clip(lower=0).rolling(14).mean()
        lo = (-dd.clip(upper=0)).rolling(14).mean()
        df2["RSI"]  = 100 - 100/(1 + g/lo.replace(0, np.nan))
        macd_l = c2.ewm(span=12,min_periods=1).mean() - c2.ewm(span=26,min_periods=1).mean()
        df2["MACD"] = macd_l
        df2["SIG"]  = macd_l.ewm(span=9,min_periods=1).mean()
        df2["HIST"] = macd_l - df2["SIG"]
        df2["BB_UP"]  = c2.rolling(20,min_periods=1).mean() + 2*c2.rolling(20,min_periods=1).std()
        df2["BB_LO"]  = c2.rolling(20,min_periods=1).mean() - 2*c2.rolling(20,min_periods=1).std()
        df2["EMA20"]  = c2.ewm(span=20,min_periods=1).mean()
        df2["EMA50"]  = c2.ewm(span=50,min_periods=1).mean()
        df2 = df2.dropna(subset=["Close","Open"])
        last = float(df2["Close"].iloc[-1]); prev = float(df2["Close"].iloc[-2])
        chg  = last - prev; pct = chg/prev*100
        ct   = "#00FF41" if chg >= 0 else "#FF4444"
        sg   = "â–²" if chg >= 0 else "â–¼"

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[.50,.15,.18,.17], vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(x=df2.index, open=df2["Open"], high=df2["High"],
            low=df2["Low"], close=df2["Close"],
            increasing_line_color="#00FF41", decreasing_line_color="#FF4444",
            name=ticker), row=1, col=1)
        fig.add_trace(go.Scatter(x=df2.index, y=df2["BB_UP"],
            line=dict(color="rgba(168,85,247,0.4)",width=1,dash="dash"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df2.index, y=df2["BB_LO"],
            line=dict(color="rgba(168,85,247,0.4)",width=1,dash="dash"),
            fill="tonexty", fillcolor="rgba(168,85,247,0.04)", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df2.index, y=df2["EMA20"],
            line=dict(color="#F59E0B",width=1.3), name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df2.index, y=df2["EMA50"],
            line=dict(color="#06B6D4",width=1.3), name="EMA50"), row=1, col=1)
        vcols = ["#00FF41" if float(cl)>=float(op) else "#FF4444"
                 for cl,op in zip(df2["Close"],df2["Open"])]
        fig.add_trace(go.Bar(x=df2.index, y=df2["Volume"],
            marker_color=vcols, showlegend=False, opacity=0.6), row=2, col=1)
        fig.add_trace(go.Scatter(x=df2.index, y=df2["RSI"],
            line=dict(color="#06B6D4",width=1.3), showlegend=False), row=3, col=1)
        fig.add_hline(y=70, line=dict(color="#FF4444",dash="dash",width=0.8), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="#00FF41",dash="dash",width=0.8), row=3, col=1)
        hcols = ["#00FF41" if v >= 0 else "#FF4444" for v in df2["HIST"]]
        fig.add_trace(go.Bar(x=df2.index, y=df2["HIST"],
            marker_color=hcols, showlegend=False, opacity=0.7), row=4, col=1)
        fig.add_trace(go.Scatter(x=df2.index, y=df2["MACD"],
            line=dict(color="#06B6D4",width=1.3), name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df2.index, y=df2["SIG"],
            line=dict(color="#A855F7",width=1.3), name="Signal"), row=4, col=1)
        fig.update_layout(
            **PLOT_CFG, height=680, xaxis_rangeslider_visible=False,
            title=dict(
                text=f"<b style='color:#A855F7'>{ticker}</b>  "
                     f"<span style='color:{ct}'>{fp(last)} {sg}{fc(chg)} ({pct:+.2f}%)</span>",
                font=dict(family="Courier New", size=14), x=0
            ),
            legend=dict(orientation="h", x=0, y=1.02,
                        font=dict(family="Courier New", size=9), bgcolor="rgba(0,0,0,0)")
        )
        fig.update_xaxes(gridcolor="#111"); fig.update_yaxes(gridcolor="#111")
        st.plotly_chart(fig, use_container_width=True)

    for tk in chart_tickers:
        with st.spinner(f"Building {tk} chart..."):
            build_chart(tk, chart_per)
        st.markdown("<hr style='border-color:#111;margin:4px 0'>", unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ════════════════════════════════════════════════════════════
# ── CFA CALCULATORS BY CATEGORY ─────────────────────────────
cfa_tab_q, cfa_tab_fi, cfa_tab_eq, cfa_tab_pm, cfa_tab_der = st.tabs(
    ["QUANTITATIVE", "FIXED INCOME", "EQUITY", "PORTFOLIO", "DERIVATIVES"]
)

# ================= QUANTITATIVE METHODS TAB =================
with cfa_tab_q:
    st.markdown("### 📊 Quantitative Methods Calculators")

    # ---- Future Value & Present Value ----
    col_fv, col_pv, col_perp = st.columns(3)

    with col_fv:
        st.caption("Future Value: FV = PV × (1 + r)^N")
        q_pv = st.number_input("PV", value=1000.0, key="q_pv")
        q_r  = st.number_input("Rate r", value=0.05, format="%.4f", key="q_r")
        q_n  = st.number_input("Periods N", value=10, step=1, key="q_n")
        q_fv = q_pv * (1 + q_r) ** q_n
        st.write(f"FV = {q_fv:,.2f}")

    with col_pv:
        st.caption("Present Value: PV = FV / (1 + r)^N")
        q_fv2 = st.number_input("FV", value=1000.0, key="q_fv2")
        q_r2  = st.number_input("Rate r ", value=0.05, format="%.4f", key="q_r2")
        q_n2  = st.number_input("Periods N ", value=5, step=1, key="q_n2")
        q_pv2 = q_fv2 / ((1 + q_r2) ** q_n2)
        st.write(f"PV = {q_pv2:,.2f}")

    with col_perp:
        st.caption("Perpetuity PV: PV = PMT / r")
        q_pmt = st.number_input("PMT", value=100.0, key="q_pmt")
        q_r3  = st.number_input("Discount rate r", value=0.04, format="%.4f", key="q_r3")
        q_pv_perp = q_pmt / q_r3 if q_r3 != 0 else 0.0
        st.write(f"PV = {q_pv_perp:,.2f}")

    st.markdown("---")

    # ---- HPR & Geometric Mean Return ----
    col_hpr, col_geo = st.columns(2)

    with col_hpr:
        st.caption("Holding Period Return: HPR = (P_t − P_{t−1} + D_t) / P_{t−1}")
        q_pt1 = st.number_input("P_t (ending price)", value=110.0, key="q_pt1")
        q_pt0 = st.number_input("P_{t−1} (begin price)", value=100.0, key="q_pt0")
        q_dt  = st.number_input("D_t (dividend)", value=2.0, key="q_dt")
        hpr = (q_pt1 - q_pt0 + q_dt) / q_pt0 if q_pt0 != 0 else 0.0
        st.write(f"HPR = {hpr:.4f}  ({hpr*100:.2f}%)")

    with col_geo:
        st.caption("Geometric Mean: R_G = [(1+R1)…(1+Rn)]^(1/n) − 1")
        q_geo_str = st.text_input("Returns (comma %, e.g. 10,-5,8)", value="10,-5,8", key="q_geo")
        try:
            vals = [float(x.strip())/100 for x in q_geo_str.split(",") if x.strip() != ""]
            if len(vals) > 0:
                prod = 1.0
                for r in vals:
                    prod *= (1 + r)
                r_geo = prod**(1/len(vals)) - 1
                st.write(f"Geometric mean = {r_geo:.4f}  ({r_geo*100:.2f}%)")
            else:
                st.write("Enter at least one return.")
        except Exception:
            st.write("Check return format.")

    st.markdown("---")

    # ---- Sharpe, CV, Z‑score ----
    col_sh, col_cv, col_z = st.columns(3)

    with col_sh:
        st.caption("Sharpe Ratio: (r_p − r_f) / σ_p")
        q_rp = st.number_input("Portfolio return r_p", value=0.08, format="%.4f", key="q_rp")
        q_rf = st.number_input("Risk‑free r_f",       value=0.03, format="%.4f", key="q_rf")
        q_sp = st.number_input("σ_p",                 value=0.15, format="%.4f", key="q_sp")
        sharpe = (q_rp - q_rf) / q_sp if q_sp != 0 else 0.0
        st.write(f"Sharpe = {sharpe:.3f}")

    with col_cv:
        st.caption("Coefficient of Variation: CV = s / X̄")
        q_s  = st.number_input("Standard deviation s", value=0.12, format="%.4f", key="q_s")
        q_xb = st.number_input("Mean X̄",              value=0.10, format="%.4f", key="q_xb")
        cv = q_s / q_xb if q_xb != 0 else 0.0
        st.write(f"CV = {cv:.3f}")

    with col_z:
        st.caption("Z‑Score: z = (x − μ) / σ")
        q_x  = st.number_input("Observation x", value=75.0, key="q_x")
        q_mu = st.number_input("Mean μ",        value=70.0, key="q_mu")
        q_sd = st.number_input("Std dev σ",     value=10.0, key="q_sd")
        z = (q_x - q_mu) / q_sd if q_sd != 0 else 0.0
        st.write(f"z = {z:.3f}")

    st.markdown("---")

    # ---- Confidence Interval & Correlation ----
    col_ci, col_corr = st.columns(2)

    with col_ci:
        st.caption("CI: X̄ ± z_{α/2} × (σ/√n)")
        q_xb2 = st.number_input("Sample mean X̄", value=10.0, key="q_xb2")
        q_sig = st.number_input("Population σ",   value=2.0,  key="q_sig")
        q_n3  = st.number_input("Sample size n",  value=30,   step=1, key="q_n3")
        q_zc  = st.number_input("z_{α/2}",        value=1.96, key="q_zc")
        if q_n3 > 0:
            se = q_sig / (q_n3 ** 0.5)
            low = q_xb2 - q_zc * se
            high = q_xb2 + q_zc * se
            st.write(f"CI: [{low:.3f}, {high:.3f}]")
        else:
            st.write("n must be > 0.")

    with col_corr:
        st.caption("Correlation: corr = COV(X,Y) / (σ_X σ_Y)")
        q_cov = st.number_input("Covariance COV(X,Y)", value=0.015, format="%.4f", key="q_cov")
        q_sx  = st.number_input("σ_X", value=0.10, format="%.4f", key="q_sx")
        q_sy  = st.number_input("σ_Y", value=0.12, format="%.4f", key="q_sy")
        corr = q_cov / (q_sx * q_sy) if q_sx != 0 and q_sy != 0 else 0.0
        st.write(f"Correlation = {corr:.3f}")
# ================= FIXED INCOME TAB =================
with cfa_tab_fi:
    st.markdown("### 🏦 Fixed Income Calculators")

    # ---- Basic Bond Pricing (level YTM) ----
    st.subheader("Bond Price (Level Yield)")
    col_basic1, col_basic2 = st.columns(2)

    with col_basic1:
        fi_par   = st.number_input("Par value (FV)", value=1000.0, key="fi_par")
        fi_coupon_rate = st.number_input("Annual coupon rate", value=0.05, format="%.4f", key="fi_cpn_rate")
        fi_yield = st.number_input("Yield to maturity (YTM)", value=0.06, format="%.4f", key="fi_ytm")
    with col_basic2:
        fi_maturity = st.number_input("Years to maturity", value=5.0, step=1.0, key="fi_mat")
        fi_freq     = st.number_input("Coupons per year (m)", value=2, step=1, min_value=1, key="fi_freq")

    cpn = fi_par * fi_coupon_rate / fi_freq
    n   = int(fi_maturity * fi_freq)
    r   = fi_yield / fi_freq

    pv_coupons = sum([cpn / ((1 + r) ** t) for t in range(1, n + 1)])
    pv_par     = fi_par / ((1 + r) ** n)
    fi_price   = pv_coupons + pv_par

    st.write(f"Bond price = {fi_price:,.2f}")

    st.markdown("---")

    # ---- Flat price, accrued interest, full price (Street convention simplified) ----
    st.subheader("Flat Price, Accrued Interest, Full Price")
    col_ai1, col_ai2 = st.columns(2)

    with col_ai1:
        clean_coupon_rate = st.number_input("Coupon rate (for AI)", value=0.05, format="%.4f", key="ai_cpn_rate")
        clean_freq        = st.number_input("Coupons per year", value=2, step=1, min_value=1, key="ai_freq")
        days_since_last   = st.number_input("Days since last coupon", value=30.0, key="ai_days_since")
        days_in_period    = st.number_input("Days in coupon period", value=180.0, key="ai_days_period")
    with col_ai2:
        quoted_clean_price = st.number_input("Quoted flat (clean) price", value=fi_price, key="ai_clean_price")

    cpn_ai = fi_par * clean_coupon_rate / clean_freq
    accrued_interest = cpn_ai * (days_since_last / days_in_period)
    full_price = quoted_clean_price + accrued_interest

    st.write(f"Accrued interest = {accrued_interest:,.4f}")
    st.write(f"Full (dirty) price = {full_price:,.2f}")

    st.markdown("---")

    # ---- Current yield and simple yield ----
    st.subheader("Current Yield & Simple Yield")
    col_y1, col_y2 = st.columns(2)

    with col_y1:
        cy_coupon = st.number_input("Annual coupon (amount)", value=fi_par * fi_coupon_rate, key="cy_coupon_amt")
        cy_price  = st.number_input("Bond price for current yield", value=fi_price, key="cy_price")
        current_yield = cy_coupon / cy_price if cy_price != 0 else 0.0
        st.write(f"Current yield = {current_yield:.4f}  ({current_yield*100:.2f}%)")

    with col_y2:
        sy_years_to_maturity = st.number_input("Years to maturity (for simple yield)", value=fi_maturity, key="sy_years")
        sy_par               = st.number_input("Par value (for simple yield)", value=fi_par, key="sy_par")
        sy_price             = st.number_input("Price (for simple yield)", value=fi_price, key="sy_price")
        if sy_years_to_maturity > 0 and sy_price != 0:
            annual_coupon = sy_par * fi_coupon_rate
            straight_line_amort = (sy_par - sy_price) / sy_years_to_maturity
            simple_yield = (annual_coupon + straight_line_amort) / sy_price
        else:
            simple_yield = 0.0
        st.write(f"Simple yield = {simple_yield:.4f}  ({simple_yield*100:.2f}%)")

    st.markdown("---")

    # ---- Money market discount & add-on rates ----
    st.subheader("Money Market Instruments")
    col_mm1, col_mm2 = st.columns(2)

    with col_mm1:
        mm_fv   = st.number_input("Face value (FV)", value=100000.0, key="mm_fv")
        mm_disc = st.number_input("Discount rate (DR)", value=0.03, format="%.4f", key="mm_disc")
        mm_days = st.number_input("Days to maturity", value=90.0, key="mm_days")
        mm_day_count = st.number_input("Days in year (money mkt)", value=360.0, key="mm_year")
        mm_price = mm_fv * (1 - mm_disc * (mm_days / mm_day_count))
        st.write(f"Price (discount basis) = {mm_price:,.2f}")

    with col_mm2:
        mm_price2 = st.number_input("Price (for add-on)", value=mm_price, key="mm_price2")
        mm_addon_days = st.number_input("Days to maturity (add-on)", value=mm_days, key="mm_addon_days")
        mm_year2 = st.number_input("Days in year (add-on)", value=365.0, key="mm_year2")
        if mm_price2 != 0:
            add_on_rate = (mm_fv - mm_price2) / mm_price2 * (mm_year2 / mm_addon_days)
        else:
            add_on_rate = 0.0
        st.write(f"Add-on yield (AOR) = {add_on_rate:.4f}  ({add_on_rate*100:.2f}%)")

    st.markdown("---")

    # ---- Duration and Convexity (approximate, price-yield) ----
    st.subheader("Effective Duration & Convexity (Price-Yield Approximation)")
    col_dc1, col_dc2 = st.columns(2)

    with col_dc1:
        p0 = st.number_input("Current full price P0", value=fi_price, key="dur_p0")
        p_up = st.number_input("Price if yield ↑ (P−)", value=fi_price * 0.95, key="dur_p_minus")
        p_down = st.number_input("Price if yield ↓ (P+)", value=fi_price * 1.05, key="dur_p_plus")
        delta_y = st.number_input("Change in yield (Δy, decimal)", value=0.01, format="%.4f", key="dur_dy")

    with col_dc2:
        if delta_y != 0 and p0 != 0:
            eff_dur = (p_down - p_up) / (2 * p0 * delta_y)
            eff_conv = (p_down + p_up - 2 * p0) / (p0 * (delta_y ** 2))
        else:
            eff_dur = 0.0
            eff_conv = 0.0
        st.write(f"Effective duration ≈ {eff_dur:.4f}")
        st.write(f"Effective convexity ≈ {eff_conv:.4f}")

    st.markdown("---")

    # ---- Price change using duration + convexity ----
    st.subheader("Price Change (Duration + Convexity)")
    col_pc1, col_pc2 = st.columns(2)

    with col_pc1:
        pc_p0 = st.number_input("Initial full price", value=fi_price, key="pc_p0")
        pc_dur = st.number_input("Effective/Modified duration", value=eff_dur, key="pc_dur")
        pc_conv = st.number_input("Effective convexity", value=eff_conv, key="pc_conv")
        pc_dy = st.number_input("Yield change Δy (decimal)", value=-0.01, format="%.4f", key="pc_dy")

    with col_pc2:
        # %ΔP ≈ (−Dur × Δy) + (0.5 × Conv × (Δy)^2)
        pct_delta_p = (-pc_dur * pc_dy) + (0.5 * pc_conv * (pc_dy ** 2))
        new_price = pc_p0 * (1 + pct_delta_p)
        st.write(f"Approx. %ΔPrice = {pct_delta_p*100:.3f}%")
        st.write(f"Estimated new price = {new_price:,.2f}")
# ================= EQUITY TAB =================
with cfa_tab_eq:
    st.markdown("### 📈 Equity Valuation Calculators")

    # ---- Gordon Growth (Dividend Discount Model) ----
    st.subheader("Dividend Discount Model (Constant Growth)")
    col_ddm1, col_ddm2 = st.columns(2)

    with col_ddm1:
        eq_d1 = st.number_input("Next dividend D₁", value=2.0, key="eq_d1")
        eq_re = st.number_input("Required return r", value=0.08, format="%.4f", key="eq_re")
        eq_g  = st.number_input("Growth rate g", value=0.03, format="%.4f", key="eq_g")

    with col_ddm2:
        if eq_re > eq_g:
            ddm_value = eq_d1 / (eq_re - eq_g)
        else:
            ddm_value = 0.0
        st.write(f"Intrinsic value (P₀) = {ddm_value:,.2f}")
        st.caption("Formula: P₀ = D₁ / (r − g)")

    st.markdown("---")

    # ---- Multi-stage DDM (2-stage) ----
    st.subheader("Two-Stage Dividend Discount Model")
    col_2ddm1, col_2ddm2 = st.columns(2)

    with col_2ddm1:
        ms_d0 = st.number_input("Current dividend D₀", value=1.50, key="ms_d0")
        ms_g1 = st.number_input("High growth rate g₁ (years 1–N)", value=0.10, format="%.4f", key="ms_g1")
        ms_g2 = st.number_input("Stable growth rate g₂ (after N)", value=0.04, format="%.4f", key="ms_g2")
        ms_re = st.number_input("Required return r", value=0.09, format="%.4f", key="ms_re")
        ms_n  = st.number_input("High growth period N (years)", value=5, step=1, min_value=1, key="ms_n")

    with col_2ddm2:
        dividends = []
        pv_dividends = 0.0
        for t in range(1, ms_n + 1):
            dt = ms_d0 * ((1 + ms_g1) ** t)
            dividends.append(dt)
            pv_dividends += dt / ((1 + ms_re) ** t)

        d_n1 = ms_d0 * ((1 + ms_g1) ** ms_n) * (1 + ms_g2)
        if ms_re > ms_g2:
            terminal_value = d_n1 / (ms_re - ms_g2)
            pv_terminal = terminal_value / ((1 + ms_re) ** ms_n)
        else:
            terminal_value = 0.0
            pv_terminal = 0.0

        ms_p0 = pv_dividends + pv_terminal
        st.write(f"Intrinsic value (P₀) ≈ {ms_p0:,.2f}")
        st.caption("Sum PV of high-growth dividends + PV of terminal value at year N.")

    st.markdown("---")

    # ---- Free Cash Flow to Equity (FCFE) Model ----
    st.subheader("Free Cash Flow to Equity (FCFE) Valuation")
    col_fcfe1, col_fcfe2 = st.columns(2)

    with col_fcfe1:
        fcfe_1 = st.number_input("Next year FCFE₁", value=5.0, key="fcfe_1")
        fcfe_re = st.number_input("Required return to equity r", value=0.10, format="%.4f", key="fcfe_re")
        fcfe_g  = st.number_input("Long-term growth rate g", value=0.04, format="%.4f", key="fcfe_g")

    with col_fcfe2:
        if fcfe_re > fcfe_g:
            fcfe_p0 = fcfe_1 / (fcfe_re - fcfe_g)
        else:
            fcfe_p0 = 0.0
        st.write(f"Equity value (per share) = {fcfe_p0:,.2f}")
        st.caption("Constant-growth FCFE: P₀ = FCFE₁ / (r − g)")

    st.markdown("---")

    # ---- P/E based valuation ----
    st.subheader("P/E-Based Valuation")
    col_pe1, col_pe2 = st.columns(2)

    with col_pe1:
        pe_eps = st.number_input("Expected EPS₁", value=4.0, key="pe_eps")
        pe_justified = st.number_input("Justified P/E or target P/E", value=15.0, key="pe_justified")

    with col_pe2:
        pe_price = pe_eps * pe_justified
        st.write(f"Value from P/E = {pe_price:,.2f}")
        st.caption("P₀ = (Justified P/E) × EPS₁")

    st.markdown("---")

    # ---- Justified leading and trailing P/E (Gordon) ----
    st.subheader("Justified P/E (Gordon Growth)")
    col_jpe1, col_jpe2 = st.columns(2)

    with col_jpe1:
        jpe_payout = st.number_input("Dividend payout ratio (D₁ / E₁)", value=0.40, format="%.4f", key="jpe_payout")
        jpe_re = st.number_input("Required return r", value=0.09, format="%.4f", key="jpe_re")
        jpe_g  = st.number_input("Growth rate g", value=0.04, format="%.4f", key="jpe_g")

    with col_jpe2:
        if jpe_re > jpe_g:
            justified_leading_pe = jpe_payout / (jpe_re - jpe_g)
        else:
            justified_leading_pe = 0.0
        st.write(f"Justified leading P/E = {justified_leading_pe:.2f}")
        st.caption("Leading P/E = payout ratio / (r − g)")

    st.markdown("---")

    # ---- Residual Income Model ----
    st.subheader("Residual Income Valuation (Simple Single-stage)")
    col_ri1, col_ri2 = st.columns(2)

    with col_ri1:
        ri_b0 = st.number_input("Current book value per share B₀", value=20.0, key="ri_b0")
        ri_roe = st.number_input("Return on equity ROE", value=0.15, format="%.4f", key="ri_roe")
        ri_re  = st.number_input("Required return r", value=0.10, format="%.4f", key="ri_re")
        ri_growth = st.number_input("Growth rate of residual income g", value=0.03, format="%.4f", key="ri_growth")

    with col_ri2:
        # Residual income in year 1: RI₁ = (ROE − r) × B₀
        ri1 = (ri_roe - ri_re) * ri_b0
        if ri_re > ri_growth:
            pv_ri_stream = ri1 / (ri_re - ri_growth)
        else:
            pv_ri_stream = 0.0
        ri_p0 = ri_b0 + pv_ri_stream
        st.write(f"Intrinsic value (P₀) ≈ {ri_p0:,.2f}")
        st.caption("P₀ = B₀ + PV of expected residual income stream.")

    st.markdown("---")

    # ---- Equity return metrics: Dividend yield, capital gains yield, total return ----
    st.subheader("Dividend Yield, Capital Gains Yield, Total Return")
    col_ret1, col_ret2, col_ret3 = st.columns(3)

    with col_ret1:
        er_p0 = st.number_input("Initial price P₀", value=50.0, key="er_p0")
        er_p1 = st.number_input("Ending price P₁", value=55.0, key="er_p1")
        er_d1 = st.number_input("Dividend D₁", value=2.0, key="er_d1")

    with col_ret2:
        if er_p0 != 0:
            div_yield = er_d1 / er_p0
            cap_gain_yield = (er_p1 - er_p0) / er_p0
            total_return = (er_p1 - er_p0 + er_d1) / er_p0
        else:
            div_yield = cap_gain_yield = total_return = 0.0

    with col_ret3:
        st.write(f"Dividend yield = {div_yield:.4f}  ({div_yield*100:.2f}%)")
        st.write(f"Capital gains yield = {cap_gain_yield:.4f}  ({cap_gain_yield*100:.2f}%)")
        st.write(f"Total return = {total_return:.4f}  ({total_return*100:.2f}%)")
# ================= PORTFOLIO MANAGEMENT TAB =================
with cfa_tab_pm:
    st.markdown("### 📚 Portfolio Management Calculators")

    # ---- 2-Asset Portfolio: Expected Return & Risk ----
    st.subheader("Two-Asset Portfolio: E[R_p] and σ_p")

    col_2p1, col_2p2, col_2p3 = st.columns(3)

    with col_2p1:
        w1 = st.number_input("Weight w₁", value=0.6, format="%.3f", key="pm_w1")
        w2 = st.number_input("Weight w₂", value=0.4, format="%.3f", key="pm_w2")
        r1 = st.number_input("Expected return R₁", value=0.08, format="%.4f", key="pm_r1")
        r2 = st.number_input("Expected return R₂", value=0.12, format="%.4f", key="pm_r2")

    with col_2p2:
        s1 = st.number_input("Std dev σ₁", value=0.15, format="%.4f", key="pm_s1")
        s2 = st.number_input("Std dev σ₂", value=0.20, format="%.4f", key="pm_s2")
        rho12 = st.number_input("Correlation ρ₁₂", value=0.30, format="%.4f", key="pm_rho12")

    with col_2p3:
        er_p = w1 * r1 + w2 * r2
        var_p = (w1**2)*(s1**2) + (w2**2)*(s2**2) + 2*w1*w2*s1*s2*rho12
        sd_p = var_p**0.5 if var_p >= 0 else 0.0
        st.write(f"E[R_p] = {er_p:.4f}  ({er_p*100:.2f}%)")
        st.write(f"σ_p = {sd_p:.4f}  ({sd_p*100:.2f}%)")
        st.caption("E[R_p] = w₁R₁ + w₂R₂;  σ_p² = w₁²σ₁² + w₂²σ₂² + 2w₁w₂σ₁σ₂ρ₁₂")

    st.markdown("---")

    # ---- N-Asset Portfolio Expected Return ----
    st.subheader("N-Asset Portfolio Expected Return (Weights & Returns)")

    pm_list_str = st.text_input(
        "Enter weights and returns as pairs (w,R) in %; ex: 60,8; 40,12",
        value="60,8; 40,12",
        key="pm_nasset_str"
    )

    try:
        # Parse "60,8; 40,12" -> [(0.60, 0.08), (0.40, 0.12)]
        pairs = []
        for part in pm_list_str.split(";"):
            part = part.strip()
            if not part:
                continue
            w_str, r_str = part.split(",")
            w_val = float(w_str.strip())/100
            r_val = float(r_str.strip())/100
            pairs.append((w_val, r_val))

        total_w = sum(w for w, _ in pairs)
        er_n = sum(w * r for w, r in pairs)
        st.write(f"Sum of weights = {total_w:.4f}")
        st.write(f"E[R_p] = {er_n:.4f}  ({er_n*100:.2f}%)")
    except Exception:
        st.write("Check format. Example: 60,8; 40,12")

    st.markdown("---")

    # ---- CAPM: Required / Expected Return ----
    st.subheader("CAPM: Required Return")

    col_capm1, col_capm2 = st.columns(2)

    with col_capm1:
        capm_rf = st.number_input("Risk-free rate R_f", value=0.03, format="%.4f", key="capm_rf")
        capm_rm = st.number_input("Market return E[R_m]", value=0.09, format="%.4f", key="capm_rm")
        capm_beta = st.number_input("Asset beta β", value=1.2, format="%.4f", key="capm_beta")

    with col_capm2:
        capm_er = capm_rf + capm_beta * (capm_rm - capm_rf)
        st.write(f"E[R_i] (CAPM) = {capm_er:.4f}  ({capm_er*100:.2f}%)")
        st.caption("E[R_i] = R_f + β_i [E(R_m) − R_f]")

    st.markdown("---")

    # ---- Portfolio Beta ----
    st.subheader("Portfolio Beta (Weighted Average)")

    col_pb1, col_pb2 = st.columns(2)

    with col_pb1:
        beta_str = st.text_input(
            "Enter weights and betas (w,β); ex: 50,1.1; 30,0.8; 20,1.4",
            value="50,1.1; 30,0.8; 20,1.4",
            key="pm_beta_str"
        )

    with col_pb2:
        try:
            pairs_b = []
            for part in beta_str.split(";"):
                part = part.strip()
                if not part:
                    continue
                w_str, b_str = part.split(",")
                w_val = float(w_str.strip())/100
                b_val = float(b_str.strip())
                pairs_b.append((w_val, b_val))

            beta_p = sum(w * b for w, b in pairs_b)
            w_sum = sum(w for w, _ in pairs_b)
            st.write(f"Sum of weights = {w_sum:.4f}")
            st.write(f"Portfolio beta β_p = {beta_p:.4f}")
        except Exception:
            st.write("Check format. Example: 50,1.1; 30,0.8; 20,1.4")

    st.markdown("---")

    # ---- Minimum-Variance Weight (2-Asset) ----
    st.subheader("Minimum-Variance Portfolio (Two Assets)")

    col_mvp1, col_mvp2 = st.columns(2)

    with col_mvp1:
        mvp_s1 = st.number_input("σ₁ (asset 1)", value=0.15, format="%.4f", key="mvp_s1")
        mvp_s2 = st.number_input("σ₂ (asset 2)", value=0.20, format="%.4f", key="mvp_s2")
        mvp_rho = st.number_input("Correlation ρ₁₂", value=0.30, format="%.4f", key="mvp_rho")

    with col_mvp2:
        denom = (mvp_s1**2 + mvp_s2**2 - 2*mvp_s1*mvp_s2*mvp_rho)
        if denom != 0:
            w1_mvp = (mvp_s2**2 - mvp_rho*mvp_s1*mvp_s2) / denom
            w2_mvp = 1 - w1_mvp
        else:
            w1_mvp = w2_mvp = 0.0

        st.write(f"w₁(min-var) = {w1_mvp:.4f}")
        st.write(f"w₂(min-var) = {w2_mvp:.4f}")
        st.caption("Formula from Markowitz minimum-variance two-asset case.")

    st.markdown("---")

    # ---- Sharpe Ratio for Portfolio ----
    st.subheader("Portfolio Sharpe Ratio")

    col_shp1, col_shp2 = st.columns(2)

    with col_shp1:
        shp_rp = st.number_input("Portfolio expected return E[R_p]", value=er_p, format="%.4f", key="shp_rp")
        shp_rf = st.number_input("Risk-free rate R_f (Sharpe)", value=capm_rf, format="%.4f", key="shp_rf")
        shp_sd = st.number_input("Portfolio σ_p", value=sd_p, format="%.4f", key="shp_sd")

    with col_shp2:
        shp_ratio = (shp_rp - shp_rf) / shp_sd if shp_sd != 0 else 0.0
        st.write(f"Sharpe ratio = {shp_ratio:.4f}")
        st.caption("Sharpe = (E[R_p] − R_f) / σ_p")
# ================= DERIVATIVES TAB =================
with cfa_tab_der:
    st.markdown("### ⚙️ Derivatives Pricing & Payoff Calculators")

    # ---------- FORWARDS & FUTURES ----------
    st.subheader("Forward/Futures Pricing (No Income, Known Income, Yield)")

    col_fwd1, col_fwd2 = st.columns(2)

    with col_fwd1:
        f_S0   = st.number_input("Spot price S₀", value=100.0, key="f_S0")
        f_r    = st.number_input("Risk-free rate r (annual, dec)", value=0.05, format="%.4f", key="f_r")
        f_T    = st.number_input("Time to maturity T (years)", value=1.0, format="%.4f", key="f_T")
        f_inc  = st.number_input("Present value of known income (PV(dividends))", value=0.0, key="f_inc")
        f_yield= st.number_input("Dividend/Convenience yield q (if using yield model)", value=0.0, format="%.4f", key="f_q")

    with col_fwd2:
        # Discrete compounding, no income: F0 = S0 × (1+r)^T
        F0_no_income = f_S0 * ((1 + f_r) ** f_T)

        # Discrete compounding, known PV income: F0 = (S0 − PV(inc)) × (1+r)^T
        F0_pv_income = (f_S0 - f_inc) * ((1 + f_r) ** f_T)

        # Continuous yield model (approx): F0 = S0 × e^{(r−q)T}
        import math
        F0_yield = f_S0 * math.exp((f_r - f_yield) * f_T)

        st.write(f"F₀ (no income) ≈ {F0_no_income:,.4f}")
        st.write(f"F₀ (PV income) ≈ {F0_pv_income:,.4f}")
        st.write(f"F₀ (yield model) ≈ {F0_yield:,.4f}")
        st.caption("Common forms: F₀ = S₀(1+r)^T ; F₀ = (S₀ − PV(inc))(1+r)^T ; F₀ = S₀ e^{(r−q)T}")

    st.markdown("---")

    st.subheader("Forward/Futures Payoff & Profit at Expiration")

    col_fpay1, col_fpay2 = st.columns(2)

    with col_fpay1:
        f_ST   = st.number_input("Underlying price at maturity S_T", value=110.0, key="f_ST")
        f_F0   = st.number_input("Contract forward price F₀", value=F0_no_income, key="f_F0")
        f_long = st.checkbox("Show long position payoff", value=True, key="f_long")
        f_short= st.checkbox("Show short position payoff", value=True, key="f_short")

    with col_fpay2:
        if f_long:
            long_payoff = f_ST - f_F0
            st.write(f"Long forward payoff = S_T − F₀ = {long_payoff:,.4f}")
        if f_short:
            short_payoff = f_F0 - f_ST
            st.write(f"Short forward payoff = F₀ − S_T = {short_payoff:,.4f}")
        st.caption("Ignoring financing; profit ≈ payoff at expiration for forwards/futures.")

    st.markdown("---")

    # ---------- OPTIONS PAYOFF (CALL & PUT) ----------
    st.subheader("European Call & Put Payoff / Profit")

    col_opt1, col_opt2, col_opt3 = st.columns(3)

    with col_opt1:
        o_ST  = st.number_input("Underlying price at expiration S_T", value=100.0, key="o_ST")
        o_X   = st.number_input("Strike price X", value=95.0, key="o_X")
        o_c0  = st.number_input("Call premium c₀", value=4.0, key="o_c0")
        o_p0  = st.number_input("Put premium p₀", value=3.0, key="o_p0")

    with col_opt2:
        # Payoffs
        call_payoff = max(0.0, o_ST - o_X)
        put_payoff  = max(0.0, o_X - o_ST)
        st.write(f"Call payoff = max(0, S_T − X) = {call_payoff:.4f}")
        st.write(f"Put payoff = max(0, X − S_T) = {put_payoff:.4f}")

    with col_opt3:
        # Profit = payoff − premium
        call_profit_long  = call_payoff - o_c0
        put_profit_long   = put_payoff - o_p0
        call_profit_short = -call_profit_long
        put_profit_short  = -put_profit_long

        st.write(f"Long call profit = {call_profit_long:.4f}")
        st.write(f"Short call profit = {call_profit_short:.4f}")
        st.write(f"Long put profit = {put_profit_long:.4f}")
        st.write(f"Short put profit = {put_profit_short:.4f}")
        st.caption("European options, profit at expiration ignoring time value of money.")

    st.markdown("---")

    # ---------- BASIC STRATEGIES ----------
    st.subheader("Option Strategies: Protective Put & Covered Call")

    col_str1, col_str2 = st.columns(2)

    with col_str1:
        s_S0  = st.number_input("Stock purchase price S₀", value=100.0, key="s_S0")
        s_ST  = st.number_input("Stock price at expiration S_T (strat)", value=o_ST, key="s_ST")
        s_X   = st.number_input("Option strike X (strat)", value=o_X, key="s_X")
        s_c0  = st.number_input("Call premium c₀ (strat)", value=o_c0, key="s_c0")
        s_p0  = st.number_input("Put premium p₀ (strat)", value=o_p0, key="s_p0")

    with col_str2:
        # Protective put: long stock + long put
        stock_payoff = s_ST
        put_payoff_strat = max(0.0, s_X - s_ST)
        prot_put_profit = stock_payoff + put_payoff_strat - s_S0 - s_p0

        # Covered call: long stock + short call
        call_payoff_strat = max(0.0, s_ST - s_X)
        cov_call_profit = stock_payoff - call_payoff_strat - s_S0 + s_c0

        st.write(f"Protective put profit = {prot_put_profit:.4f}")
        st.write(f"Covered call profit = {cov_call_profit:.4f}")
        st.caption("Both evaluated at expiration; ignores dividends and financing.")

    st.markdown("---")

    # ---------- PUT–CALL PARITY ----------
    st.subheader("Put–Call Parity (No Dividends)")

    col_pcp1, col_pcp2 = st.columns(2)

    with col_pcp1:
        pcp_S0 = st.number_input("Spot price S₀ (parity)", value=52.0, key="pcp_S0")
        pcp_X  = st.number_input("Strike price X", value=50.0, key="pcp_X")
        pcp_r  = st.number_input("Risk-free rate r", value=0.045, format="%.4f", key="pcp_r")
        pcp_T  = st.number_input("Time to expiry T (years)", value=0.25, format="%.4f", key="pcp_T")
        pcp_p0 = st.number_input("Put price p₀", value=3.80, key="pcp_p0")

    with col_pcp2:
        # C0 = S0 + p0 − X / (1+r)^T  (discrete compounding)
        disc_factor = (1 + pcp_r) ** pcp_T
        pcp_c0_theoretical = pcp_S0 + pcp_p0 - pcp_X / disc_factor
        st.write(f"Theoretical call price c₀ (from parity) ≈ {pcp_c0_theoretical:.4f}")
        st.caption("c₀ + X/(1+r)^T = p₀ + S₀   ⇒   c₀ = S₀ + p₀ − X/(1+r)^T")

    st.markdown("---")

    # ---------- FORWARD ON STOCK VIA PREPAID FORWARD ----------
    st.subheader("Prepaid Forward & Forward on Stock (Dividends)")

    col_pf1, col_pf2 = st.columns(2)

    with col_pf1:
        pf_S0 = st.number_input("Spot price S₀ (stock)", value=100.0, key="pf_S0")
        pf_div_pv = st.number_input("PV of all dividends to T", value=0.0, key="pf_div_pv")
        pf_r  = st.number_input("Risk-free rate r (annual)", value=0.05, format="%.4f", key="pf_r")
        pf_T  = st.number_input("Time to maturity T (years)", value=1.0, format="%.4f", key="pf_T")

    with col_pf2:
        # Prepaid forward: FP0,T = S0 − PV(dividends)
        FP0T = pf_S0 - pf_div_pv

        # Forward price: F0,T = FP0,T × (1+r)^T
        F0T = FP0T * ((1 + pf_r) ** pf_T)

        st.write(f"Prepaid forward price FP₀,T = {FP0T:,.4f}")
        st.write(f"Forward price F₀,T = {F0T:,.4f}")
        st.caption("FP₀,T = S₀ − PV(div);  F₀,T = FP₀,T(1+r)^T.")

#  TAB 5 â€” CFO CALCULATORS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with T_CFO:
    st.markdown(
        f"<h4 style='color:{CF['primary']};font-family:monospace'>ðŸ’¼ CFO FINANCIAL TOOLKIT</h4>",
        unsafe_allow_html=True
    )
    cfo1, cfo2 = st.tabs(["ðŸ“Š DCF VALUATION", "âš–ï¸ WACC CALCULATOR"])

    with cfo1:
        st.markdown(
            f"<div style='background:{CF['dark']};border:1px solid {CF['border']};"
            "border-radius:8px;padding:14px;margin-bottom:14px'>"
            f"<div style='color:{CF['primary']};font-family:monospace;font-weight:bold;margin-bottom:4px'>"
            "ðŸ“Š DISCOUNTED CASH FLOW (DCF) VALUATION MODEL</div>"
            "<div style='color:#555;font-family:monospace;font-size:11px'>"
            "Projects FCFs and discounts at WACC Â· terminal value Â· equity intrinsic value Â· sensitivity table"
            "</div></div>",
            unsafe_allow_html=True
        )
        d1a, d1b, d1c = st.columns(3)
        with d1a:
            dcf_fcf    = st.number_input("ðŸ’µ Base FCF ($M)",      value=500.0, step=50.0, key="dcf_fcf",    format="%.1f")
            dcf_gr1    = st.slider("ðŸ“ˆ Growth Yr 1-5 (%)",        0.0, 50.0, 15.0, 0.5, key="dcf_gr1",   format="%.1f")
            dcf_gr2    = st.slider("ðŸ“ˆ Growth Yr 6-10 (%)",       0.0, 30.0, 8.0,  0.5, key="dcf_gr2",   format="%.1f")
        with d1b:
            dcf_tgr    = st.slider("â™¾ï¸ Terminal Growth (%)",      0.0, 5.0,  2.5,  0.1, key="dcf_tgr",   format="%.1f")
            dcf_wacc   = st.slider("âš–ï¸ WACC (%)",                 1.0, 25.0, 10.0, 0.25,key="dcf_wacc",  format="%.2f")
            dcf_debt   = st.number_input("ðŸ¦ Net Debt ($M)",      value=200.0, step=10.0, key="dcf_debt", format="%.1f")
        with d1c:
            dcf_shares = st.number_input("ðŸ“‹ Shares Out (M)",     value=100.0, step=5.0,  key="dcf_sh",   format="%.1f")
            dcf_cash   = st.number_input("ðŸ’° Cash ($M)",          value=150.0, step=10.0, key="dcf_cash", format="%.1f")
            dcf_mos    = st.slider("ðŸŽ¯ Margin of Safety (%)",     0.0, 40.0, 20.0, 5.0, key="dcf_mos",  format="%.0f")

        st.markdown("<div class='cf-btn'>", unsafe_allow_html=True)
        run_dcf = st.button("ðŸ’¼ Run DCF", key="run_dcf")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_dcf:
            wacc_d = dcf_wacc/100; tgr_d = dcf_tgr/100
            gr1_d  = dcf_gr1/100;  gr2_d = dcf_gr2/100
            fcfs = []; fcf_v = dcf_fcf
            for yr in range(1, 11):
                fcf_v *= (1+gr1_d) if yr<=5 else (1+gr2_d)
                pv = fcf_v/(1+wacc_d)**yr
                fcfs.append((yr, fcf_v, pv))
            tv    = fcfs[-1][1]*(1+tgr_d)/(wacc_d-tgr_d) if wacc_d > tgr_d else 0
            pv_tv = tv/(1+wacc_d)**10
            pv_f  = sum(r[2] for r in fcfs)
            ev    = pv_f + pv_tv
            eq    = ev - dcf_debt + dcf_cash
            intr  = eq/dcf_shares if dcf_shares > 0 else 0
            mos_p = intr*(1-dcf_mos/100)

            v1,v2,v3,v4,v5 = st.columns(5)
            def vbox(col, label, val, sub="", color=None):
                c = color or CF["primary"]
                col.markdown(
                    f"<div style='background:{CF['mid']};border:1px solid {CF['border']};"
                    "border-radius:6px;padding:12px;text-align:center;font-family:monospace'>"
                    f"<div style='color:#555;font-size:10px'>{label}</div>"
                    f"<div style='color:{c};font-size:20px;font-weight:bold'>{val}</div>"
                    f"{'<div style=color:#444;font-size:10px>' + sub + '</div>' if sub else ''}"
                    "</div>",
                    unsafe_allow_html=True
                )
            vbox(v1,"PV of FCFs",    f"${pv_f:,.0f}M", "10-year DCFs")
            vbox(v2,"PV Terminal",   f"${pv_tv:,.0f}M",f"TV={tv:,.0f}M")
            vbox(v3,"Enterprise Val",f"${ev:,.0f}M",   "EV=PV(FCF)+PV(TV)")
            vbox(v4,"Intrinsic /sh", f"${intr:.2f}",   "per share",     CF["light"])
            vbox(v5,"MoS Price",     f"${mos_p:.2f}",  f"{dcf_mos:.0f}% discount", "#00FF41")

            yrs  = [f"Y{r[0]}" for r in fcfs] + ["Terminal","Total EV"]
            vals = [r[2]         for r in fcfs] + [pv_tv, ev]
            bcs  = [CF["primary"]]*10 + ["#F59E0B","#10B981"]
            fig  = go.Figure(go.Bar(x=yrs, y=vals, marker_color=bcs, opacity=0.85,
                text=[f"${v:,.0f}M" for v in vals], textposition="auto",
                textfont=dict(family="Courier New",size=9)))
            fig.update_layout(**PLOT_CFG, height=360,
                title=dict(text="ðŸ’¼ DCF Cash Flow Breakdown",
                           font=dict(family="Courier New",size=13,color=CF["primary"]),x=0))
            fig.update_xaxes(gridcolor="#111",title_text="Period")
            fig.update_yaxes(gridcolor="#111",title_text="PV ($M)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"<div style='color:{CF['primary']};font-family:monospace;font-weight:bold;"
                "margin:10px 0 6px'>ðŸ“Š SENSITIVITY â€” Intrinsic Value vs WACC & Terminal Growth</div>",
                unsafe_allow_html=True
            )
            waccs = [wacc_d-0.02, wacc_d-0.01, wacc_d, wacc_d+0.01, wacc_d+0.02]
            tgrs  = [tgr_d-0.01,  tgr_d,       tgr_d+0.01, tgr_d+0.02]
            rows_s = {}
            for tg in tgrs:
                row = {}
                for wc in waccs:
                    if wc <= tg: row[f"WACC {wc:.1%}"] = "âˆž"
                    else:
                        fcf2 = dcf_fcf; pv2 = 0
                        for yr in range(1,11):
                            fcf2 *= (1+gr1_d) if yr<=5 else (1+gr2_d)
                            pv2  += fcf2/(1+wc)**yr
                        tv2  = fcf2*(1+tg)/(wc-tg)
                        ptv2 = tv2/(1+wc)**10
                        eq2  = (pv2+ptv2) - dcf_debt + dcf_cash
                        row[f"WACC {wc:.1%}"] = f"${eq2/dcf_shares:.2f}"
                rows_s[f"TGR {tg:.1%}"] = row
            st.dataframe(pd.DataFrame(rows_s).T, use_container_width=True)

    with cfo2:
        st.markdown(
            f"<div style='background:{CF['dark']};border:1px solid {CF['border']};"
            "border-radius:8px;padding:14px;margin-bottom:14px'>"
            f"<div style='color:{CF['primary']};font-family:monospace;font-weight:bold;margin-bottom:4px'>"
            "âš–ï¸ WACC â€” WEIGHTED AVERAGE COST OF CAPITAL</div>"
            "<div style='color:#555;font-family:monospace;font-size:11px'>"
            "WACC = (E/V)Ã—Re + (D/V)Ã—RdÃ—(1âˆ’Tc)"
            "</div></div>",
            unsafe_allow_html=True
        )
        w1, w2, w3 = st.columns(3)
        with w1:
            wacc_e    = st.number_input("ðŸ’¹ Equity ($M)",     value=800.0, step=50.0, key="wacc_e",    format="%.1f")
            wacc_d_v  = st.number_input("ðŸ¦ Debt ($M)",       value=200.0, step=50.0, key="wacc_d_v",  format="%.1f")
            wacc_rf   = st.slider("ðŸ›ï¸ Risk-Free Rate (%)",    0.0,10.0, 4.5, 0.1, key="wacc_rf",  format="%.1f")
        with w2:
            wacc_beta = st.slider("ðŸ“Š Beta",                  0.1, 3.0, 1.2, 0.05,key="wacc_beta",format="%.2f")
            wacc_erp  = st.slider("ðŸ“ˆ Equity Risk Premium (%)",3.0,10.0, 5.5, 0.1, key="wacc_erp", format="%.1f")
            wacc_rd   = st.slider("ðŸ’³ Cost of Debt (%)",      1.0,15.0, 5.0, 0.25,key="wacc_rd",  format="%.2f")
        with w3:
            wacc_tc   = st.slider("ðŸ›ï¸ Tax Rate (%)",          0.0,40.0,21.0, 0.5, key="wacc_tc",  format="%.1f")

        st.markdown("<div class='cf-btn'>", unsafe_allow_html=True)
        run_wacc = st.button("âš–ï¸ Calculate WACC", key="run_wacc")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_wacc:
            V    = wacc_e + wacc_d_v
            E_w  = wacc_e / V; D_w = wacc_d_v / V
            Re   = wacc_rf/100 + wacc_beta*(wacc_erp/100)
            Rd   = wacc_rd/100; Tc = wacc_tc/100
            WACC = E_w*Re + D_w*Rd*(1-Tc)

            wc1,wc2,wc3,wc4,wc5 = st.columns(5)
            def wbox(col, label, val, sub="", color=None):
                c = color or CF["primary"]
                col.markdown(
                    f"<div style='background:{CF['mid']};border:1px solid {CF['border']};"
                    "border-radius:6px;padding:12px;text-align:center;font-family:monospace'>"
                    f"<div style='color:#555;font-size:10px'>{label}</div>"
                    f"<div style='color:{c};font-size:20px;font-weight:bold'>{val}</div>"
                    f"{'<div style=color:#444;font-size:10px>' + sub + '</div>' if sub else ''}"
                    "</div>",
                    unsafe_allow_html=True
                )
            wbox(wc1,"WACC",          f"{WACC:.2%}",      "Blended rate",      CF["light"])
            wbox(wc2,"Cost of Equity",f"{Re:.2%}",         "CAPM: Rf+Î²Ã—ERP")
            wbox(wc3,"Cost of Debt",  f"{Rd*(1-Tc):.2%}", "After-tax")
            wbox(wc4,"Equity Weight", f"{E_w:.1%}",        f"${wacc_e:,.0f}M")
            wbox(wc5,"Debt Weight",   f"{D_w:.1%}",        f"${wacc_d_v:,.0f}M")

            fig = make_subplots(rows=1, cols=2, specs=[[{"type":"pie"},{"type":"scatter"}]])
            fig.add_trace(go.Pie(
                labels=["Equity","Debt"], values=[wacc_e, wacc_d_v],
                marker_colors=[CF["primary"],"#06B6D4"], hole=0.45,
                textfont=dict(family="Courier New",size=11)
            ), row=1, col=1)
            betas_arr = np.linspace(0.5, 2.5, 60)
            wacc_arr  = [E_w*(wacc_rf/100 + b*(wacc_erp/100)) + D_w*Rd*(1-Tc) for b in betas_arr]
            fig.add_trace(go.Scatter(
                x=betas_arr, y=[w*100 for w in wacc_arr],
                line=dict(color=CF["primary"],width=2.5), name="WACC vs Î²"
            ), row=1, col=2)
            curr_wacc = E_w*(wacc_rf/100 + wacc_beta*(wacc_erp/100)) + D_w*Rd*(1-Tc)
            fig.add_trace(go.Scatter(
                x=[wacc_beta], y=[curr_wacc*100],
                mode="markers+text",
                marker=dict(color="#FFD700",size=12,symbol="diamond"),
                text=[f"Î²={wacc_beta}"], textposition="top center",
                textfont=dict(family="Courier New",color="#FFD700"),
                name=f"Î²={wacc_beta}"
            ), row=1, col=2)
            fig.update_layout(**PLOT_CFG, height=360,
                legend=dict(font=dict(family="Courier New",size=10),bgcolor="rgba(0,0,0,0)"))
            fig.update_xaxes(gridcolor="#111",title_text="Beta", row=1, col=2)
            fig.update_yaxes(gridcolor="#111",title_text="WACC (%)", row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  TAB 6 â€” QUANT LAB
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with T_QUANT:
    st.markdown(
        f"<h4 style='color:{QT['primary']};font-family:monospace'>ðŸ”¬ QUANT LAB â€” RISK & PORTFOLIO ANALYTICS</h4>",
        unsafe_allow_html=True
    )
    q1tab, q2tab = st.tabs(["ðŸ“‰ VaR & RISK METRICS", "ðŸ”— MARKOWITZ OPTIMIZER"])

    # â”€â”€ VaR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with q1tab:
        st.markdown(
            f"<div style='background:{QT['dark']};border:1px solid {QT['border']};"
            "border-radius:8px;padding:14px;margin-bottom:14px'>"
            f"<div style='color:{QT['primary']};font-family:monospace;font-weight:bold;margin-bottom:4px'>"
            "ðŸ“‰ VALUE AT RISK (VaR) + CVaR / ES â€” Risk Measurement Suite</div>"
            "<div style='color:#555;font-family:monospace;font-size:11px'>"
            "Parametric Â· Historical Â· Monte Carlo VaR Â· CVaR/ES Â· Sharpe Â· Sortino Â· Calmar"
            "</div></div>",
            unsafe_allow_html=True
        )
        vq1, vq2, vq3 = st.columns(3)
        with vq1:
            var_tk  = st.text_input("ðŸ“ Ticker", value="SPY", key="var_tk").upper().strip()
            var_per = st.selectbox("ðŸ“… Period", ["1y","2y","3y","5y"], index=1, key="var_per")
        with vq2:
            var_port = st.number_input("ðŸ’µ Portfolio ($)", value=1000000.0, step=50000.0, key="var_port", format="%.0f")
            var_conf = st.select_slider("ðŸŽ¯ Confidence Level", [0.90,0.95,0.99], value=0.95, key="var_conf")
        with vq3:
            var_hrz = st.select_slider("ðŸ“† Horizon (days)", [1,5,10,20,30,60], value=1, key="var_hrz")
            var_rf2 = st.slider("ðŸ›ï¸ Risk-Free Rate (%)", 0.0, 10.0, 4.5, 0.1, key="var_rf2", format="%.1f")

        st.markdown("<div class='qt-btn'>", unsafe_allow_html=True)
        run_var = st.button("ðŸ“‰ Calculate VaR & Risk", key="run_var")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_var:
            with st.spinner(f"Fetching {var_tk} data..."):
                df_v = fetch_ohlcv(var_tk, var_per)
            if not df_v.empty and len(df_v) > 30:
                rets = df_v["Close"].pct_change().dropna()
                mu_v = float(rets.mean()); sig_v = float(rets.std())
                z    = norm.ppf(1 - var_conf)
                var_p  = -(mu_v + z*sig_v)*math.sqrt(var_hrz)*var_port
                var_h  = -np.percentile(rets,(1-var_conf)*100)*math.sqrt(var_hrz)*var_port
                np.random.seed(42)
                mc_r   = np.random.normal(mu_v, sig_v, (10000, var_hrz))
                mc_pnl = (np.prod(1+mc_r, axis=1)-1)*var_port
                var_mc = -np.percentile(mc_pnl, (1-var_conf)*100)
                cutoff = np.percentile(rets, (1-var_conf)*100)
                cvar   = -rets[rets<=cutoff].mean()*math.sqrt(var_hrz)*var_port
                ann_r  = float(rets.mean()*252); ann_v = float(rets.std()*math.sqrt(252))
                sharpe = (ann_r - var_rf2/100)/ann_v if ann_v > 0 else 0
                neg_r  = rets[rets<0]
                sortino= (ann_r - var_rf2/100)/(float(neg_r.std())*math.sqrt(252)) if len(neg_r)>0 else 0
                mdd    = float(((df_v["Close"]/df_v["Close"].cummax())-1).min()*100)
                calmar = ann_r/abs(mdd/100) if mdd != 0 else 0

                rc1,rc2,rc3,rc4,rc5,rc6 = st.columns(6)
                def rbox(col, label, val, sub="", color=None):
                    c = color or QT["primary"]
                    col.markdown(
                        f"<div style='background:{QT['mid']};border:1px solid {QT['border']};"
                        "border-radius:6px;padding:10px;text-align:center;font-family:monospace'>"
                        f"<div style='color:#555;font-size:10px'>{label}</div>"
                        f"<div style='color:{c};font-size:18px;font-weight:bold'>{val}</div>"
                        f"{'<div style=color:#444;font-size:9px>' + sub + '</div>' if sub else ''}"
                        "</div>",
                        unsafe_allow_html=True
                    )
                rbox(rc1,"Param VaR",  f"${var_p:,.0f}", f"{var_conf:.0%}/{var_hrz}d","#FF4444")
                rbox(rc2,"Hist VaR",   f"${var_h:,.0f}", "hist sim",                  "#FF4444")
                rbox(rc3,"MC VaR",     f"${var_mc:,.0f}","10k paths",                 "#FF4444")
                rbox(rc4,"CVaR/ES",    f"${cvar:,.0f}",  "exp. shortfall",            "#EC4899")
                rbox(rc5,"Sharpe",     f"{sharpe:.3f}",  "annualized")
                rbox(rc6,"Sortino",    f"{sortino:.3f}", "downside only")

                rc7,rc8,rc9,rc10 = st.columns(4)
                rbox(rc7, "Ann Return",   f"{ann_r:.2%}", "", "#00FF41" if ann_r>=0 else "#FF4444")
                rbox(rc8, "Ann Vol",      f"{ann_v:.2%}")
                rbox(rc9, "Max Drawdown", f"{mdd:.1f}%", "peak-to-trough", "#FF4444")
                rbox(rc10,"Calmar",       f"{calmar:.3f}")

                fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                                    row_heights=[0.55,0.45], vertical_spacing=0.07)
                fig.add_trace(go.Scatter(
                    x=df_v.index, y=df_v["Close"],
                    line=dict(color=QT["primary"],width=1.5),
                    name=f"{var_tk} Price"
                ), row=1, col=1)
                svc = df_v["Close"].cummax()
                fig.add_trace(go.Scatter(
                    x=df_v.index, y=svc,
                    line=dict(color="#444",width=1, dash="dot"),
                    name="Rolling High"
                ), row=1, col=1)
                fig2_rets = rets.copy()
                fig.add_trace(go.Histogram(
                    x=fig2_rets, nbinsx=60,
                    marker_color=QT["primary"], opacity=0.75,
                    name="Daily Returns"
                ), row=2, col=1)
                thr = np.percentile(fig2_rets,(1-var_conf)*100)
                fig.add_vline(x=thr, line=dict(color="#FF4444",dash="dash",width=2),
                              annotation_text=f"VaR cutoff {thr:.2%}", annotation_font_color="#FF8888")
                fig.update_layout(**PLOT_CFG, height=480,
                    legend=dict(orientation="h", x=0, y=1.08,
                                font=dict(family="Courier New",size=10), bgcolor="rgba(0,0,0,0)"))
                fig.update_xaxes(gridcolor="#111", row=1,col=1); fig.update_yaxes(gridcolor="#111", row=1,col=1)
                fig.update_xaxes(gridcolor="#111", row=2,col=1, title_text="Daily Returns"); 
                fig.update_yaxes(gridcolor="#111", row=2,col=1, title_text="Frequency")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data for VaR calculation. Try a different ticker or longer period.")

    # â”€â”€ Markowitz â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with q2tab:
        st.markdown(
            f"<div style='background:{QT['dark']};border:1px solid {QT['border']};"
            "border-radius:8px;padding:14px;margin-bottom:14px'>"
            f"<div style='color:{QT['primary']};font-family:monospace;font-weight:bold;margin-bottom:4px'>"
            "ðŸ”— MEAN-VARIANCE PORTFOLIO OPTIMIZER â€” Efficient Frontier</div>"
            "<div style='color:#555;font-family:monospace;font-size:11px'>"
            "Upload tickers, estimate return/cov, compute min variance & max Sharpe portfolios, plot frontier"
            "</div></div>",
            unsafe_allow_html=True
        )

        mq1, mq2 = st.columns([3,2])
        with mq1:
            tickers_str = st.text_input(
                "ðŸ“‹ Tickers (comma-separated)", 
                value="SPY, QQQ, IWM, EFA, EEM, TLT",
                key="m_tks"
            )
            per_m = st.selectbox("ðŸ“… Lookback", ["1y","3y","5y"], index=1, key="m_per")
        with mq2:
            rf_m = st.slider("ðŸ›ï¸ Risk-Free Rate (%)", 0.0, 10.0, 4.5, 0.1, key="m_rf", format="%.1f")
            pts_m= st.slider("ðŸ“ˆ Frontier Points", 20, 200, 80, 10, key="m_pts")

        st.markdown("<div class='qt-btn'>", unsafe_allow_html=True)
        run_mpt = st.button("ðŸ”— Run Markowitz Optimization", key="run_mpt")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_mpt:
            tks = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
            if len(tks) < 2:
                st.warning("Please enter at least two tickers."); 
            else:
                with st.spinner("Downloading price history and computing stats..."):
                    data = {}
                    for tk in tks:
                        df = fetch_ohlcv(tk, per_m)
                        if df.empty or len(df)<60: 
                            continue
                        data[tk] = df["Close"]
                    if len(data) < 2:
                        st.warning("Not enough valid tickers with history. Try different symbols or a longer period.")
                    else:
                        prices = pd.DataFrame(data).dropna()
                        rets   = prices.pct_change().dropna()
                        mu     = rets.mean().values*252
                        cov    = rets.cov().values*252
                        rf     = rf_m/100

                        w_mv   = min_var_w(mu, cov)
                        mv_r, mv_v = port_stats(w_mv, mu, cov)
                        w_sh   = max_sharpe_w(mu, cov, rf)
                        sh_r, sh_v = port_stats(w_sh, mu, cov)
                        vols_f, rets_f = calc_frontier(mu, cov, n_pts=pts_m)

                        c1, c2, c3 = st.columns(3)
                        stat_box(c1, "Min-Var Return", f"{mv_r:.2%}", "#10B981", f"Vol {mv_v:.2%}")
                        stat_box(c2, "Max-Sharpe Return", f"{sh_r:.2%}", "#F59E0B", f"Vol {sh_v:.2%}")
                        sh_ratio = (sh_r - rf)/sh_v if sh_v>0 else 0
                        stat_box(c3, "Sharpe (Max)", f"{sh_ratio:.2f}", "#EC4899", f"Rf {rf_m:.2f}%")

                        fig = go.Figure()
                        for i, tk in enumerate(tks):
                            fig.add_trace(go.Scatter(
                                x=[np.sqrt(cov[i,i])], y=[mu[i]],
                                mode="markers", name=tk,
                                marker=dict(size=9,color="#888"),
                                text=[tk]
                            ))
                        fig.add_trace(go.Scatter(
                            x=vols_f, y=rets_f,
                            mode="lines", name="Efficient Frontier",
                            line=dict(color=QT["primary"],width=2.5)
                        ))
                        fig.add_trace(go.Scatter(
                            x=[mv_v], y=[mv_r],
                            mode="markers", name="Min Variance",
                            marker=dict(size=11,color="#10B981",symbol="diamond")
                        ))
                        fig.add_trace(go.Scatter(
                            x=[sh_v], y=[sh_r],
                            mode="markers", name="Max Sharpe",
                            marker=dict(size=11,color="#F59E0B",symbol="star")
                        ))
                        fig.update_layout(**PLOT_CFG, height=420,
                            title=dict(text="ðŸ”— Efficient Frontier & Key Portfolios",
                                       font=dict(family="Courier New",size=13,color=QT["primary"]),x=0),
                            legend=dict(orientation="h",x=0,y=1.05,
                                        font=dict(family="Courier New",size=10),bgcolor="rgba(0,0,0,0)"))
                        fig.update_xaxes(gridcolor="#111",title_text="Volatility (Ïƒ)")
                        fig.update_yaxes(gridcolor="#111",title_text="Expected Return")
                        st.plotly_chart(fig, use_container_width=True)

                        wm_df = pd.DataFrame({"Ticker": tks,
                                              "Min-Var W": w_mv,
                                              "Max-Sharpe W": w_sh})
                        wm_df["Min-Var W"]   = wm_df["Min-Var W"].map(lambda x: f"{x:.2%}")
                        wm_df["Max-Sharpe W"]= wm_df["Max-Sharpe W"].map(lambda x: f"{x:.2%}")
                        st.markdown("#### Optimal Weights")
                        st.dataframe(wm_df.set_index("Ticker"), use_container_width=True)

                        rets_p = rets@w_sh
                        cum_p  = (1+rets_p).cumprod()
                        cum_b  = (1+rets).cumprod()
                        fig2   = go.Figure()
                        for tk in tks:
                            fig2.add_trace(go.Scatter(
                                x=cum_b.index, y=cum_b[tk],
                                mode="lines", name=tk,
                                line=dict(width=1)
                            ))
                        fig2.add_trace(go.Scatter(
                            x=cum_p.index, y=cum_p,
                            mode="lines", name="Max Sharpe Portfolio",
                            line=dict(width=2.3,color=QT["primary"])
                        ))
                        fig2.update_layout(**PLOT_CFG, height=420,
                            title=dict(text="ðŸ“ˆ Growth of $1 â€” Max Sharpe vs Constituents",
                                       font=dict(family="Courier New",size=13,color=QT["primary"]),x=0),
                            legend=dict(orientation="h",x=0,y=1.05,
                                        font=dict(family="Courier New",size=10),bgcolor="rgba(0,0,0,0)"))
                        fig2.update_xaxes(gridcolor="#111",title_text="Date")
                        fig2.update_yaxes(gridcolor="#111",title_text="Growth of 1")
                        st.plotly_chart(fig2, use_container_width=True)
