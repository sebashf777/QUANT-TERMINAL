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

st.set_page_config(page_title="⚡ Quant Terminal", layout="wide", page_icon="⚡", initial_sidebar_state="collapsed")
NY  = ZoneInfo("America/New_York")
NOW = datetime.datetime.now(NY)

BS  = dict(primary="#A855F7",light="#D8B4FE",dark="#1E0A2E",mid="#2D1B4E",border="#7C3AED",glow="rgba(168,85,247,0.15)",emoji="🔮",label="BLACK-SCHOLES")
MC  = dict(primary="#06B6D4",light="#A5F3FC",dark="#021B2E",mid="#0C2D3E",border="#0891B2",glow="rgba(6,182,212,0.15)",emoji="🎲",label="MONTE CARLO")
BTH = dict(primary="#10B981",light="#A7F3D0",dark="#022C22",mid="#064E3B",border="#059669",glow="rgba(16,185,129,0.15)",emoji="🌳",label="BINOMIAL TREE")
CF  = dict(primary="#F59E0B",light="#FDE68A",dark="#1C1400",mid="#292000",border="#D97706",glow="rgba(245,158,11,0.15)",emoji="💼",label="CFO SUITE")
QT  = dict(primary="#EC4899",light="#FBCFE8",dark="#1F0010",mid="#3B0023",border="#DB2777",glow="rgba(236,72,153,0.15)",emoji="📐",label="QUANT LAB")

st.markdown("""
<style>
  html,body,.stApp{background:#020202!important;color:#E5E7EB;}
  section[data-testid="stSidebar"]{display:none;}
  .block-container{padding-top:0.3rem!important;max-width:100%!important;}
  .stTabs [data-baseweb="tab-list"]{background:#020202;border-bottom:1px solid #222;gap:2px;padding:0 4px;}
  .stTabs [data-baseweb="tab"]{background:#0a0a0a;color:#555;border:1px solid #222;
    border-radius:6px 6px 0 0;font-family:monospace;font-weight:bold;padding:8px 16px;font-size:12px;}
  .stTabs [aria-selected="true"]{background:linear-gradient(135deg,#A855F7,#06B6D4)!important;color:#fff!important;border-color:transparent!important;}
  .stButton>button{font-family:monospace;font-weight:bold;border:none;border-radius:6px;padding:7px 18px;font-size:13px;transition:all 0.2s;}
  .bs-btn>button{background:linear-gradient(135deg,#7C3AED,#A855F7);color:#fff;}
  .mc-btn>button{background:linear-gradient(135deg,#0891B2,#06B6D4);color:#fff;}
  .bt-btn>button{background:linear-gradient(135deg,#059669,#10B981);color:#fff;}
  .cf-btn>button{background:linear-gradient(135deg,#D97706,#F59E0B);color:#000;}
  .qt-btn>button{background:linear-gradient(135deg,#DB2777,#EC4899);color:#fff;}
  .all-btn>button{background:linear-gradient(135deg,#A855F7,#06B6D4,#10B981);color:#fff;font-size:14px;}
  label,.stSlider label,.stNumberInput label,.stSelectbox label,.stTextInput label,.stDateInput label{
    font-family:monospace!important;font-weight:bold!important;font-size:11px!important;color:#888!important;}
  input,select{background:#0D0D0D!important;color:#E5E7EB!important;border:1px solid #333!important;
    font-family:monospace!important;border-radius:5px!important;}
  h1,h2,h3{font-family:monospace!important;color:#E5E7EB!important;}
  div[data-testid="stDateInput"] input{background:#0D0D0D!important;color:#E5E7EB!important;}
</style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  DATA HELPERS
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def batch_quotes(tickers_tuple):
    tickers=list(tickers_tuple)
    result={t:dict(price=0.0,chg=0.0,pct=0.0,vol=0) for t in tickers}
    try:
        raw=yf.download(tickers,period="5d",auto_adjust=True,progress=False)
        for tk in tickers:
            try:
                if isinstance(raw.columns,pd.MultiIndex):
                    cl=raw["Close"][tk].dropna()
                    vo=raw["Volume"][tk].dropna() if "Volume" in raw else pd.Series()
                else:
                    cl=raw["Close"].dropna(); vo=raw.get("Volume",pd.Series()).dropna()
                if len(cl)<2: continue
                p=float(cl.iloc[-1]); pr=float(cl.iloc[-2])
                chg=p-pr; pct=chg/pr*100
                vol=int(vo.iloc[-1]) if len(vo)>0 else 0
                result[tk]=dict(price=p,chg=chg,pct=pct,vol=vol)
            except: pass
    except: pass
    return result

@st.cache_data(ttl=300)
def fetch_ohlcv(ticker,period="3mo"):
    try:
        df=yf.download(ticker,period=period,auto_adjust=True,progress=False)
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        return df[df["Close"].notna()].copy()
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_range(ticker,start,end):
    try:
        df=yf.download(ticker,start=str(start),end=str(end),auto_adjust=True,progress=False)
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
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

def fp(p):
    if p>10000: return f"{p:,.0f}"
    elif p>100: return f"{p:,.2f}"
    elif p>1:   return f"{p:.4f}"
    elif p>0:   return f"{p:.6f}"
    return "—"
def fc(c):
    a=abs(c)
    if a>100: return f"{a:,.2f}"
    elif a>0.1: return f"{a:.4f}"
    return f"{a:.6f}"

# ── FIXED stat_box (no nested conditional in f-string) ───────
def stat_box(col, label, val, color="#E5E7EB", sub=""):
    sub_html = f"<div style='color:#444;font-size:10px;margin-top:3px'>{sub}</div>" if sub else ""
    html = (
        "<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:8px;"
        "padding:12px 16px;font-family:Courier New,monospace'>"
        f"<div style='color:#555;font-size:10px;margin-bottom:4px'>{label}</div>"
        f"<div style='color:{color};font-size:19px;font-weight:bold'>{val}</div>"
        f"{sub_html}"
        "</div>"
    )
    col.markdown(html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  MATH: OPTIONS
# ════════════════════════════════════════════════════════════
def bs_params(St,K,r,sigma,T,q=0):
    d1=(math.log(St/K)+(r-q+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    return d1,d1-sigma*math.sqrt(T)

def run_bs(St,K,sigma,T,r,q):
    d1,d2=bs_params(St,K,r,sigma,T,q)
    sc=St*math.exp(-q*T)*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)
    sp=K*math.exp(-r*T)*norm.cdf(-d2)-St*math.exp(-q*T)*norm.cdf(-d1)
    dc=math.exp(-r*T)*norm.cdf(d2); dp=math.exp(-r*T)*norm.cdf(-d2)
    delta_c=math.exp(-q*T)*norm.cdf(d1); delta_p=-math.exp(-q*T)*norm.cdf(-d1)
    gamma=math.exp(-q*T)*norm.pdf(d1)/(St*sigma*math.sqrt(T))
    vega=St*math.exp(-q*T)*norm.pdf(d1)*math.sqrt(T)/100
    theta_c=(-(St*norm.pdf(d1)*sigma*math.exp(-q*T))/(2*math.sqrt(T))-r*K*math.exp(-r*T)*norm.cdf(d2)+q*St*math.exp(-q*T)*norm.cdf(d1))/365
    theta_p=(-(St*norm.pdf(d1)*sigma*math.exp(-q*T))/(2*math.sqrt(T))+r*K*math.exp(-r*T)*norm.cdf(-d2)-q*St*math.exp(-q*T)*norm.cdf(-d1))/365
    return dict(d1=d1,d2=d2,dc=dc,sc=sc,dp=dp,sp=sp,delta_c=delta_c,delta_p=delta_p,gamma=gamma,vega=vega,theta_c=theta_c,theta_p=theta_p)

def run_mc(St,K,sigma,T,r,reps):
    np.random.seed(42)
    Z=np.random.standard_normal(reps)
    ST=St*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*Z)
    d1,d2=bs_params(St,K,r,sigma,T)
    return dict(d1=d1,d2=d2,dc=np.exp(-r*T)*np.mean(ST>K),sc=np.exp(-r*T)*np.mean(np.maximum(ST-K,0)),
                dp=np.exp(-r*T)*np.mean(ST<K),sp=np.exp(-r*T)*np.mean(np.maximum(K-ST,0)),ST=ST,reps=reps)

def run_bt(St,K,sigma,T,r,N):
    dt=T/N; u=math.exp(sigma*math.sqrt(dt)); d=1/u
    p=(math.exp(r*dt)-d)/(u-d)
    ST=np.array([St*(u**j)*(d**(N-j)) for j in range(N+1)])
    cp=np.maximum(ST-K,0); pp=np.maximum(K-ST,0)
    dcp=(ST>K).astype(float); dpp=(ST<K).astype(float)
    for _ in range(N,0,-1):
        cp=np.exp(-r*dt)*(p*cp[1:]+(1-p)*cp[:-1])
        pp=np.exp(-r*dt)*(p*pp[1:]+(1-p)*pp[:-1])
        dcp=np.exp(-r*dt)*(p*dcp[1:]+(1-p)*dcp[:-1])
        dpp=np.exp(-r*dt)*(p*dpp[1:]+(1-p)*dpp[:-1])
    d1,d2=bs_params(St,K,r,sigma,T)
    return dict(d1=d1,d2=d2,dc=dcp[0],sc=cp[0],dp=dpp[0],sp=pp[0])

# ════════════════════════════════════════════════════════════
#  MATH: PORTFOLIO OPTIMIZATION
# ════════════════════════════════════════════════════════════
def port_stats(w,mu,cov):
    ret=float(np.dot(w,mu))
    vol=float(np.sqrt(w@cov@w))
    return ret,vol

def max_sharpe_w(mu,cov,rf):
    n=len(mu)
    def neg_sh(w):
        r,v=port_stats(w,mu,cov)
        return -(r-rf)/v if v>1e-10 else 0
    res=minimize(neg_sh,np.ones(n)/n,bounds=[(0,1)]*n,
                 constraints={'type':'eq','fun':lambda x:sum(x)-1},method='SLSQP',
                 options={'ftol':1e-12,'maxiter':1000})
    return res.x if res.success else np.ones(n)/n

def min_var_w(mu,cov):
    n=len(mu)
    res=minimize(lambda w:port_stats(w,mu,cov)[1],np.ones(n)/n,
                 bounds=[(0,1)]*n,
                 constraints={'type':'eq','fun':lambda x:sum(x)-1},method='SLSQP',
                 options={'ftol':1e-12,'maxiter':1000})
    return res.x if res.success else np.ones(n)/n

def calc_frontier(mu,cov,n_pts=60):
    mv=min_var_w(mu,cov)
    mv_r,_=port_stats(mv,mu,cov)
    max_r=float(np.max(mu))*0.92
    targets=np.linspace(mv_r,max_r,n_pts)
    vols,rets=[],[]
    x0=mv.copy()
    for t in targets:
        try:
            res=minimize(lambda w:port_stats(w,mu,cov)[1],x0,
                         bounds=[(0,1)]*len(mu),
                         constraints=[
                             {'type':'eq','fun':lambda x:sum(x)-1},
                             {'type':'eq','fun':lambda x,tgt=t:port_stats(x,mu,cov)[0]-tgt}
                         ],method='SLSQP',options={'ftol':1e-12,'maxiter':500})
            if res.success and res.fun>1e-6:
                r,v=port_stats(res.x,mu,cov)
                if abs(r-t)<0.008:
                    vols.append(v); rets.append(r); x0=res.x.copy()
        except: pass
    return np.array(vols),np.array(rets)

# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(90deg,#0A0010,#001020,#001A0E);
            padding:14px 24px;border-bottom:1px solid #222;
            display:flex;justify-content:space-between;align-items:center;margin-bottom:0'>
  <div>
    <span style='font-size:24px;font-weight:bold;font-family:monospace;
      background:linear-gradient(90deg,#A855F7,#06B6D4,#10B981,#F59E0B);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:3px'>
      ⚡ QUANT TERMINAL
    </span>
    <span style='color:#333;font-family:monospace;font-size:11px;margin-left:12px'>
      Market Intelligence · Options · Quant · CFO Suite
    </span>
  </div>
  <div style='font-family:monospace;font-size:11px;color:#444'>
    {NOW.strftime("%A %b %d, %Y  |  %I:%M %p ET")} &nbsp;|&nbsp;
    <span style='color:#10B981'>● LIVE</span>
  </div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
T_MARKET,T_SCREEN,T_CHART,T_CALC,T_CFO,T_QUANT = st.tabs([
    "🌍 MARKET OVERVIEW","📡 STOCK SIGNALS","📊 CHART ANALYZER",
    "📐 FINANCE CALCULATOR","💼 CFO CALCULATORS","🔬 QUANT LAB"
])

# ════════════════════════════════════════════════════════════
#  TAB 1 — MARKET OVERVIEW
# ════════════════════════════════════════════════════════════
with T_MARKET:
    if st.button("🔄 Refresh",key="ref_mkt"): st.cache_data.clear()
    INDICES={"S&P 500":"^GSPC","NASDAQ":"^IXIC","Dow Jones":"^DJI","Russell 2K":"^RUT",
             "VIX":"^VIX","10Y Yield":"^TNX","Gold":"GC=F","WTI Oil":"CL=F",
             "BTC":"BTC-USD","EUR/USD":"EURUSD=X","USD/JPY":"USDJPY=X","Silver":"SI=F"}
    with st.spinner("Loading market data..."):
        q=batch_quotes(tuple(INDICES.values()))
    bar_html="<div style='display:flex;gap:8px;flex-wrap:wrap;padding:10px 0;margin-bottom:8px'>"
    for name,tk in INDICES.items():
        d=q.get(tk,dict(price=0,chg=0,pct=0))
        col_c="#00FF41" if d["chg"]>=0 else "#FF4444"
        sign="▲" if d["chg"]>=0 else "▼"
        bar_html+=(f"<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:6px;"
                   f"padding:7px 12px;font-family:monospace;min-width:110px'>"
                   f"<div style='color:#777;font-size:9px'>{name}</div>"
                   f"<div style='color:#E5E7EB;font-size:13px;font-weight:bold'>{fp(d['price'])}</div>"
                   f"<div style='color:{col_c};font-size:11px'>{sign}{abs(d['pct']):.2f}%</div></div>")
    bar_html+="</div>"
    st.markdown(bar_html,unsafe_allow_html=True)

    c1,c2=st.columns([3,2])
    with c1:
        advancing=2007;declining=3315;total=advancing+declining;adv_pct=advancing/total*100
        st.markdown(
            "<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:8px;padding:14px;margin-bottom:10px'>"
            "<div style='color:#A855F7;font-family:monospace;font-weight:bold;font-size:12px;margin-bottom:10px'>📊 MARKET BREADTH</div>"
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
            f"<div style='background:linear-gradient(90deg,#00FF41,#10B981);height:100%;width:{adv_pct:.1f}%'></div></div>"
            "</div>",
            unsafe_allow_html=True
        )
        g2a,g2b=st.columns(2)
        GAINERS=[("BIAF",2.12,98.13),("AIFF",2.75,58.05),("SVCO",5.03,52.42),("ELPW",5.16,43.73),("PLYX",6.36,36.77),("APEI",57.66,21.19)]
        LOSERS=[("IMMP",0.48,-81.64),("IBG",1.10,-55.28),("CMCT",0.63,-44.31),("BHAT",0.68,-43.75),("KLC",1.95,-42.65),("CDIO",2.80,-38.60)]
        def sig_table(title,data,color,icon):
            rows="".join(f"<tr><td style='color:{color};padding:5px 8px;font-weight:bold'>{tk}</td>"
                         f"<td style='color:#E5E7EB;padding:5px 8px'>{px}</td>"
                         f"<td style='color:{color};padding:5px 8px'>{pct:+.2f}%</td></tr>"
                         for tk,px,pct in data)
            return (f"<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:8px;padding:12px;margin-bottom:8px'>"
                    f"<div style='color:{color};font-family:monospace;font-weight:bold;font-size:12px;margin-bottom:8px'>{icon} {title}</div>"
                    f"<table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:12px'>"
                    f"<thead><tr><td style='color:#555;padding:4px 8px;font-size:10px'>TICKER</td>"
                    f"<td style='color:#555;padding:4px 8px;font-size:10px'>PRICE</td>"
                    f"<td style='color:#555;padding:4px 8px;font-size:10px'>CHG%</td></tr></thead>"
                    f"<tbody>{rows}</tbody></table></div>")
        with g2a: st.markdown(sig_table("TOP GAINERS",GAINERS,"#00FF41","🚀"),unsafe_allow_html=True)
        with g2b: st.markdown(sig_table("TOP LOSERS",LOSERS,"#FF4444","📉"),unsafe_allow_html=True)

    with c2:
        FUTURES={"Crude Oil":("CL=F","99.31","+3.74%","#00FF41"),"Natural Gas":("NG=F","3.13","-3.12%","#FF4444"),
                 "Gold":("GC=F","5,023","-2.00%","#FF4444"),"S&P 500 Fut":("ES=F","6,625","-1.52%","#FF4444"),
                 "Nasdaq Fut":("NQ=F","24,335","-1.76%","#FF4444"),"Dow Fut":("YM=F","46,511","-1.07%","#FF4444")}
        rows="".join(f"<tr><td style='color:#E5E7EB;padding:6px 10px;font-weight:bold'>{name}</td>"
                     f"<td style='color:#F59E0B;padding:6px 10px'>{px}</td>"
                     f"<td style='color:{col_c};padding:6px 10px'>{pct}</td></tr>"
                     for name,(tk,px,pct,col_c) in FUTURES.items())
        st.markdown(f"<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:8px;padding:12px;margin-bottom:10px'>"
                    f"<div style='color:#F59E0B;font-family:monospace;font-weight:bold;font-size:12px;margin-bottom:8px'>⚡ FUTURES</div>"
                    f"<table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:12px'><tbody>{rows}</tbody></table></div>",
                    unsafe_allow_html=True)
        ECON=[("GDP Price Index QoQ","3.8%","3.7%","#F59E0B"),("PCE Price Index YoY","2.8%","2.9%","#00FF41"),
              ("Personal Income MoM","0.4%","0.5%","#FF4444"),("Michigan Sentiment","55.5","55.0","#F59E0B"),
              ("JOLTs Job Openings","6.95M","6.70M","#00FF41")]
        rows="".join(f"<tr><td style='color:#aaa;padding:5px 8px;font-size:11px'>{ev}</td>"
                     f"<td style='color:{col_c};padding:5px 8px;font-weight:bold'>{act}</td>"
                     f"<td style='color:#555;padding:5px 8px'>{exp}</td></tr>"
                     for ev,act,exp,col_c in ECON)
        st.markdown(f"<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:8px;padding:12px;margin-bottom:10px'>"
                    f"<div style='color:#06B6D4;font-family:monospace;font-weight:bold;font-size:12px;margin-bottom:8px'>📅 ECONOMIC CALENDAR</div>"
                    f"<table style='width:100%;border-collapse:collapse;font-family:monospace'>"
                    f"<thead><tr><td style='color:#555;padding:4px 8px;font-size:10px'>EVENT</td>"
                    f"<td style='color:#555;padding:4px 8px;font-size:10px'>ACTUAL</td>"
                    f"<td style='color:#555;padding:4px 8px;font-size:10px'>EXPECTED</td></tr></thead>"
                    f"<tbody>{rows}</tbody></table></div>",unsafe_allow_html=True)
        INSIDERS=[("AMR","Buy","$1.87M","#00FF41"),("XENE","Sale","$403K","#FF4444"),
                  ("VRT","Prop. Sale","$263M","#FF4444"),("SVRE","Buy","$688M","#00FF41")]
        rows="".join(f"<tr><td style='color:#F59E0B;padding:5px 8px;font-weight:bold'>{tk}</td>"
                     f"<td style='color:{col_c};padding:5px 8px'>{txn}</td>"
                     f"<td style='color:#aaa;padding:5px 8px'>{val}</td></tr>"
                     for tk,txn,val,col_c in INSIDERS)
        st.markdown(f"<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:8px;padding:12px'>"
                    f"<div style='color:#EC4899;font-family:monospace;font-weight:bold;font-size:12px;margin-bottom:8px'>🕵️ INSIDER ACTIVITY</div>"
                    f"<table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:12px'><tbody>{rows}</tbody></table></div>",
                    unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TAB 2 — STOCK SIGNALS (FIXED: stat_box HTML)
# ════════════════════════════════════════════════════════════
with T_SCREEN:
    st.markdown("<h4 style='color:#A855F7;font-family:monospace;margin-bottom:10px'>📡 LIVE STOCK SIGNALS</h4>",unsafe_allow_html=True)
    sc1,sc2,sc3=st.columns([2,2,8])
    with sc1: screen_tk=st.text_input("Ticker",value="AAPL",key="sc_tk").upper().strip()
    with sc2: screen_per=st.selectbox("Period",["1mo","3mo","6mo","1y"],index=2,key="sc_per")
    with sc3:
        st.markdown("<br>",unsafe_allow_html=True)
        run_screen=st.button("🔍 Analyze Stock",key="run_screen")

    if (run_screen or screen_tk) and screen_tk:
        with st.spinner(f"Analyzing {screen_tk}..."):
            df=fetch_ohlcv(screen_tk,screen_per)
            info=get_info(screen_tk)
            qd=batch_quotes((screen_tk,)).get(screen_tk,dict(price=0,chg=0,pct=0,vol=0))
            news=get_news(screen_tk)

        if not df.empty and len(df)>30:
            c=df["Close"]
            sma20=float(c.rolling(20).mean().iloc[-1])
            sma50=float(c.rolling(50).mean().iloc[-1]) if len(c)>50 else sma20
            ema12=c.ewm(span=12).mean(); ema26=c.ewm(span=26).mean()
            macd_line=float((ema12-ema26).iloc[-1])
            sig_line=float(((ema12-ema26).ewm(span=9).mean()).iloc[-1])
            dd=c.diff(); g=dd.clip(lower=0).rolling(14).mean(); l=(-dd.clip(upper=0)).rolling(14).mean()
            rsi_val=float((100-100/(1+g/l.replace(0,np.nan))).iloc[-1])
            bb_mid=float(c.rolling(20).mean().iloc[-1]); bb_std=float(c.rolling(20).std().iloc[-1])
            bb_up=bb_mid+2*bb_std; bb_lo=bb_mid-2*bb_std
            hi52=float(c.tail(252).max()) if len(c)>252 else float(c.max())
            lo52=float(c.tail(252).min()) if len(c)>252 else float(c.min())
            price=qd["price"] if qd["price"]>0 else float(c.iloc[-1])

            signals=[]
            if price>sma20: signals.append(("▲ Above SMA20","#00FF41"))
            else: signals.append(("▼ Below SMA20","#FF4444"))
            if price>sma50: signals.append(("▲ Above SMA50","#00FF41"))
            else: signals.append(("▼ Below SMA50","#FF4444"))
            if macd_line>sig_line: signals.append(("📈 MACD Bullish","#00FF41"))
            else: signals.append(("📉 MACD Bearish","#FF4444"))
            if rsi_val>70: signals.append(("🔴 RSI Overbought","#FF4444"))
            elif rsi_val<30: signals.append(("🟢 RSI Oversold","#00FF41"))
            else: signals.append((f"🟡 RSI Neutral {rsi_val:.1f}","#F59E0B"))
            if price>bb_up: signals.append(("📊 Above BB Upper","#FF4444"))
            elif price<bb_lo: signals.append(("📊 Below BB Lower","#00FF41"))
            else: signals.append(("📊 Inside BB","#06B6D4"))

            bull=sum(1 for _,c2 in signals if "#00FF41" in c2)
            bear=sum(1 for _,c2 in signals if "#FF4444" in c2)
            score_col="#00FF41" if bull>bear else ("#FF4444" if bear>bull else "#F59E0B")
            score_label="BULLISH 🐂" if bull>bear else ("BEARISH 🐻" if bear>bull else "NEUTRAL ⚖️")

            # ── FIXED: stat_box called with clean string values ──
            sa,sb,sc2x,sd=st.columns(4)
            price_str=fp(price)
            chg_sign="▲" if qd["chg"]>=0 else "▼"
            chg_color="#00FF41" if qd["chg"]>=0 else "#FF4444"
            chg_str=f"{chg_sign}{fc(qd['chg'])} ({qd['pct']:+.2f}%)"
            rsi_color="#FF4444" if rsi_val>70 else ("#00FF41" if rsi_val<30 else "#F59E0B")

            stat_box(sa,"PRICE",price_str,"#F59E0B")
            stat_box(sb,"CHANGE",chg_str,chg_color)
            stat_box(sc2x,"RSI (14)",f"{rsi_val:.1f}",rsi_color)
            stat_box(sd,"SIGNAL",score_label,score_col,f"{bull} bull · {bear} bear")

            sig_html="<div style='display:flex;flex-wrap:wrap;gap:8px;margin:12px 0'>"
            for s,c2x in signals:
                sig_html+=f"<span style='background:#0D0D0D;color:{c2x};border:1px solid {c2x}44;padding:5px 12px;border-radius:20px;font-family:monospace;font-size:11px'>{s}</span>"
            sig_html+="</div>"
            st.markdown(sig_html,unsafe_allow_html=True)

            fundamentals={
                "Market Cap":   f"${info.get('marketCap',0)/1e9:.1f}B" if info.get('marketCap') else "—",
                "P/E Ratio":    f"{info.get('trailingPE',0):.1f}x" if info.get('trailingPE') else "—",
                "EPS (TTM)":    f"${info.get('trailingEps',0):.2f}" if info.get('trailingEps') else "—",
                "Revenue":      f"${info.get('totalRevenue',0)/1e9:.1f}B" if info.get('totalRevenue') else "—",
                "Profit Margin":f"{info.get('profitMargins',0)*100:.1f}%" if info.get('profitMargins') else "—",
                "52W High":     fp(hi52),"52W Low":fp(lo52),
                "Beta":         f"{info.get('beta',0):.2f}" if info.get('beta') else "—",
                "Div Yield":    f"{info.get('dividendYield',0)*100:.2f}%" if info.get('dividendYield') else "0%",
                "Volume":       f"{qd['vol']:,.0f}",
            }
            fund_html="<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:12px'>"
            for k,v in fundamentals.items():
                fund_html+=(f"<div style='background:#0D0D0D;border:1px solid #1a1a1a;border-radius:6px;padding:10px 12px'>"
                            f"<div style='color:#555;font-size:10px;font-family:monospace'>{k}</div>"
                            f"<div style='color:#E5E7EB;font-size:14px;font-weight:bold;font-family:monospace'>{v}</div>"
                            f"</div>")
            fund_html+="</div>"
            st.markdown(fund_html,unsafe_allow_html=True)

            d1v,_,d3v=st.columns([4,1,2])
            with d1v:
                fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.75,.25],vertical_spacing=0.02)
                fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
                    increasing_line_color="#00FF41",decreasing_line_color="#FF4444",name=screen_tk),row=1,col=1)
                fig.add_trace(go.Scatter(x=df.index,y=c.rolling(20).mean(),line=dict(color="#F59E0B",width=1.2),name="SMA20"),row=1,col=1)
                fig.add_trace(go.Scatter(x=df.index,y=c.rolling(50).mean(),line=dict(color="#06B6D4",width=1.2),name="SMA50"),row=1,col=1)
                bb_mids=c.rolling(20).mean(); bb_stds=c.rolling(20).std()
                fig.add_trace(go.Scatter(x=df.index,y=bb_mids+2*bb_stds,line=dict(color="rgba(168,85,247,0.4)",width=1,dash="dash"),showlegend=False),row=1,col=1)
                fig.add_trace(go.Scatter(x=df.index,y=bb_mids-2*bb_stds,line=dict(color="rgba(168,85,247,0.4)",width=1,dash="dash"),fill="tonexty",fillcolor="rgba(168,85,247,0.03)",showlegend=False),row=1,col=1)
                vcols=["#00FF41" if float(cl)>=float(op) else "#FF4444" for cl,op in zip(df["Close"],df["Open"])]
                fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vcols,showlegend=False,opacity=0.6),row=2,col=1)
                fig.update_layout(template="plotly_dark",paper_bgcolor="#020202",plot_bgcolor="#0D0D0D",height=420,
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h",x=0,y=1.05,font=dict(family="
