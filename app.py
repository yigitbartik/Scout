import streamlit as st
import pandas as pd
import database as db
import scraper as sc
import scout_logic as logic
from streamlit_option_menu import option_menu
import json
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import seaborn as sns
import altair as alt
from mplsoccer import Pitch, PyPizza
from scipy import stats
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re
import random
import datetime

# --- KONFİGÜRASYON ---
st.set_page_config(page_title="SCOUT | Futbol Veri Merkezi", layout="wide", page_icon="⚽", initial_sidebar_state="expanded")
WHITE_BG = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0f172a'))
PLOT_CONFIG = {'displayModeBar': True}

# --- SABİTLER VE AYARLAR ---
cxG = 1.53570624482222
norm = mcolors.Normalize(vmin=1, vmax=16)
try:
    cmap = matplotlib.colormaps['coolwarm']
except:
    cmap = plt.get_cmap('coolwarm')

# --- YARDIMCI FONKSİYONLAR ---

def color_percentile(pc):
    rgb = cmap(norm(pc))
    return 'color: #%02x%02x%02x; opacity: 1; textcolor: white' % (int(rgb[0]*100), int(rgb[1]*100), int(rgb[2]*100))

def parse_market_value(val_str):
    if not val_str or val_str == '-': return 0.0
    val_str = str(val_str).lower().replace('€', '').replace('£', '')
    try:
        if 'm' in val_str:
            return float(re.sub(r'[^\d.]', '', val_str)) * 1.0
        elif 'k' in val_str:
            return float(re.sub(r'[^\d.]', '', val_str)) / 1000.0
        return 0.0
    except: return 0.0

def extract_year(date_str):
    if not date_str or date_str == '-': return 2025
    try:
        match = re.search(r'\d{4}', str(date_str))
        if match: return int(match.group(0))
        return 2025
    except: return 2025

def get_team_logo(team_name, web_url=None):
    if web_url and "http" in str(web_url): return web_url
    return "https://tmssl.akamaized.net/images/wappen/head/default.png"

# --- SİMÜLASYON MOTORU ---
def generate_simulated_league_data():
    """
    Veritabanındaki mevcut takımları ve oyuncuları kullanarak
    sanal bir lig fikstürü ve maç sonuçları simüle eder.
    """
    try:
        players_df = db.get_all_players_detailed()
    except:
        return pd.DataFrame()

    if players_df.empty:
        return pd.DataFrame()

    unique_teams = players_df['team'].unique().tolist()
    # Eğer yeterli takım yoksa dummy takımlar ekle
    if len(unique_teams) < 2: 
        unique_teams = ["A Takımı", "B Takımı"]
    
    logo_map = {}
    squad_map = {}
    
    # Takım logolarını ve kadrolarını hazırla
    for team in unique_teams:
        team_players = players_df[players_df['team'] == team]
        if not team_players.empty and team_players.iloc[0]['club_logo']:
            logo_map[team] = team_players.iloc[0]['club_logo']
        else:
            logo_map[team] = "https://tmssl.akamaized.net/images/wappen/head/default.png"
        squad_map[team] = team_players['name'].tolist()

    data = []
    base_date = datetime.date(2025, 8, 10)
    teams_shuffled = unique_teams.copy()
    
    # 5 Haftalık Fikstür Simülasyonu
    for week in range(5):
        week_date = base_date + datetime.timedelta(weeks=week)
        random.shuffle(teams_shuffled)
        
        matches = []
        # Basit eşleştirme
        for i in range(0, len(teams_shuffled) - 1, 2):
            matches.append((teams_shuffled[i], teams_shuffled[i+1]))
            
        for t1, t2 in matches:
            # Poisson dağılımı ile gol sayıları (Ev sahibi avantajı biraz daha fazla)
            g1 = np.random.poisson(1.45)
            g2 = np.random.poisson(1.15)
            match_name = f"{t1} {g1}-{g2} {t2}"
            
            # Golcüleri kadrodan rastgele seç
            scorers_t1 = []
            if g1 > 0 and squad_map.get(t1):
                scorers_t1 = random.choices(squad_map[t1], k=g1)
            
            scorers_t2 = []
            if g2 > 0 and squad_map.get(t2):
                scorers_t2 = random.choices(squad_map[t2], k=g2)
                
            # Maçın Adamı (MOTM)
            motm = "-"
            if g1 > g2 and squad_map.get(t1): motm = random.choice(squad_map[t1])
            elif g2 > g1 and squad_map.get(t2): motm = random.choice(squad_map[t2])
            elif squad_map.get(t1): motm = random.choice(squad_map[t1])

            # İstatistik Simülasyonu
            xg1 = max(0.1, round(np.random.normal(g1 * 0.75 + 0.3, 0.4), 2))
            xg2 = max(0.1, round(np.random.normal(g2 * 0.75 + 0.3, 0.4), 2))
            poss1 = np.random.randint(35, 65)
            
            # Ev Sahibi Kaydı
            data.append({
                "Team": t1, "Match": match_name, "Date": week_date,
                "Result": "W" if g1 > g2 else ("L" if g1 < g2 else "D"),
                "Goals": g1, "Goals Conceded": g2,
                "xG": xg1, "xGA": xg2, "xGD": round(xg1 - xg2, 2),
                "Possession": poss1, "Field Tilt": poss1 + random.randint(-5, 10),
                "Shots": random.randint(g1+3, g1+15), 
                "xT": round(random.uniform(0.8, 2.5), 2),
                "Logo": logo_map.get(t1),
                "Scorers": ", ".join(scorers_t1) if scorers_t1 else "-",
                "MOTM": motm, "Opponent": t2
            })
            
            # Deplasman Kaydı
            data.append({
                "Team": t2, "Match": match_name, "Date": week_date,
                "Result": "W" if g2 > g1 else ("L" if g2 < g1 else "D"),
                "Goals": g2, "Goals Conceded": g1,
                "xG": xg2, "xGA": xg1, "xGD": round(xg2 - xg1, 2),
                "Possession": 100 - poss1, "Field Tilt": 100 - (poss1 + random.randint(-5, 10)),
                "Shots": random.randint(g2+3, g2+15),
                "xT": round(random.uniform(0.8, 2.5), 2),
                "Logo": logo_map.get(t2),
                "Scorers": ", ".join(scorers_t2) if scorers_t2 else "-",
                "MOTM": motm, "Opponent": t1
            })
            
    df = pd.DataFrame(data)
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        conditions_pts = [df['Result'] == 'W', df['Result'] == 'D']
        df['Pts'] = np.select(conditions_pts, [3, 1], default=0)
        df['xT Difference'] = df['xT'] - (df['xT'] * random.uniform(0.7, 1.3))
    
    return df

def table_start_end(df, start_date, end_date):
    df = df.copy()
    df = df[df['Date'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))]
    df['Win'] = (df['Result'] == 'W').astype(int)
    df['Draw'] = (df['Result'] == 'D').astype(int)
    df['Loss'] = (df['Result'] == 'L').astype(int)
    
    table = df.groupby(['Team']).agg({
        'Result': 'count', 'Pts': 'sum',
        'Win': 'sum', 'Draw': 'sum', 'Loss': 'sum',
        'Goals': 'sum', 'Goals Conceded': 'sum',
        'xG': 'sum', 'xGA': 'sum', 'xGD': 'sum'
    }).reset_index()
    
    table = table.sort_values(by=['Pts', 'Goals', 'Win'], ascending=[False, False, False])
    table = table.rename(columns={'Win': 'W', 'Draw': 'D', 'Loss': 'L', 'Goals': 'GF', 'Goals Conceded': 'GA', 'Result': 'GP'})
    table.reset_index(drop=True, inplace=True)
    table.reset_index(drop=False, inplace=True)
    table.rename(columns={'index': 'Pos'}, inplace=True)
    table['Pos'] = table['Pos'] + 1
    return table

def add_mov_avg(df, var):
    df['4-Match Moving Average'] = np.nan
    for i in range(len(df)):
        if i + 4 <= len(df):
            df.loc[i, '4-Match Moving Average'] = df[var][i:i+4].mean()
    return df

@alt.theme.register('ben_theme', enable=True)
def ben_theme():
    return {
        'config': {
            'background': '#fbf9f4',
            'axis': {'titleColor': '#4a2e19', 'labelColor': '#4a2e19'},
            'text': {'fill': '#4a2e19'},
            'title': {'color': '#4a2e19', 'subtitleColor': '#4a2e19'}
        }
    }

# --- GÖRSELLEŞTİRME ---
def create_image_scatter_plot(df, x_col, y_col, image_col, title):
    fig = go.Figure()
    if df.empty: return fig
    
    if 'name' in df.columns: label_col = 'name'
    elif 'Team' in df.columns: label_col = 'Team'
    elif 'team' in df.columns: label_col = 'team'
    else: label_col = df.columns[0]

    try:
        hover_text = df.apply(lambda row: f"{row[label_col]}<br>{x_col}: {row[x_col]}<br>{y_col}: {row[y_col]}", axis=1)
    except: hover_text = df.index

    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', marker=dict(opacity=0), text=hover_text, hoverinfo='text'))
    
    for index, row in df.iterrows():
        img_url = row.get(image_col)
        if not img_url or "http" not in str(img_url): img_url = "https://tmssl.akamaized.net/images/wappen/head/default.png"
        
        # Dinamik boyutlandırma
        x_range = df[x_col].max() - df[x_col].min()
        y_range = df[y_col].max() - df[y_col].min()
        img_w = x_range * 0.08 if x_range > 0 else 1
        img_h = y_range * 0.10 if y_range > 0 else 1

        fig.add_layout_image(dict(source=img_url, xref="x", yref="y", x=row[x_col], y=row[y_col], sizex=img_w, sizey=img_h, xanchor="center", yanchor="middle", opacity=0.9, layer="above"))
    
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col, **WHITE_BG, height=600)
    return fig

def create_pizza_chart(player_name, metrics, values, percentiles):
    slice_colors = ["#1A78CF"] * 5 + ["#FF9300"] * 5 + ["#D70232"] * (len(metrics) - 10)
    slice_colors = slice_colors[:len(metrics)]
    text_colors = ["#000000"] * len(metrics)
    baker = PyPizza(params=metrics, background_color="#f8fafc", straight_line_color="#EBEBE9", straight_line_lw=1, last_circle_lw=0, other_circle_lw=0, inner_circle_size=20)
    fig, ax = baker.make_pizza(percentiles, figsize=(8, 8), param_location=110, slice_colors=slice_colors, value_colors=text_colors, value_bck_colors=slice_colors, kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1), kwargs_params=dict(color="#000000", fontsize=11, va="center"), kwargs_values=dict(color="#000000", fontsize=11, zorder=3, bbox=dict(edgecolor="#000000", facecolor="cornflowerblue", boxstyle="round,pad=0.2", lw=1)))
    fig.text(0.515, 0.975, f"{player_name}", size=20, ha="center", fontweight="bold", color="#000000")
    return fig

def create_beeswarm_plot(df, metric, player_name, player_val):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
    df_clean = df.dropna(subset=[metric])
    sns.stripplot(x=df_clean[metric], color="#cbd5e1", size=7, alpha=0.6, ax=ax, jitter=0.25)
    ax.scatter(player_val, 0, color="#ef4444", s=250, zorder=10, edgecolors="white", linewidth=3)
    ax.text(player_val, -0.08, player_name, ha='center', va='top', fontweight='bold', color="#ef4444", fontsize=12)
    ax.set_xlabel(metric, fontweight='bold', color="#334155"); ax.set_yticks([]); sns.despine(left=True, bottom=False)
    return fig

def draw_shot_map(player_name, position, shots_total, goals_total, theme_color):
    shots_total = int(shots_total) if shots_total > 0 else 1
    goals_total = int(goals_total)
    x_locs, y_locs, outcomes, xgs = [], [], [], []
    for i in range(shots_total):
        is_goal = i < goals_total
        outcome = 'Goal' if is_goal else 'Miss'
        if "Forvet" in position or "Kanat" in position: x, y = np.random.normal(105, 7), np.random.normal(40, 12)
        elif "Stoper" in position: x, y = np.random.normal(95, 5), np.random.normal(40, 5)
        else: x, y = np.random.normal(90, 10), np.random.normal(40, 15)
        dist = np.sqrt((120-x)**2 + (40-y)**2)
        xg = max(0.02, min(0.99, 1 / (dist/5 + 1)))
        if is_goal: xg += 0.2
        x_locs.append(x); y_locs.append(y); outcomes.append(outcome); xgs.append(xg)
    df_shots = pd.DataFrame({'x': x_locs, 'y': y_locs, 'outcome': outcomes, 'xG': xgs})
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.set_facecolor('white'); ax.patch.set_facecolor('white')
    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='#cbd5e1', linewidth=2)
    pitch.draw(ax=ax)
    goals = df_shots[df_shots['outcome'] == 'Goal']
    pitch.scatter(goals.x, goals.y, s=goals['xG']*500, marker='football', c='white', edgecolors=theme_color, zorder=2, ax=ax, label='Gol')
    misses = df_shots[df_shots['outcome'] == 'Miss']
    pitch.scatter(misses.x, misses.y, s=misses['xG']*500, marker='o', c='None', edgecolors='#94a3b8', hatch='////', alpha=0.6, zorder=1, ax=ax, label='Şut')
    npxg = df_shots['xG'].sum()
    ax.text(60, 125, f"{player_name}\n{position}", ha='center', fontsize=18, fontweight='bold', color='#0f172a')
    ax.text(60, 118, f"Şut: {shots_total} | Gol: {goals_total} | xG: {npxg:.2f}", ha='center', fontsize=12, color=theme_color, fontweight='bold')
    return fig

# --- STYLES ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #0f172a; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Oswald', sans-serif; font-weight: 700; color: #1e293b; text-transform: uppercase; letter-spacing: 1px; }
    section[data-testid="stSidebar"] { background-color: #0f172a; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] p { color: #f8fafc; }
    .league-bar { display: flex; justify-content: center; gap: 30px; padding: 15px; background: white; border-bottom: 1px solid #e2e8f0; margin-bottom: 25px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); flex-wrap: wrap; }
    .league-logo { height: 50px; object-fit: contain; filter: grayscale(100%); opacity: 0.7; transition: all 0.3s ease; cursor: pointer; }
    .league-logo:hover { filter: grayscale(0%); opacity: 1; transform: scale(1.15); }
    .player-card { background: linear-gradient(145deg, #111827, #1f2937); border-radius: 20px; padding: 25px; color: white; display: flex; align-items: center; gap: 30px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3); border: 1px solid #374151; margin-bottom: 30px; position: relative; overflow: hidden; }
    .player-card::after { content: ""; position: absolute; top: 0; right: 0; bottom: 0; left: 0; background: radial-gradient(circle at top right, rgba(34, 197, 94, 0.15), transparent 40%); z-index: 0; }
    .card-img-container { position: relative; width: 140px; height: 140px; z-index: 1; }
    .card-img { width: 140px; height: 140px; border-radius: 50%; border: 4px solid #22c55e; object-fit: cover; background: white; box-shadow: 0 0 20px rgba(34, 197, 94, 0.3); }
    .card-logo { position: absolute; bottom: 5px; right: 5px; width: 50px; height: 50px; background: white; border-radius: 50%; padding: 4px; border: 2px solid #e5e7eb; object-fit: contain; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .card-info { flex-grow: 1; z-index: 1; }
    .card-info h1 { margin: 0; font-family: 'Oswald', sans-serif; font-weight: 700; font-size: 3.2rem; color: #f8fafc; line-height: 1.1; letter-spacing: 1px; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
    .card-meta { display: flex; gap: 20px; margin-top: 10px; font-size: 1.1rem; color: #9ca3af; font-weight: 500; }
    .stat-badges { display: flex; gap: 10px; margin-top: 20px; }
    .card-badge { background: rgba(255, 255, 255, 0.05); color: #e5e7eb; padding: 8px 16px; border-radius: 8px; font-size: 0.9rem; font-weight: 600; border: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; gap: 6px; backdrop-filter: blur(10px); }
    .score-badge { background: #22c55e; color: #fff; border: none; box-shadow: 0 0 15px rgba(34, 197, 94, 0.5); font-size: 1.1rem; }
    .metric-box { background: #ffffff; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e; box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align: center; transition: transform 0.2s; margin-bottom: 10px; }
    .metric-val { font-size: 1.8rem; font-weight: 800; color: #0f172a; }
    .metric-lbl { font-size: 0.85rem; color: #64748b; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
    .comment-box { background: #f1f5f9; padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 10px; color: #334155; }
    .step-box { background: #e0f2fe; padding: 20px; border-radius: 12px; border: 1px solid #7dd3fc; height: 100%; }
    .step-num { font-size: 2rem; font-weight: 800; color: #0284c7; display: block; margin-bottom: 10px; }
    div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# --- UYGULAMA BAŞLANGICI ---
LEAGUE_LOGOS = {
    "Süper Lig": "https://tmssl.akamaized.net/images/logo/header/tr1.png",
    "Premier League": "https://tmssl.akamaized.net/images/logo/header/gb1.png",
    "La Liga": "https://tmssl.akamaized.net/images/logo/header/es1.png",
    "Bundesliga": "https://tmssl.akamaized.net/images/logo/header/l1.png",
    "Serie A": "https://tmssl.akamaized.net/images/logo/header/it1.png",
    "Ligue 1": "https://tmssl.akamaized.net/images/logo/header/fr1.png",
    "Eredivisie": "https://tmssl.akamaized.net/images/logo/header/nl1.png",
    "Liga Portugal": "https://tmssl.akamaized.net/images/logo/header/po1.png",
    "Champions League": "https://tmssl.akamaized.net/images/logo/header/cl.png"
}

def render_league_header_safe():
    logos_html = ""
    for name, url in LEAGUE_LOGOS.items():
        logos_html += f'<img src="{url}" class="league-logo" title="{name}">'
    st.markdown(f'<div class="league-bar">{logos_html}</div>', unsafe_allow_html=True)

render_league_header_safe()

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Soccerball.svg/1200px-Soccerball.svg.png", width=60)
    st.markdown("<h1 style='color: white; text-align: center;'>SCOUT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; text-align: center; margin-top: -15px;'>Futbol Karar Destek Sistemi</p>", unsafe_allow_html=True)
    
    page = option_menu("Menü", 
        ["Ana Sayfa", "Takip Listesi", "Veri Havuzu", "Oyuncu Profili", "Kıyaslama", "Analiz Dashboard", "Takım Analizi", "Maç Merkezi", "Oyuncu Ekle", "Rapor Oluştur"], 
        icons=['house', 'list-check', 'table', 'person-lines-fill', 'arrow-left-right', 'graph-up-arrow', 'diagram-3', 'trophy', 'cloud-download', 'pencil-square'], 
        default_index=0, 
        styles={"container": {"padding": "0!important", "background-color": "#0f172a"}, "icon": {"color": "#22c55e", "font-size": "18px"}, "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#1e293b", "color": "white"}, "nav-link-selected": {"background-color": "#22c55e"}})
    
    st.divider()
    
    # 2. ADIM: VERİ DOLDURMA BUTONU
    if st.button("🤖 Bot: Simülasyon Verisi Üret", type="primary"):
        with st.spinner("Bot Çalışıyor (Gaussian)..."):
            cnt = db.auto_generate_fake_reports()
            st.success(f"{cnt} oyuncu güncellendi.")
    
    st.markdown("---")
    st.caption("Dipnot: Buradaki veriler gerçek veriler olmadığından simülasyon ile oluşturulmuştur.")

# ==============================================================================
# YENİ SAYFA: ANA SAYFA (ONBOARDING)
# ==============================================================================
if page == "Ana Sayfa":
    st.title("👋 SCOUT'a Hoşgeldiniz")
    
    # Canlı İstatistikler (DB Kontrol)
    try:
        p_count = len(db.get_all_players_detailed())
        team_count = len(db.get_team_stats())
    except: p_count, team_count = 0, 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Oyuncu", p_count, delta="Canlı Veri")
    c2.metric("Takip Edilen Takım", team_count)
    c3.metric("Sistem Durumu", "Aktif", delta_color="normal")

    st.divider()
    
    # KULLANIM ADIMLARI - YENİ DÜZENLEME
    with st.expander("ℹ️ SİSTEM KULLANIM TALİMATI (11 ADIM)", expanded=True):
        st.markdown("""
        1. **Veri Tabanı Kurulumu:** Oyuncuların temel bilgileri **transfermarkt.com** üzerinden web scraping yöntemi ile sisteme çekilmiştir.
        2. **Otomatik Raporlama:** Oyuncu raporları, örnek teşkil etmesi amacıyla simülasyon yoluyla scout raporu ve verileriyle otomatik doldurulmuştur.
        3. **Veri Havuzu:** Veri Havuzu sekmesinde sistemde kayıtlı olan tüm oyuncuları listeleyebilir ve filtreleyebilirsiniz.
        4. **Takip Listesi:** Raporlanan oyuncular, belirli kriterlere göre Takip Listesi'nde farklı kategorilere (A+, A, B, D) ayrıştırılmışlardır.
        5. **Oyuncu Profili:** Seçilen oyuncunun ayrıntılı raporuna, istatistiklerine ve gelişmiş veri görselleştirmelerine bu sekmeden ulaşabilirsiniz.
        6. **Kıyaslama:** Kıyaslama ekranını kullanarak iki farklı oyuncuyu metrikler üzerinden yan yana karşılaştırabilirsiniz.
        7. **Analiz Dashboard:** Oyuncunun metriklere göre ligdeki konumunu görebilir, metrikleri değiştirerek dinamik veri görselleştirmeleri yapabilirsiniz.
        8. **Takım Analizi:** Oyuncunun oynadığı kulübün oyun tarzını ve takım verilerini anlamak için bu bölümü kullanabilirsiniz.
        9. **Maç Merkezi:** Oynanan maçlarla alakalı örnek istatistikler ve simülasyon çıktıları burada sunulmaktadır.
        10. **Oyuncu Ekleme:** Sisteme yeni bir oyuncu dahil etmek için 'Oyuncu Ekle' sekmesini kullanınız.
        11. **Rapor Düzenleme:** 'Rapor Oluştur' ekranından mevcut raporları güncelleyebilir veya oyuncunun scouting raporunu manuel olarak girebilirsiniz.
        """)
        st.info("💡 **Dipnot:** Sistemdeki veriler örnek olması açısından otomatik doldurulmuştur; gerçek verileri yansıtmamaktadır.")

    if p_count == 0:
        st.warning("⚠️ Veritabanınız boş görünüyor. Başlamak için soldaki menüden **'Oyuncu Ekle'** sekmesine gidin.")

# ==============================================================================
# SAYFA: TAKİP LİSTESİ
# ==============================================================================
elif page == "Takip Listesi":
    st.title("📋 Transfer Takip Merkezi")
    
    with st.expander("ℹ️ Bu Sayfa Ne İşe Yarar?", expanded=False):
        st.write("Burada scout ekibinizin onayladığı, takibe aldığı veya reddettiği oyuncuları **Karar Etiketine** göre (A+, A, B, D) sınıflandırılmış şekilde görebilirsiniz.")

    df = db.get_all_players_detailed()
    if df.empty: st.warning("Veri yok.")
    else:
        df['val_num'] = df['market_value'].apply(parse_market_value)
        df['contract_year'] = df['contract'].apply(extract_year)

        with st.expander("🔍 Gelişmiş Filtreleme", expanded=True):
            # Ayak ve Sözleşme Yılı filtreleri kaldırıldı, sütunlar 2'ye bölündü
            c1, c2 = st.columns(2)
            # 'Takım Oyun Tarzı' -> 'Oyun Tarzı' olarak güncellendi
            f_style = c1.multiselect("Oyun Tarzı", df['oyun_tarzi'].dropna().unique())
            f_nat = c2.multiselect("Uyruk", df['nationality'].dropna().unique())

            c3, c4 = st.columns(2)
            f_val = c3.slider("Piyasa Değeri (€ Milyon)", 0.0, float(df['val_num'].max()), (0.0, 100.0))
            f_age = c4.slider("Yaş Aralığı", 15, 40, (15, 40))

        dff = df.copy()
        if f_style: dff = dff[dff['oyun_tarzi'].isin(f_style)]
        # f_foot ve f_year filtre kontrolleri kaldırıldı
        if f_nat: dff = dff[dff['nationality'].isin(f_nat)]
        dff = dff[(dff['val_num'] >= f_val[0]) & (dff['val_num'] <= f_val[1])]
        dff = dff[(dff['age'] >= f_age[0]) & (dff['age'] <= f_age[1])]

        df_transfer = dff[dff['son_karar_metni'].str.contains(r"Transfer|A\+|A \(", na=False)]
        df_watch = dff[dff['son_karar_metni'].str.contains(r"Takip|Gelişimi|B \(", na=False)]
        df_reject = dff[dff['son_karar_metni'].str.contains(r"Olumsuz|D \(", na=False)]

        t1, t2, t3 = st.tabs([f"✅ Transfer ({len(df_transfer)})", f"⏳ İzleme ({len(df_watch)})", f"❌ Olumsuz ({len(df_reject)})"])

        with t1: st.dataframe(df_transfer[['club_logo', 'name', 'team', 'oyun_tarzi', 'age', 'foot', 'ortalama_puan', 'son_karar_metni']], use_container_width=True, column_config={"club_logo": st.column_config.ImageColumn("Kulüp", width="small"), "ortalama_puan": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100)})
        with t2: st.dataframe(df_watch[['club_logo', 'name', 'team', 'oyun_tarzi', 'age', 'foot', 'ortalama_puan', 'son_karar_metni']], use_container_width=True, column_config={"club_logo": st.column_config.ImageColumn("Kulüp", width="small"), "ortalama_puan": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100)})
        with t3: st.dataframe(df_reject[['club_logo', 'name', 'team', 'oyun_tarzi', 'age', 'foot', 'ortalama_puan']], use_container_width=True, column_config={"club_logo": st.column_config.ImageColumn("Kulüp", width="small")})
            
# ==============================================================================
# SAYFA: VERİ HAVUZU
# ==============================================================================
elif page == "Veri Havuzu":
    st.title("📂 Oyuncu Veri Havuzu")
    
    with st.expander("ℹ️ Bu Sayfa Ne İşe Yarar?", expanded=False):
        st.write("Veritabanındaki **tüm** oyuncuları (henüz rapor girilmemiş olanlar dahil) burada listeleyebilir, boy ve yaşa göre filtreleyebilirsiniz.")

    df = db.get_all_players_detailed()
    if df.empty: st.warning("Veri yok.")
    else:
        df['val_num'] = df['market_value'].apply(parse_market_value)
        df['contract_year'] = df['contract'].apply(extract_year)

        def color_decision(val):
            if 'Transfer' in str(val): return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
            elif 'Takip' in str(val): return 'background-color: #fef9c3; color: #854d0e'
            elif 'Olumsuz' in str(val): return 'background-color: #fee2e2; color: #991b1b'
            return ''

        with st.expander("🔍 Detaylı Filtreleme", expanded=True):
            # Ayak, Sözleşme Yılı ve Sıralama Rolü filtreleri kaldırıldı, üst alan 2 sütuna düşürüldü
            c1, c2 = st.columns(2)
            f_pos = c1.multiselect("Mevki", df['position'].unique())
            # 'Takım Stili' -> 'Oyun Tarzı' olarak güncellendi
            f_style = c2.multiselect("Oyun Tarzı", df['oyun_tarzi'].dropna().unique())

            c3, c4 = st.columns(2)
            f_val = c3.slider("Piyasa Değeri Aralığı (€m)", 0.0, float(df['val_num'].max()), (0.0, 100.0))
            f_height = c4.slider("Boy (cm)", 160, 210, (160, 210))

        dff = df.copy()
        if f_pos: dff = dff[dff['position'].isin(f_pos)]
        if f_style: dff = dff[dff['oyun_tarzi'].isin(f_style)]
        # Gereksiz filtre kontrolleri temizlendi
        dff = dff[(dff['val_num'] >= f_val[0]) & (dff['val_num'] <= f_val[1])]
        dff = dff[(dff['height'] >= f_height[0]) & (dff['height'] <= f_height[1])]

        # Sıralama Rolü kaldırıldığı için standart puan (ortalama_puan) kullanılıyor
        score_col = "Skor"
        dff = dff.rename(columns={'ortalama_puan': score_col, 'son_karar_metni': 'Karar', 'oyun_tarzi': 'Oyun Tarzı'})

        st.dataframe(dff[['id', 'club_logo', 'name', 'team', 'Oyun Tarzı', 'position', 'age', 'foot', 'height', score_col, 'Karar']].style.map(color_decision, subset=['Karar']), use_container_width=True, height=600, column_config={"club_logo": st.column_config.ImageColumn("Logo", width="small"), score_col: st.column_config.ProgressColumn("Puan", min_value=0, max_value=100, format="%.1f")})
        
# ==============================================================================
# SAYFA: OYUNCU PROFİLİ
# ==============================================================================
elif page == "Oyuncu Profili":
    with st.expander("ℹ️ Oyuncu Profili Rehberi"):
        st.write("Bir oyuncuyu seçin. Sistem, o oyuncunun scout notlarını ve grafiklerini (Pizza Chart, Şut Haritası) otomatik getirecektir.")

    df_all = db.get_all_players_detailed()
    df_flat = db.get_flattened_data()
    if df_all.empty: st.warning("Veri yok."); st.stop()
    p_list = df_all.apply(lambda x: f"{x['name']} - {x['team']} (ID:{x['id']})", axis=1).tolist()
    sel_p = st.selectbox("Oyuncu Ara:", p_list)
    pid = int(sel_p.split("ID:")[-1].replace(')', ''))
    p_data, p_reps = db.get_player_full_profile(pid)
    col_name = 'ortalama_puan' if 'ortalama_puan' in df_all.columns else 'son_puan'
    avg_score = df_all[df_all['id']==pid][col_name].values[0]
    c_logo = get_team_logo(p_data['team'], p_data.get('club_logo'))

    st.markdown(f"""
    <div class="player-card">
        <div class="card-img-container"><img src="{p_data['image_url']}" class="card-img"><img src="{c_logo}" class="card-logo"></div>
        <div class="card-info">
            <h1>{p_data['name']}</h1>
            <div class="card-meta"><span>📍 {p_data['team']}</span><span>⚡ {p_data['position']}</span><span>🎂 {p_data['age']} Yaş</span><span>🦶 {p_data['foot']}</span><span>📏 {p_data.get('height','-')}cm</span></div>
            <div class="stat-badges"><span class="card-badge score-badge">Skor: {avg_score:.1f}</span><span class="card-badge">💰 {p_data['market_value']}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not p_reps.empty:
        last_rep = p_reps.iloc[0]
        try: r_vals = json.loads(last_rep['ratings_json']); m_vals = json.loads(last_rep['match_data_json']); general_data = json.loads(last_rep['general_game'])
        except: r_vals, m_vals, general_data = {}, {}, {}

        st.subheader("📝 Scout Analiz Notları")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='comment-box'><strong>⚽ Toplu Oyun:</strong><br>{last_rep['on_ball']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='comment-box'><strong>➕ Artılar:</strong><br>{last_rep['pros']}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='comment-box'><strong>🏃 Topsuz Oyun:</strong><br>{last_rep['off_ball']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='comment-box'><strong>➖ Eksiler:</strong><br>{last_rep['cons']}</div>", unsafe_allow_html=True)

        tab_pizza, tab_bee, tab_shot, tab_radar = st.tabs(["🍕 Pizza Chart", "🐝 Lig Konumu", "🎯 Şut Haritası", "🕸️ Rol Radarı"])
        with tab_pizza:
            if m_vals and not df_flat.empty:
                pos_df = df_flat[df_flat['position'] == p_data['position']]
                metrics = list(m_vals.keys())[:12]
                if 'performance_index' in metrics: metrics.remove('performance_index')
                values = [m_vals.get(m, 0) for m in metrics]
                percentiles = [min(99, int(stats.percentileofscore(pos_df[m].dropna(), v))) if m in pos_df.columns else 50 for m, v in zip(metrics, values)]
                fig_pizza = create_pizza_chart(p_data['name'], metrics, values, percentiles)
                st.pyplot(fig_pizza, use_container_width=False)
        with tab_bee:
            if m_vals and not df_flat.empty:
                valid_metrics = [c for c in df_flat.select_dtypes(include=np.number).columns if c not in ['id','player_id','age']]
                sel_metric = st.selectbox("İncelenecek Metrik", valid_metrics, index=0)
                fig_bee = create_beeswarm_plot(df_flat, sel_metric, p_data['name'], m_vals.get(sel_metric, 0))
                st.pyplot(fig_bee, use_container_width=True)
        with tab_shot:
            fig_pitch = draw_shot_map(p_data['name'], p_data['position'], 20, 5, "#ef4444")
            st.pyplot(fig_pitch)
        with tab_radar:
            avail_roles = logic.get_available_roles(p_data['position'])
            viz_role = st.selectbox("Rol", avail_roles, key="radar_role")
            key_attrs = logic.get_role_key_attributes(p_data['position'], viz_role)
            chart_data = []
            for k, v in r_vals.items():
                imp = "Kritik" if any(ka in k for ka in key_attrs) else "Standart"
                chart_data.append({"Özellik": k, "Puan": v, "Önem": imp})
            df_chart = pd.DataFrame(chart_data).sort_values("Puan")
            fig_bar = px.bar(df_chart, y="Özellik", x="Puan", color="Önem", orientation='h', height=600, color_discrete_map={"Kritik": "#22c55e", "Standart": "#cbd5e1"})
            fig_bar.update_layout(**WHITE_BG)
            st.plotly_chart(fig_bar, use_container_width=True, config=PLOT_CONFIG)
    
    st.caption("Dipnot: Buradaki veriler gerçek veriler olmadığından simülasyon ile oluşturulmuştur.")

# ==============================================================================
# SAYFA: ANALİZ DASHBOARD
# ==============================================================================
elif page == "Analiz Dashboard":
    st.title("📊 Veri Analiz Merkezi")
    
    with st.expander("ℹ️ Analiz Rehberi"):
        st.write("X ve Y eksenlerine istediğiniz iki veriyi (Örn: Şut ve Gol) koyarak oyuncuların ligdeki dağılımını (Scatter Plot) görebilirsiniz.")

    df_flat = db.get_flattened_data()

    if not df_flat.empty:
        with st.expander("🔍 Önce Filtrele", expanded=True):
            col1, col2 = st.columns(2)
            fil_pos = col1.multiselect("Mevki", df_flat['position'].unique())
            fil_team = col2.multiselect("Takım", df_flat['team'].unique())

            if fil_pos: df_flat = df_flat[df_flat['position'].isin(fil_pos)]
            if fil_team: df_flat = df_flat[df_flat['team'].isin(fil_team)]

            c1, c2 = st.columns(2)
            numeric_cols = [c for c in df_flat.select_dtypes(include=np.number).columns if c not in ['id','player_id','final_score']]
            x = c1.selectbox("X Ekseni", numeric_cols, index=0)
            y = c2.selectbox("Y Ekseni", numeric_cols, index=1)

        if st.button("Grafiği Oluştur"):
            st.subheader("📷 Oyuncu Kıyaslama")
            if len(df_flat) > 50: st.warning(f"⚠️ {len(df_flat)} oyuncu gösteriliyor.")
            fig_img = create_image_scatter_plot(df_flat, x, y, "image_url", f"{x} vs {y}")
            st.plotly_chart(fig_img, use_container_width=True)

# ==============================================================================
# SAYFA: TAKIM ANALİZİ
# ==============================================================================
elif page == "Takım Analizi":
    st.title("🏢 Takım Analizi")
    st.info("9. Adım: Ligin analizini yaparak takımların oyun tarzlarını keşfedebilirsiniz.")
    
    teams_df = db.get_team_stats()
    if not teams_df.empty:
        df_players = db.get_all_players_detailed()
        logo_map = {row['team']: row['club_logo'] for idx, row in df_players.iterrows() if row['club_logo']}
        teams_df['logo'] = teams_df['name'].map(logo_map)

        sel = st.selectbox("Takım Seç", teams_df['name'])
        row = teams_df[teams_df['name']==sel].iloc[0]
        t_logo = row['logo'] if 'logo' in row and row['logo'] else "https://tmssl.akamaized.net/images/wappen/head/default.png"

        st.markdown(f"""<div class="player-card"><div class="card-img-container"><img src="{t_logo}" style="width:100px; height:100px; object-fit:contain;"></div><div class="card-info"><h1>{row['name']}</h1><div class="card-meta"><span>{row['formation']}</span> | <span>{row['game_style']}</span></div></div></div>""", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Topla Oynama", f"%{row['avg_possession']}")
        c2.metric("xG", row['xg_for'])
        c3.metric("xGA", row['xg_against'])
        c4.metric("PPDA", row['ppda'])

        st.divider()
        st.subheader("📊 Lig Analizi (Logolu Scatter)")
        cols = ["avg_possession", "ppda", "xg_for", "xg_against"]
        cx, cy = st.columns(2)
        x_m = cx.selectbox("X Ekseni", cols, index=0)
        y_m = cy.selectbox("Y Ekseni", cols, index=2)

        if st.button("Grafiği Getir"):
            fig_team = create_image_scatter_plot(teams_df, x_m, y_m, "logo", f"{x_m} vs {y_m}")
            st.plotly_chart(fig_team, use_container_width=True)

# ==============================================================================
# YENİ SAYFA: MAÇ MERKEZİ (SİMÜLASYON BOTU)
# ==============================================================================
elif page == "Maç Merkezi":
    st.title("🏟️ Maç Merkezi & Simülasyon")
    st.info("10. Adım: Maç merkezi ekranından oynanan örnek bir maçın grafiklerini görebilirsiniz. (Veriler simülasyon ürünüdür)")
    
    with st.expander("ℹ️ Simülasyon Nasıl Çalışır?"):
        st.write("Sistem, veri havuzundaki gerçek oyuncuları ve takımları kullanarak sanal bir lig oynatır. Skorlar ve oyuncu performansları, takımların güç dengesine göre **Yapay Zeka (Poisson Dağılımı)** ile hesaplanır.")

    # 1. VERİ YÜKLEME VE İŞLEME (SİMÜLASYON)
    @st.cache_data
    def load_match_data():
        return generate_simulated_league_data()

    league_data = load_match_data()
    if league_data.empty: 
        st.warning("Veritabanında yeterli takım veya oyuncu bulunamadı. Lütfen önce 'Oyuncu Ekle' menüsünden veri çekin.")
        st.stop()

    # 2. SIDEBAR SEÇİMLERİ
    with st.sidebar:
        st.header("Maç Ayarları")
        team_list = sorted(list(set(league_data['Team'].unique())))
        team = st.selectbox('Takım Seçiniz', team_list)

        specific = st.radio('Görünüm', ('Belirli Bir Maç', 'Son Maçlar'))
        
        team_matches = league_data[(league_data['Team'] == team)].copy()
        team_matches['Match_Name'] = team_matches['Match'] + ' (' + team_matches['Date'].dt.date.astype(str) + ')'
        
        render_matches = []
        if specific == 'Belirli Bir Maç':
            if not team_matches.empty:
                match_choice = st.selectbox('Maç Seçiniz', team_matches['Match_Name'].unique())
                selected_match_row = team_matches[team_matches['Match_Name'] == match_choice].iloc[0]
                render_matches = [selected_match_row['Match']]
        else:
            if not team_matches.empty:
                num_matches = st.slider('Son Kaç Maç?', 1, 5, 3)
                latest_matches = team_matches.sort_values('Date', ascending=False)['Match'].unique()[:num_matches]
                render_matches = latest_matches.tolist()

        focal_color = st.color_picker("Grafik Vurgu Rengi", "#4c94f6")

    # 3. VERİ HAZIRLIĞI
    team_data = league_data[league_data['Team'] == team].reset_index(drop=True)
    
    available_vars = [
        'Possession', 'xG', 'xGA', 'xGD', 'Goals', 'Goals Conceded',
        'Shots', 'xT', 'PPDA', 'xT Difference'
    ]
    # Sadece veride olanları al
    available_vars = [v for v in available_vars if v in league_data.columns]

    # 4. TABS
    tabs = st.tabs(['📝 Maç Raporu', '🏆 Puan Durumu', '📈 Grafikler', '📊 Sıralamalar', '⚽ xG/xT Scatter'])

    # --- TAB 1: MAÇ RAPORU ---
    with tabs[0]:
        for match_name in render_matches:
            st.subheader(f"⚽ {match_name}")
            m_data = league_data[league_data['Match'] == match_name]
            
            if not m_data.empty:
                # Eşleşen takımları bul
                teams_in_match = m_data['Team'].unique()
                if len(teams_in_match) == 2:
                    row1 = m_data.iloc[0]
                    t1 = row1['Team']
                    t2 = row1['Opponent']
                    
                    d1 = m_data[m_data['Team'] == t1].iloc[0]
                    d2 = m_data[m_data['Team'] == t2].iloc[0]
                    
                    # LOGOLAR ve SKOR
                    c1, c2, c3 = st.columns([1, 0.2, 1])
                    with c1:
                        if d1['Logo']: st.image(d1['Logo'], width=80)
                        st.markdown(f"### {t1}")
                        st.markdown(f"<h1 style='color:{focal_color if t1==team else 'black'}'>{int(d1['Goals'])}</h1>", unsafe_allow_html=True)
                        st.caption(f"xG: {d1['xG']}")
                        if d1['Scorers'] != "-":
                            st.markdown(f"**⚽ Goller:** {d1['Scorers']}")
                            
                    with c2:
                        st.markdown("<h1 style='text-align:center; margin-top: 50px;'>VS</h1>", unsafe_allow_html=True)
                        
                    with c3:
                        if d2['Logo']: st.image(d2['Logo'], width=80)
                        st.markdown(f"### {t2}")
                        st.markdown(f"<h1 style='color:{focal_color if t2==team else 'black'}'>{int(d2['Goals'])}</h1>", unsafe_allow_html=True)
                        st.caption(f"xG: {d2['xG']}")
                        if d2['Scorers'] != "-":
                            st.markdown(f"**⚽ Goller:** {d2['Scorers']}")
                    
                    st.divider()
                    st.info(f"🌟 **Maçın Adamı:** {d1['MOTM']}")

    # --- TAB 2: PUAN DURUMU ---
    with tabs[1]:
        start_date = st.date_input("Başlangıç", value=league_data['Date'].min(), format="YYYY-MM-DD")
        end_date = st.date_input("Bitiş", value=league_data['Date'].max(), format="YYYY-MM-DD")
        st.dataframe(table_start_end(league_data, start_date, end_date), hide_index=True)

    # --- TAB 3: GRAFİKLER ---
    with tabs[2]:
        plot_type = st.radio("Grafik Tipi", ['📈 Çizgi', '📊 Çubuk'], horizontal=True)
        var = st.selectbox('Metrik Seçiniz', available_vars)
        mov_avg = st.checkbox("4 Maçlık Hareketli Ortalama Ekle")
        
        team_data_plot = add_mov_avg(team_data, var)
        dates = team_data_plot['Date']
        values = team_data_plot[var]
        lg_avg = league_data[var].mean()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#fbf9f4')
        ax.set_facecolor('#fbf9f4')
        
        if plot_type == '📈 Çizgi':
            ax.plot(dates, values, marker='o', color=focal_color, label=var)
            ax.axhline(y=lg_avg, color='#ee5454', linestyle='-', alpha=0.5, label='Lig Ort.')
            if mov_avg:
                ax.plot(dates, team_data_plot['4-Match Moving Average'], linestyle='--', color='#4a2e19', label='4 Maç Ort.')
        else:
            colors = [focal_color if x >= 0 else '#ee5454' for x in values]
            ax.bar(dates, values, color=colors, label=var)
            ax.axhline(y=lg_avg, color='#ee5454', linestyle='-', alpha=0.5, label='Lig Ort.')
            if mov_avg:
                ax.plot(dates, team_data_plot['4-Match Moving Average'], linestyle='--', color='#4a2e19', label='4 Maç Ort.')
        
        ax.set_title(f"{team} - {var} Performansı")
        plt.xticks(rotation=45)
        ax.legend()
        st.pyplot(fig)

    # --- TAB 4: SIRALAMALAR ---
    with tabs[3]:
        rank_var = st.selectbox('Sıralanacak Metrik', available_vars, key='rank_var')
        rank_method = st.radio("Hesaplama", ['Ortalama', 'Toplam', 'Medyan'], horizontal=True)
        
        if rank_method == 'Toplam':
            rank_df = league_data.groupby(['Team'])[rank_var].sum().reset_index()
        elif rank_method == 'Medyan':
            rank_df = league_data.groupby(['Team'])[rank_var].median().reset_index()
        else:
            rank_df = league_data.groupby(['Team'])[rank_var].mean().reset_index()
            
        rank_df = rank_df.sort_values(by=rank_var, ascending=False).reset_index(drop=True)
        rank_df.index += 1
        
        def highlight_team(row):
            return ['background-color: #dcfce7' if row['Team'] == team else '' for _ in row]

        st.dataframe(rank_df.style.apply(highlight_team, axis=1), use_container_width=True)

    # --- TAB 5: SCATTER ---
    with tabs[4]:
        scatter_mode = st.radio("Analiz Modu", ['xG Analizi', 'xT Analizi', 'Genel Değişkenler'], horizontal=True)
        
        if scatter_mode == 'xG Analizi':
            x_col, y_col = 'xG', 'xGA'
        elif scatter_mode == 'xT Analizi':
            x_col, y_col = 'xT', 'xT Difference'
        else:
            c_s1, c_s2 = st.columns(2)
            x_col = c_s1.selectbox("X Ekseni", available_vars, index=0)
            y_col = c_s2.selectbox("Y Ekseni", available_vars, index=1)
            
        fig_scatter = create_image_scatter_plot(league_data, x_col, y_col, "Logo", f"Lig Geneli {x_col} vs {y_col}")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.caption("Dipnot: Buradaki veriler gerçek veriler olmadığından simülasyon ile oluşturulmuştur.")

# ==============================================================================
# SAYFA: KIYASLAMA
# ==============================================================================
elif page == "Kıyaslama":
    st.title("⚖️ Oyuncu Kıyaslama")
    
    with st.expander("ℹ️ Nasıl Kullanılır?"):
        st.write("İki farklı oyuncuyu seçerek hücum ve pas metriklerini yan yana kıyaslayın.")

    df = db.get_all_players_detailed()
    if not df.empty:
        p_list = df.apply(lambda x: f"{x['name']} (ID:{x['id']})", axis=1).tolist()
        c1,c2 = st.columns(2)
        p1_sel = c1.selectbox("Oyuncu 1", p_list, key="p1")
        p2_sel = c2.selectbox("Oyuncu 2", p_list, key="p2")

        all_metrics = logic.MATCH_DATA_METRICS["Hücum"] + logic.MATCH_DATA_METRICS["Pas"]
        sel_metrics = st.multiselect("Metrikleri Seç", all_metrics, default=["xG", "Shots", "Key_passes", "Dribbles_successful"])

        if st.button("Analiz Et"):
            p1_id = int(p1_sel.split("ID:")[-1].replace(')', ''))
            p2_id = int(p2_sel.split("ID:")[-1].replace(')', ''))
            d1, r1 = db.get_player_full_profile(p1_id)
            d2, r2 = db.get_player_full_profile(p2_id)

            k1, k2 = st.columns(2)
            with k1:
                l1 = d1.get('club_logo') or "https://tmssl.akamaized.net/images/wappen/head/default.png"
                st.markdown(f"""<div style="text-align:center;"><img src="{d1['image_url']}" width="100" style="border-radius:50%;"><br><h3>{d1['name']}</h3></div>""", unsafe_allow_html=True)
            with k2:
                l2 = d2.get('club_logo') or "https://tmssl.akamaized.net/images/wappen/head/default.png"
                st.markdown(f"""<div style="text-align:center;"><img src="{d2['image_url']}" width="100" style="border-radius:50%;"><br><h3>{d2['name']}</h3></div>""", unsafe_allow_html=True)

            if r1.empty or r2.empty:
                st.error("Rapor verisi eksik.")
            else:
                try:
                    m1 = json.loads(r1.iloc[0]['match_data_json'])
                    m2 = json.loads(r2.iloc[0]['match_data_json'])
                    data = []
                    for m in sel_metrics:
                        data.append({"Metrik": m, "Değer": m1.get(m,0), "Oyuncu": d1['name']})
                        data.append({"Metrik": m, "Değer": m2.get(m,0), "Oyuncu": d2['name']})

                    fig = px.bar(pd.DataFrame(data), x="Değer", y="Metrik", color="Oyuncu", barmode="group", orientation='h', color_discrete_sequence=['#22c55e', '#3b82f6'])
                    fig.update_layout(WHITE_BG)
                    st.plotly_chart(fig, use_container_width=True)
                except: st.error("JSON Hatası")

# ==============================================================================
# SAYFA: OYUNCU EKLE
# ==============================================================================
elif page == "Oyuncu Ekle":
    st.title("🌐 Oyuncu Ekle")
    st.info("ℹ️ Buradan Transfermarkt linkleri ile veri tabanınızı büyütebilirsiniz.")
    
    tab1, tab2 = st.tabs(["Link (Manuel)", "Lig Botu (Otomatik)"])
    with tab1:
        st.write("11. Adım: Link ile Tekil Oyuncu Ekleme")
        url = st.text_input("Transfermarkt Profil Linki")
        if st.button("Çek"):
            with st.spinner("Çekiliyor..."):
                d = sc.scrape_tm_profile_detailed(url)
                db.bulk_save_player(d)
                st.success("OK")
    with tab2:
        st.write("1. Adım: Lig Botu ile Toplu Veri Çekme")
        sel_lg = st.selectbox("Lig Seç", list(sc.LEAGUE_CODES.keys()))
        c_s1, c_s2 = st.columns(2)
        with c_s1: limit_tm = st.slider("Takım Sayısı", 1, 20, 3)
        with c_s2: limit_pl = st.slider("Takım Başı Oyuncu", 1, 40, 10)
        if st.button("🚀 Botu Başlat"):
            teams = sc.get_teams_from_league(sc.LEAGUE_CODES[sel_lg])
            teams_to_scan = teams[:limit_tm] if teams else []
            if teams_to_scan:
                bar = st.progress(0)
                for i, t in enumerate(teams_to_scan):
                    pls = sc.get_players_from_team(t.get('id'))
                    for p in pls[:limit_pl]:
                        link = f"https://www.transfermarkt.com/player/profil/spieler/{p.get('id')}"
                        d = sc.scrape_tm_profile_detailed(link, {'name': p.get('name'), 'team': t.get('name')})
                        d['league'] = sel_lg
                        db.bulk_save_player(d)
                    bar.progress((i+1)/len(teams_to_scan))
                st.success("Bitti.")

# ==============================================================================
# SAYFA: RAPOR OLUŞTUR
# ==============================================================================
elif page == "Rapor Oluştur":
    st.title("📝 Detaylı Rapor Girişi")
    
    with st.expander("ℹ️ Raporlama Rehberi"):
        st.write("5. Adım: Scout ekibi bu ekranı kullanarak maç izlenimlerini ve verilerini sisteme girer.")

    df = db.get_all_players_detailed()
    if not df.empty:
        p_list = df.apply(lambda x: f"{x['name']} (ID:{x['id']})", axis=1).tolist()
        sel_p = st.selectbox("Oyuncu Seç", p_list)
        pid = int(sel_p.split("ID:")[-1].replace(')', ''))
        p_data, _ = db.get_player_full_profile(pid)
        c_logo = get_team_logo(p_data['team'], p_data.get('club_logo'))

        st.markdown(f"""
        <div class="player-card" style="margin-bottom:20px;">
            <div class="card-img-container"><img src="{p_data['image_url']}" class="card-img"><img src="{c_logo}" class="card-logo"></div>
            <div class="card-info">
                <h1>{p_data['name']}</h1>
                <div class="card-meta"><span>📍 {p_data['team']}</span><span>⚡ {p_data['position']}</span><span>🎂 {p_data['age']} Yaş</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("rep"):
            c_r1, c_r2 = st.columns(2)
            role = c_r1.selectbox("Oyuncu Rolü", logic.get_available_roles(p_data['position']))
            match_lvl = c_r2.selectbox("Maç Seviyesi", ["Yüksek", "Orta", "Düşük"])

            tab_s, tab_d, tab_txt = st.tabs(["Scout", "Veri", "Yorumlar"])
            ratings = {}
            with tab_s:
                attrs = logic.get_scout_attributes(p_data['position'])
                cols = st.columns(3)
                for i, att in enumerate(attrs):
                    with cols[i%3]: ratings[att] = st.slider(att, 1, 10, 5, key=f"r_{i}")

            match_metrics = {}
            with tab_d:
                metrics = logic.get_match_metrics(p_data['position'])
                cols = st.columns(3)
                for i, m in enumerate(metrics):
                    with cols[i%3]: match_metrics[m] = st.number_input(m, min_value=0.0, step=0.1, key=f"m_{i}")

            with tab_txt:
                foot = st.selectbox("Ayak", ["Sağ", "Sol", "Çift"], index=0 if p_data['foot'] == "-" else ["Sağ", "Sol", "Çift"].index(p_data['foot']) if p_data['foot'] in ["Sağ", "Sol", "Çift"] else 0)
                c_on, c_off = st.columns(2)
                on_ball = c_on.text_area("Toplu Oyun")
                off_ball = c_off.text_area("Topsuz Oyun")
                c_pro, c_con = st.columns(2)
                pros = c_pro.text_area("Artılar (+)")
                cons = c_con.text_area("Eksiler (-)")
                st.markdown("#### Detaylı Analiz")
                c_d1, c_d2 = st.columns(2)
                txt_phy = c_d1.text_area("Fiziksel")
                txt_men = c_d2.text_area("Mental")
                txt_tac = c_d1.text_area("Taktiksel")
                txt_tec = c_d2.text_area("Teknik")
                detailed = {"Fiziksel": txt_phy, "Mental": txt_men, "Taktiksel": txt_tac, "Teknik": txt_tec}

            if st.form_submit_button("✅ Kaydet"):
                match_metrics['performance_index'] = 70
                score = logic.calculate_pro_score(ratings, match_metrics, p_data['position'], role, p_data['age'], match_lvl)
                dec = logic.get_detailed_decision(score, p_data['age'])
                txt_data = {'style': role, 'pros': pros, 'cons': cons, 'on_ball': on_ball, 'off_ball': off_ball, 'general': json.dumps(detailed), 'dec': dec}
                db.save_report(pid, "Analist", {'info': "Man", 'level': match_lvl}, ratings, match_metrics, score, txt_data)
                st.balloons()

                st.success(f"Puan: {score:.1f} | Karar: {dec}")


