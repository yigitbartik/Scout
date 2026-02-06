# scout_logic.py - FİNAL V4

# --- 1. VERİ SÖZLÜKLERİ ---
MATCH_DATA_METRICS = {
    "Hücum": ["Goals", "xG", "Shots", "Shots_on_target", "Dribbles_successful", "Touches_box", "Big_chances", "Progressive_runs"],
    "Pas": ["Pass_accuracy", "Key_passes", "Progressive_passes", "Long_pass_accuracy", "Smart_passes", "Through_balls", "Crosses_completed"],
    "Savunma": ["Tackles_won", "Interceptions", "Duels_won_defensive", "Aerial_won", "Recoveries", "Blocks", "Clearances", "Pressures_successful"],
    "Fiziksel": ["Sprint_count", "Distance_covered", "Top_speed", "High_intensity_runs", "Accelerations", "Decelerations"]
}

GK_MATCH_METRICS = ["Saves", "Save_percentage", "xG_prevented", "Aerial_claims", "Sweeper_actions", "Pass_accuracy_long", "Reflex_saves"]

# --- 2. TAKIM OYUN TARZLARI ---
TEAM_GAME_STYLES = [
    "Bilinmiyor",
    "Dominant / Topa Sahip Olma",
    "Gegenpress / Yüksek Baskı",
    "Kontra Atak / Direkt Oyun",
    "Katı Savunma / Derin Blok",
    "Kanat Organizasyonu",
    "Dengeli / Standart"
]

def determine_team_style(stats):
    """Verilere göre TAKIMIN oyun tarzını belirler."""
    poss = stats.get('avg_possession', 50)
    ppda = stats.get('ppda', 12)

    if poss >= 58: return "Dominant / Topa Sahip Olma"
    if ppda < 9: return "Gegenpress / Yüksek Baskı"
    if poss < 42: return "Katı Savunma / Derin Blok"
    if 42 <= poss < 50 and ppda > 12: return "Kontra Atak / Direkt Oyun"
    return "Dengeli / Standart"

# --- 3. SCOUT SORULARI ---
SCOUT_GENERAL_ATTRS = ["Karar Verme", "Oyun Zekası", "Pozisyon Alma", "Topsuz Oyun", "Mental Dayanıklılık", "Soğukkanlılık", "Agresiflik", "Çalışkanlık", "İletişim", "Teknik"]
SCOUT_GK_GENERAL = ["Refleks", "Shot Stop", "Bire Bir", "Yan Top", "Hava Hakimiyeti", "Ayak Kalitesi", "Pozisyon Alma", "İletişim", "Liderlik"]

POSITION_SPECIFIC_ATTRS = {
    "Stoper": ["Markaj", "Müdahale", "Hava Topu", "Oyun Kurma", "Uzun Pas", "Güç"],
    "Bek": ["Orta Kalitesi", "Bindirme", "Hız", "1v1 Savunma", "Dayanıklılık", "Kademe"],
    "Defansif Orta Saha": ["Top Kapma", "Pas Arası", "Pres", "Denge", "Kısa Pas", "Pozisyon Bilgisi"],
    "Merkez Orta Saha": ["Pas Vizyonu", "Dribbling", "Şut", "İkili Mücadele", "Tempo"],
    "Ofansif Orta Saha": ["Yaratıcılık", "Son Pas", "Bitiricilik", "Dar Alan", "Dribbling"],
    "Kanat": ["Hızlanma", "Çalım", "Orta", "İçe Kat Etme", "Patlayıcılık", "Ters Ayak"],
    "Forvet": ["Bitiricilik", "Kafa Vuruşu", "Sırtı Dönük", "Sezgi", "Soğukkanlılık", "Patlayıcılık"]
}

# --- 4. OYUNCU ROLLERİ ---
ROLE_CONFIG = {
    "Stoper": {
        "Standart": {"bonus": ["Markaj", "Müdahale"]},
        "Pasör (Ball Playing)": {"bonus": ["Oyun Kurma", "Uzun Pas", "Teknik"]},
        "Kesici (Stopper)": {"bonus": ["Markaj", "Müdahale", "Güç", "Agresiflik"]},
        "Sigorta (Sweeper)": {"bonus": ["Hız", "Sezgi", "Kademe"]}
    },
    "Bek": {
        "Standart": {"bonus": ["Hız", "Dayanıklılık"]},
        "Ofansif Bek (Wingback)": {"bonus": ["Orta Kalitesi", "Bindirme", "Hız", "Dribbling"]},
        "Defansif Bek": {"bonus": ["1v1 Savunma", "Pozisyon Alma", "Markaj"]},
        "İçe Kateden Bek": {"bonus": ["Pas Vizyonu", "Teknik", "Karar Verme"]}
    },
    "Orta Saha": {
        "Standart": {"bonus": ["Pas Vizyonu"]},
        "Oyun Kurucu (Playmaker)": {"bonus": ["Pas Vizyonu", "Karar Verme", "Teknik", "Uzun Pas"]},
        "Savaşçı (Box-to-Box)": {"bonus": ["Dayanıklılık", "İkili Mücadele", "Tempo", "Pres"]},
        "Maestro (Regista)": {"bonus": ["Soğukkanlılık", "Oyun Zekası", "Pas Vizyonu"]},
        "Kesici (Anchor)": {"bonus": ["Top Kapma", "Pozisyon Alma", "Güç"]}
    },
    "Kanat": {
        "Standart": {"bonus": ["Hız", "Orta"]},
        "Ters Ayaklı (Inverted)": {"bonus": ["Şut", "Bitiricilik", "İçe Kat Etme", "Dribbling"]},
        "Çizgi Oyuncusu": {"bonus": ["Hız", "Orta", "Hızlanma", "Dayanıklılık"]},
        "Yaratıcı Kanat": {"bonus": ["Yaratıcılık", "Pas Vizyonu", "Teknik"]}
    },
    "Forvet": {
        "Standart": {"bonus": ["Bitiricilik"]},
        "Pivot Santrfor": {"bonus": ["Sırtı Dönük", "Güç", "Hava Topu", "İletişim"]},
        "Fırsatçı Golcü (Poacher)": {"bonus": ["Bitiricilik", "Sezgi", "Reaksiyon", "Soğukkanlılık"]},
        "Sahte 9": {"bonus": ["Pas Vizyonu", "Teknik", "Derine Gelme"]},
        "Pressing Forward": {"bonus": ["Çalışkanlık", "Agresiflik", "Hız", "Dayanıklılık"]}
    },
    "Kaleci": {
        "Standart": {"bonus": ["Refleks"]},
        "Libero Kaleci": {"bonus": ["Ayak Kalitesi", "İletişim", "Oyun Kurma"]},
        "Çizgi Kalecisi": {"bonus": ["Refleks", "Shot Stop", "Bire Bir"]}
    }
}

# --- YARDIMCI FONKSİYONLAR ---
def normalize_position(tm_pos):
    p = str(tm_pos).lower()
    if "keeper" in p or "kaleci" in p: return "Kaleci"
    if "stoper" in p or "centre-back" in p: return "Stoper"
    if "back" in p or "bek" in p: return "Bek"
    if "defensive" in p or "ön libero" in p: return "Defansif Orta Saha"
    if "attacking" in p or "ofansif" in p or "on numara" in p: return "Ofansif Orta Saha"
    if "wing" in p or "kanat" in p: return "Kanat"
    if "striker" in p or "forvet" in p or "forward" in p: return "Forvet"
    return "Merkez Orta Saha"

def get_scout_attributes(position):
    pos = normalize_position(position)
    if pos == "Kaleci": return SCOUT_GK_GENERAL
    specific = POSITION_SPECIFIC_ATTRS.get(pos, POSITION_SPECIFIC_ATTRS["Merkez Orta Saha"])
    return list(set(SCOUT_GENERAL_ATTRS + specific))

def get_match_metrics(position):
    pos = normalize_position(position)
    if pos == "Kaleci": return GK_MATCH_METRICS
    l = []
    for k,v in MATCH_DATA_METRICS.items(): l.extend(v)
    return l

def get_available_roles(position):
    real_pos = normalize_position(position)
    if "Orta Saha" in real_pos: real_pos = "Orta Saha"
    return list(ROLE_CONFIG.get(real_pos, {"Standart":{}}).keys())

def get_role_key_attributes(position, role):
    real_pos = normalize_position(position)
    if "Orta Saha" in real_pos: real_pos = "Orta Saha"
    return ROLE_CONFIG.get(real_pos, {}).get(role, {}).get("bonus", [])

def calculate_pro_score(ratings, match_data, position, role, age, match_level):
    if not ratings: return 0

    # 1. Rol Bazlı Scout Puanı (Kritik özellikler x1.5)
    role_bonus = get_role_key_attributes(position, role)
    total_score = 0
    total_weight = 0

    for att, val in ratings.items():
        weight = 1.0
        if any(b in att for b in role_bonus): weight = 1.5
        total_score += val * weight
        total_weight += weight

    scout_score = (total_score / total_weight) * 10 if total_weight > 0 else 0

    # 2. Veri Puanı
    data_score = match_data.get('performance_index', 50)

    # 3. Ağırlıklı Ortalama (Scout %65, Veri %35)
    base_score = (scout_score * 0.65) + (data_score * 0.35)

    # 4. Katsayılar (Daha sıkı)
    coeff = 1.0
    if age < 23:
        if match_level == "Yüksek": coeff = 1.3
        elif match_level == "Orta": coeff = 1.15
    elif 23 <= age <= 28:
        if match_level == "Yüksek": coeff = 1.1
    else: # 28+
        if match_level == "Düşük": coeff = 0.9

    return min(100, base_score * coeff)

def get_detailed_decision(score, age):
    # Eşikler yükseltildi, herkes A+ olmasın
    if age < 23:
        if score >= 88: return "A+ (Wonderkid)"
        if score >= 78: return "A (Transfer)"
        if score >= 68: return "B (Takip)"
        if score >= 55: return "C (Gelişimi İzlenmeli)"
    else:
        if score >= 85: return "A+ (Yıldız)"
        if score >= 75: return "A (Fırsat)"
        if score >= 65: return "B (Alternatif)"

    return "D (Olumsuz)"