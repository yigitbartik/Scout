import requests
from bs4 import BeautifulSoup
import time
import random
import re

# --- AYARLAR ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.transfermarkt.com/',
    'Connection': 'keep-alive'
}

# LİG KODLARI
LEAGUE_CODES = {
    "Süper Lig (TR)": "TR1",
    "1. Lig (TR)": "TR2",
    "Premier League (ENG)": "GB1",
    "La Liga (ESP)": "ES1",
    "Bundesliga (GER)": "L1",
    "Serie A (ITA)": "IT1",
    "Ligue 1 (FRA)": "FR1",
    "Eredivisie (NED)": "NL1",
    "Liga Portugal (POR)": "PO1",
    "Saudi Pro League (KSA)": "SA1"
}

def clean_txt(text):
    if not text: return ""
    return text.strip().replace('\n', '').replace('&nbsp;', ' ')

def get_teams_from_league(league_code):
    """Ligdeki tüm takımları çeker."""
    url = f"https://www.transfermarkt.com/quickselect/teams/{league_code}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
        return []
    except: return []

def get_players_from_team(team_id):
    """Takımdaki tüm oyuncuları çeker."""
    url = f"https://www.transfermarkt.com/quickselect/players/{team_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
        return []
    except: return []

def scrape_tm_profile_detailed(url, basic_info=None):
    """
    ID YÖNTEMİYLE LOGO VE VERİ ÇEKME
    """
    data = {
        "success": False, "url": url,
        "name": basic_info.get('name', 'Bilinmiyor') if basic_info else 'Bilinmiyor',
        "team": basic_info.get('team', '-') if basic_info else '-',
        "league": "-", "position": "Orta Saha",
        "nationality": "Dünya", "age": 20, "foot": "Sağ",
        "value": "-", "contract": "-",
        "image": "https://tmssl.akamaized.net/images/portrait/header/default.jpg",
        "club_logo": "https://tmssl.akamaized.net/images/wappen/head/default.png"
    }

    if "transfermarkt" not in url: return data

    try:
        time.sleep(random.uniform(0.4, 0.8)) # Bot koruması için bekleme
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200: return data

        soup = BeautifulSoup(r.content, 'html.parser')
        data['success'] = True

        # 1. HEADER ANALİZİ
        h1 = soup.find('h1', class_='data-header__headline-wrapper')
        if h1: data['name'] = clean_txt(h1.get_text().replace('#', ''))

        meta_img = soup.find('meta', property='og:image')
        if meta_img: data['image'] = meta_img['content']

        # 2. KULÜP LOGOSU (EN ÖNEMLİ KISIM: ID YÖNTEMİ)
        # Header içindeki kulüp linkini buluyoruz. Link genelde şöyledir: /galatasaray-istanbul/startseite/verein/141
        header_box = soup.find('div', class_='data-header__box--big')

        if header_box:
            club_link_tag = header_box.find('a', href=re.compile(r'/verein/\d+'))

            if club_link_tag:
                href = club_link_tag['href']
                # Linkin içinden sayıyı (ID) çekiyoruz (Örn: 141)
                match_id = re.search(r'/verein/(\d+)', href)

                if match_id:
                    club_id = match_id.group(1)
                    # ID'yi resmi sunucu linkine yapıştırıyoruz. BU ASLA BOZULMAZ.
                    data['club_logo'] = f"https://tmssl.akamaized.net/images/wappen/head/{club_id}.png"

                # Takım ismini de buradan alalım, daha temiz gelir
                img_tag = club_link_tag.find('img')
                if img_tag:
                    data['team'] = img_tag.get('title') or img_tag.get('alt') or data['team']

            # Lig ve Sözleşme
            league_link = header_box.find('a', class_='data-header__league-link')
            if league_link: data['league'] = clean_txt(league_link.get_text())

            contract_label = header_box.find('span', string=re.compile('Contract expires|Sözleşme sonu'))
            if contract_label:
                contract_val = contract_label.find_next('span', class_='data-header__content')
                if contract_val: data['contract'] = clean_txt(contract_val.get_text())

        # 3. PİYASA DEĞERİ
        val_box = soup.find('a', class_='data-header__market-value-wrapper')
        if val_box:
            raw_val = clean_txt(val_box.get_text())
            data['value'] = raw_val.split('Last')[0].strip()

        # 4. KİŞİSEL BİLGİLER
        header_details = soup.find('div', class_='data-header__details')
        if header_details:
            birth_span = header_details.find('span', itemprop='birthDate')
            if birth_span:
                txt = birth_span.get_text()
                match = re.search(r'\((\d+)\)', txt)
                if match: data['age'] = int(match.group(1))

            labels = header_details.find_all('li', class_='data-header__label')
            for label in labels:
                txt = label.get_text()
                content = label.find('span', class_='data-header__content')
                if not content: continue
                c_text = clean_txt(content.get_text())

                if "Position" in txt or "Mevki" in txt: data['position'] = c_text
                elif "Citizenship" in txt or "Uyruk" in txt: data['nationality'] = c_text
                elif "Foot" in txt or "Ayak" in txt: data['foot'] = c_text

    except Exception as e:
        print(f"Scraper Hatası ({url}): {e}")

    return data