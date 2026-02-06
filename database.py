# database.py - FİNAL V4

import sqlite3
import json
import pandas as pd
import random
import scout_logic as logic
import threading

db_lock = threading.Lock()
DB_FILE = "scout_v2.db"

def get_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_lock:
        conn = get_connection()
        c = conn.cursor()
        # BOY (height) sütunu eklendi
        c.execute('''CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT, tm_link TEXT UNIQUE,
            name TEXT, team TEXT, league TEXT, position TEXT, nationality TEXT,
            age INTEGER, foot TEXT, height INTEGER, market_value TEXT, contract TEXT,
            image_url TEXT, club_logo TEXT,
            minutes_played INTEGER DEFAULT 0,
            matches_played INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        c.execute('''CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE,
            formation TEXT, game_style TEXT,
            avg_possession REAL, ppda REAL, xg_for REAL, xg_against REAL)''')

        c.execute('''CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT, player_id INTEGER, scout_name TEXT,
            match_info TEXT, match_level TEXT, ratings_json TEXT, match_data_json TEXT, final_score REAL,
            team_style TEXT, pros TEXT, cons TEXT,
            on_ball TEXT, off_ball TEXT, general_game TEXT, decision TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(player_id) REFERENCES players(id))''')
        conn.commit()
        conn.close()

def bulk_save_player(data):
    with db_lock:
        conn = get_connection()
        c = conn.cursor()
        try:
            link = data.get('url') if data.get('url') else f"auto_{data['name']}_{data['team']}"
            for k, v in data.items():
                if v is None: data[k] = "-"

            # Height sütunu eklendi
            c.execute('''INSERT INTO players (tm_link, name, team, league, position, nationality, age, foot, height, market_value, contract, image_url, club_logo)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                         ON CONFLICT(tm_link) DO UPDATE SET
                         team=excluded.team, market_value=excluded.market_value, image_url=excluded.image_url,
                         contract=excluded.contract, club_logo=excluded.club_logo, age=excluded.age,
                         nationality=excluded.nationality, foot=excluded.foot, height=excluded.height''',
                      (link, data['name'], data['team'], data['league'], data['position'],
                       data['nationality'], data['age'], data['foot'], data.get('height', 0),
                       data['value'], data['contract'], data['image'], data.get('club_logo', '')))
            conn.commit()
            return True
        except Exception as e:
            print(f"DB Kayıt Hatası: {e}")
            return False
        finally:
            conn.close()

def add_player_manual(data): return bulk_save_player(data)

def save_report(pid, sname, match_data_info, ratings, match_metrics, score, text_data, conn=None):
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        c = conn.cursor()
        gen = text_data.get('general', '')
        if isinstance(gen, dict): gen = json.dumps(gen)

        c.execute('''INSERT INTO reports
                     (player_id, scout_name, match_info, match_level, ratings_json, match_data_json, final_score,
                      team_style, pros, cons, on_ball, off_ball, general_game, decision)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                  (pid, sname, match_data_info['info'], match_data_info['level'],
                   json.dumps(ratings), json.dumps(match_metrics), score,
                   text_data.get('style', '-'), text_data.get('pros', '-'), text_data.get('cons', '-'),
                   text_data.get('on_ball', '-'), text_data.get('off_ball', '-'), gen,
                   text_data.get('dec', '-')))
        if should_close: conn.commit()
    except Exception as e:
        print(f"Rapor Hatası: {e}")
    finally:
        if should_close: conn.close()

def get_all_players_detailed():
    conn = get_connection()
    q = """
    SELECT p.id, p.name, p.team, p.position, p.age, p.nationality, p.foot, p.height, p.market_value, p.league,
           p.minutes_played, p.matches_played, p.image_url, p.club_logo, p.contract,
           COALESCE(AVG(r.final_score), 0) as ortalama_puan,
           (SELECT team_style FROM reports WHERE player_id = p.id ORDER BY id DESC LIMIT 1) as oyun_tarzi,
           (SELECT decision FROM reports WHERE player_id = p.id ORDER BY id DESC LIMIT 1) as son_karar_metni
    FROM players p
    LEFT JOIN reports r ON p.id = r.player_id
    GROUP BY p.id ORDER BY ortalama_puan DESC
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df

def get_player_full_profile(pid):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM players WHERE id=?", (pid,))
    row = c.fetchone()
    p = dict(row) if row else None
    r_df = pd.read_sql(f"SELECT * FROM reports WHERE player_id={pid} ORDER BY created_at DESC", conn)
    conn.close()
    return p, r_df

def get_flattened_data():
    conn = get_connection()
    q = """SELECT p.id, p.name, p.team, p.league, p.position, p.age, p.image_url, p.minutes_played, r.match_data_json, r.final_score
           FROM reports r JOIN players p ON p.id = r.player_id GROUP BY p.id ORDER BY r.created_at DESC"""
    df = pd.read_sql(q, conn)
    conn.close()
    if df.empty: return pd.DataFrame()
    match_data_list = []
    for index, row in df.iterrows():
        try:
            m_data = json.loads(row['match_data_json'])
            m_data['player_id'] = row['id']
            match_data_list.append(m_data)
        except: pass
    if match_data_list:
        metrics_df = pd.DataFrame(match_data_list)
        df = pd.merge(df, metrics_df, left_on='id', right_on='player_id', how='left')
    return df

def get_team_stats():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM teams", conn)
    conn.close()
    return df

def auto_generate_fake_reports():
    """
    Gerçekçi Veri Üretimi (Normal Dağılım)
    """
    with db_lock:
        conn = get_connection()
        try:
            players = pd.read_sql("SELECT id, team, position, age, foot, nationality, height FROM players", conn).to_dict('records')
            if not players: return 0
            unique_teams = set([p['team'] for p in players])
            formations = ["4-2-3-1", "4-3-3", "3-5-2", "4-4-2", "3-4-3"]

            # Takımları Oluştur
            for t_name in unique_teams:
                curr = conn.execute("SELECT id FROM teams WHERE name=?", (t_name,)).fetchone()
                if not curr:
                    poss = round(random.gauss(50, 8), 1) # Ort 50, sapma 8
                    ppda = round(random.gauss(12, 3), 1)
                    xg = round(random.uniform(0.8, 2.5), 2)
                    xga = round(random.uniform(0.8, 2.0), 2)
                    style = logic.determine_team_style({'avg_possession': poss, 'ppda': ppda})
                    conn.execute("INSERT INTO teams (name, formation, game_style, avg_possession, ppda, xg_for, xg_against) VALUES (?,?,?,?,?,?,?)",
                                 (t_name, random.choice(formations), style, poss, ppda, xg, xga))

            cnt = 0
            for p in players:
                t_row = conn.execute("SELECT game_style FROM teams WHERE name=?", (p['team'],)).fetchone()
                t_style = t_row[0] if t_row else "Standart"

                # EKSİK VERİLERİ DOLDUR (Boy, Ayak, Uyruk)
                new_foot = p['foot']
                if not new_foot or new_foot == "-" or new_foot == "0":
                    new_foot = random.choices(["Sağ", "Sol", "Çift"], weights=[70, 20, 10])[0]

                new_nat = p['nationality']
                if not new_nat or new_nat == "-": new_nat = "Bilinmiyor"

                new_height = p['height']
                if not new_height or new_height == 0:
                    new_height = int(random.gauss(180, 6)) # Ort 180cm, sapma 6cm

                conn.execute("UPDATE players SET foot=?, nationality=?, height=? WHERE id=?", (new_foot, new_nat, new_height, p['id']))

                # MAÇ VERİLERİ
                matches = random.randint(5, 34)
                mins = matches * random.randint(60, 90)
                conn.execute("UPDATE players SET minutes_played=?, matches_played=? WHERE id=?", (mins, matches, p['id']))

                # GERÇEKÇİ SCOUT PUANI (ÇAN EĞRİSİ)
                # Ortalama oyuncu 5.5-6.0 puan alır (10 üzerinden)
                # 8+ alan çok nadir olsun
                try: attributes = logic.get_scout_attributes(p['position'])
                except: attributes = logic.SCOUT_GENERAL_ATTRS

                base_skill = random.gauss(5.8, 1.2) # Ort 5.8, Sapma 1.2
                # Genç yetenek bonusu (hafif)
                if p['age'] and p['age'] < 22: base_skill += random.uniform(0, 0.5)

                # Sınırla 1-10
                ratings = {att: max(2, min(9.5, base_skill + random.uniform(-1, 1))) for att in attributes}

                metrics_list = logic.get_match_metrics(p['position'])
                match_metrics = {m: round(random.uniform(0, 5) if "Goal" in m else random.uniform(10, 90), 1) for m in metrics_list}
                match_metrics['performance_index'] = base_skill * 10 + random.randint(-5, 5)

                role = random.choice(logic.get_available_roles(p['position']))
                score = logic.calculate_pro_score(ratings, match_metrics, p['position'], role, p['age'] or 25, "Orta")
                dec = logic.get_detailed_decision(score, p['age'] or 25)

                comments = {
                    "OyunTarzi": t_style,
                    "Fiziksel": "Lig seviyesine uygun fiziksel kapasitesi var.",
                    "Mental": "Disiplinli ancak liderlik özelliği geliştirmeli.",
                    "Taktiksel": f"{t_style} sisteminde sırıtmaz.",
                    "Teknik": "Temel teknik becerileri yerinde."
                }

                save_report(p['id'], "AI Bot", {'info': "Simulasyon", 'level': "Orta"}, ratings, match_metrics, score,
                            {'dec': dec, 'style': role, 'pros': "Potansiyel", 'cons': "-",
                             'on_ball': "Basit oynuyor.", 'off_ball': "Pozisyon sadakati var.",
                             'general': json.dumps(comments)}, conn=conn)
                cnt += 1

            conn.commit()
            return cnt
        except Exception as e:
            print(f"Bot Hata: {e}")
            import traceback
            traceback.print_exc()
            return 0
        finally:
            conn.close()

init_db()