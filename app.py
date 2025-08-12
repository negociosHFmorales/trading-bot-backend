# app.py
from flask import Flask, request, jsonify
import os, requests, math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import poisson

app = Flask(__name__)

# === CONFIG: ponerlo en variables de entorno en Render/GitHub secrets ===
ASSISTANT_API_KEY = os.getenv("ASSISTANT_API_KEY","changeme")   # seguridad simple para n8n -> flask
SPORTMONKS_TOKEN   = os.getenv("SPORTMONKS_TOKEN","")         # opcional, mejor datos
ODDS_API_KEY        = os.getenv("ODDS_API_KEY","")            # The Odds API (v4)
TZ_NAME            = os.getenv("TZ","America/Bogota")        # zona horaria
ZONE = ZoneInfo(TZ_NAME)

# ================= utilidades =================
def now_local():
    return datetime.now(ZONE)

def iso_ts(dt):
    # ISO para APIs (no milis)
    return dt.isoformat()

# Convierte odds decimal -> prob implícita
def implied_prob(dec_odds):
    try:
        return 1.0 / float(dec_odds) if dec_odds and dec_odds > 0 else 0.0
    except:
        return 0.0

def normalize_probs(ps):
    s = sum(ps)
    return [ (p/s) if s>0 else 0 for p in ps ]

# Poisson simple para obtener P(home/draw/away)
def poisson_result_probs(lambda_h, lambda_a, max_goals=7):
    # matriz de probabilidades goles i (home) y j (away)
    pm = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            pm[i,j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
    p_home = float(pm[np.triu_indices(max_goals+1, k=1)].sum())   # i>j
    p_draw = float(sum(pm[i,i] for i in range(max_goals+1)))
    p_away = float(pm[np.tril_indices(max_goals+1, k=-1)].sum())   # i<j
    # fallback (small rounding)
    s = p_home + p_draw + p_away
    if s==0:
        return 0.45, 0.10, 0.45
    return p_home/s, p_draw/s, p_away/s

# ================= fuentes de datos =================
def fetch_fixtures_sportmonks(start_dt, end_dt):
    """Si tienes SPORTMONKS_TOKEN usa esto (mejor datos)."""
    if not SPORTMONKS_TOKEN:
        return []
    start = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    end   = end_dt.strftime("%Y-%m-%dT%H:%M:%S")
    url = f"https://api.sportmonks.com/v3/football/fixtures?api_token={SPORTMONKS_TOKEN}&filter[starts_between]={start},{end}&include=localTeam,visitorTeam"
    r = requests.get(url, timeout=15)
    if r.status_code==200:
        return r.json().get('data', [])
    return []

def fetch_odds_theoddsapi(start_dt, end_dt):
    """Fallback: The Odds API - buscamos deportes 'soccer*' y filtramos por fecha."""
    if not ODDS_API_KEY:
        return []
    base = "https://api.the-odds-api.com/v4"
    # obtener deportes disponibles
    resp = requests.get(f"{base}/sports?apiKey={ODDS_API_KEY}", timeout=15)
    if resp.status_code!=200:
        return []
    sports = resp.json()
    events = []
    cutoff = end_dt
    for s in sports:
        key = s.get('key','')
        if 'soccer' not in key:   # por defecto solo soccer; puedes quitar esta línea para otros deportes
            continue
        try:
            odds_url = f"{base}/sports/{key}/odds?regions=eu&markets=h2h&oddsFormat=decimal&dateFormat=iso&apiKey={ODDS_API_KEY}"
            r = requests.get(odds_url, timeout=15)
            if r.status_code==200:
                data = r.json()
                for e in data:
                    # filtrar por commence_time
                    ct = e.get('commence_time')
                    if ct:
                        # convierte iso a dt naive, compara
                        dt = datetime.fromisoformat(ct.replace('Z','+00:00')).astimezone(ZONE)
                        if dt > start_dt and dt <= cutoff:
                            events.append(e)
        except Exception:
            continue
    return events

# ================= Lógica de análisis =================
def analyze_match_from_odds_item(item):
    # item de The Odds API v4: tiene 'home_team','away_team','commence_time','bookmakers'
    home = item.get('home_team') or item.get('teams',[None, None])[0]
    away = item.get('away_team') or (item.get('teams') or [None, None])[1]
    dt_raw = item.get('commence_time')
    dt = datetime.fromisoformat(dt_raw.replace('Z','+00:00')).astimezone(ZONE) if dt_raw else None

    # tomar la mejor casa (primer bookmaker) y las h2h odds si están
    odds_local = odds_draw = odds_visit = None
    bks = item.get('bookmakers',[])
    if bks:
        markets = bks[0].get('markets', [])
        for m in markets:
            if m.get('key') in ('h2h','head2head'):
                outcomes = m.get('outcomes',[])
                # outcomes suelen tener name & price (decimal)
                for o in outcomes:
                    name = o.get('name','').lower()
                    price = o.get('price')
                    if 'draw' in name or name in ('draw','empate'):
                        odds_draw = price
                    elif 'home' in name or name==home.lower():
                        odds_local = price
                    elif 'away' in name or name==away.lower():
                        odds_visit = price
    # si no hay empate (p.e. basketball) odds_draw = None

    # convertir a probabilidades (mercado)
    odds_list = [o for o in [odds_local, odds_draw, odds_visit] if o]
    impl_probs = [implied_prob(o) for o in odds_list]
    # si hay empate o no, normalizamos por mercado (quita el margin)
    fair_probs = normalize_probs(impl_probs)

    # Modelo Poisson simple: si no tenemos datos le damos lambdas promedio
    # (ideal: trae stats de SportMonks y calcula ataque/defensa)
    avg_home_goals = 1.45  # valor de referencia liga fútbol
    avg_away_goals = 1.10
    lambda_h = avg_home_goals * 1.05  # ventaja local ligera
    lambda_a = avg_away_goals * 0.95
    model_home, model_draw, model_away = poisson_result_probs(lambda_h, lambda_a)

    # map fair_probs -> align indices
    # vamos a construir lista [p_home,p_draw,p_away] coherente
    market_probs = [0,0,0]
    idx = 0
    for i,od in enumerate([odds_local, odds_draw, odds_visit]):
        if od:
            market_probs[i] = fair_probs[idx]; idx += 1

    # combine: peso 0.6 al modelo, 0.4 al mercado
    final_p_home = 0.6*model_home + 0.4*market_probs[0]
    final_p_draw = 0.6*model_draw + 0.4*market_probs[1]
    final_p_away = 0.6*model_away + 0.4*market_probs[2]
    # normalizar
    s = final_p_home + final_p_draw + final_p_away
    if s>0:
        final_p_home/=s; final_p_draw/=s; final_p_away/=s

    # calcular EV para cada opción que tenga odds
    results = []
    if odds_local:
        ev_local = final_p_home*odds_local - 1
        results.append({'side':'local','odds':odds_local,'prob':final_p_home,'ev':ev_local})
    if odds_draw:
        ev_draw = final_p_draw*odds_draw - 1
        results.append({'side':'empate','odds':odds_draw,'prob':final_p_draw,'ev':ev_draw})
    if odds_visit:
        ev_visit = final_p_away*odds_visit - 1
        results.append({'side':'visitante','odds':odds_visit,'prob':final_p_away,'ev':ev_visit})

    # elegir mejor EV
    if not results:
        return None
    best = max(results, key=lambda x: x['ev'])

    # confianza sencilla: dependiente de diferencia entre modelo y mercado y existencia de datos
    # (más datos -> mayor confianza). Aquí hacemos una heurística.
    diff = abs((best['prob']) - ( (1/best['odds']) if best['odds'] else 0 ))
    confianza = max(30, min(90, int(70 - diff*100)))  # 30%-90% rango heurístico

    return {
        'partido': f"{home} vs {away}",
        'liga': item.get('sport_key','soccer'),
        'fecha_hora': dt.isoformat() if dt else None,
        'recomendacion': {
            'opcion': best['side'],
            'odds': best['odds'],
            'probabilidad': round(best['prob'], 4),
            'valor_esperado': round(best['ev'],4),
            'confianza': confianza
        },
        'raw': {
            'market_odds': {'home':odds_local,'draw':odds_draw,'away':odds_visit},
            'model_pd': {'home':model_home,'draw':model_draw,'away':model_away}
        }
    }

# ================= Endpoint principal =================
@app.route("/run-analysis", methods=["POST"])
def run_analysis():
    # seguridad simple
    key = request.headers.get("X-API-KEY","")
    if ASSISTANT_API_KEY and key != ASSISTANT_API_KEY:
        return jsonify({'status':'error','message':'unauthorized'}), 401

    # parámetro opcional: hours (default 12)
    body = request.get_json(silent=True) or {}
    hours = int(body.get('hours', 12))
    start_dt = now_local()
    end_dt = start_dt + timedelta(hours=hours)

    # 1) Obtener partidos preferiblemente con SportMonks
    fixtures = fetch_fixtures_sportmonks(start_dt, end_dt)
    recommendations = []

    if fixtures:
        # transformar fixtures SportMonks a formato simple y analizarlos
        for f in fixtures:
            # SportMonks data structure varía; tratamos de convertir a algo manejable
            try:
                teams = f.get('participants') or []
                # fallback keys
                local = f.get('localTeam',{}).get('data',{}).get('name') if f.get('localTeam') else f.get('home_team')
                visit = f.get('visitorTeam',{}).get('data',{}).get('name') if f.get('visitorTeam') else f.get('away_team')
                inicio = f.get('starting_at') or f.get('time')
                # construir item compatible con analyze_match_from_odds_item (puede no tener odds)
                item = {
                    'home_team': local,
                    'away_team': visit,
                    'commence_time': inicio,
                    'sport_key': 'soccer'
                }
                # TODO: podrías pedir odds por fixture si SportMonks incluye mercado
                rec = analyze_match_from_odds_item(item)
                if rec:
                    recommendations.append(rec)
            except Exception:
                continue

    # 2) fallback: The Odds API
    if not recommendations:
        odds_events = fetch_odds_theoddsapi(start_dt, end_dt)
        for it in odds_events:
            rec = analyze_match_from_odds_item(it)
            if rec:
                recommendations.append(rec)

    # 3) filtrar y ordenar por EV (solo EV>0 o top N)
    mejores = [r for r in recommendations if r['recomendacion']['valor_esperado'] > 0.03]  # >3% EV por defecto
    mejores = sorted(mejores, key=lambda x: x['recomendacion']['valor_esperado'], reverse=True)[:8]

    return jsonify({
        'status':'success',
        'timestamp': now_local().isoformat(),
        'request_hours': hours,
        'total_found': len(recommendations),
        'recomendaciones': mejores
    })

@app.route("/health")
def health():
    return jsonify({'status':'ok','tz':TZ_NAME})

if __name__ == "__main__":
    port = int(os.getenv("PORT","5000"))
    app.run(host="0.0.0.0", port=port)
