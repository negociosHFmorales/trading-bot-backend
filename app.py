from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time, timedelta
import pytz
import logging
from functools import wraps
import time as time_module

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de acciones (20 acciones diversificadas)
SYMBOLS = [
    # Tecnologia
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'ADBE',
    # Finanzas
    'JPM', 'V', 'MA', 'PYPL',
    # Salud
    'JNJ', 'PFE', 'UNH', 'MRNA',
    # Otros sectores
    'TSLA', 'DIS', 'KO', 'WMT'
]

# Cache simple para evitar rate limiting
CACHE = {}
CACHE_DURATION = 300  # 5 minutos

def cache_result(duration=300):
    """Decorator para cachear resultados"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            now = time_module.time()
            
            if cache_key in CACHE:
                cached_time, cached_result = CACHE[cache_key]
                if now - cached_time < duration:
                    return cached_result
            
            result = func(*args, **kwargs)
            CACHE[cache_key] = (now, result)
            return result
        return wrapper
    return decorator

def rate_limit():
    """Simple rate limiting"""
    time_module.sleep(0.1)  # 100ms delay between requests

def es_horario_mercado():
    """Verifica si el mercado esta abierto"""
    try:
        ahora = datetime.now(pytz.timezone('US/Eastern'))
        hora_actual = ahora.time()
        dia_semana = ahora.weekday()
        
        # Lunes=0, Viernes=4
        if dia_semana > 4:  # Fin de semana
            return False
            
        # Horario: 9:30 AM - 4:00 PM ET
        apertura = time(9, 30)
        cierre = time(16, 0)
        
        return apertura <= hora_actual <= cierre
    except Exception:
        return True  # Default to open if timezone fails

def calcular_indicadores_avanzados(data):
    """Calcula multiples indicadores tecnicos con manejo de errores mejorado"""
    try:
        if len(data) < 50:
            return None
        
        # Ensure we have clean data
        data = data.dropna()
        if len(data) < 50:
            return None
            
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9).mean()
        
        # SMA (Simple Moving Averages)
        sma_20 = data['Close'].rolling(window=20).mean()
        sma_50 = data['Close'].rolling(window=50).mean()
        
        # EMA (Exponential Moving Averages)
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        bb_middle = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # Volumen promedio
        volume_avg = data['Volume'].rolling(window=20).mean()
        
        # Volatilidad
        volatility = data['Close'].rolling(window=20).std() / data['Close'].rolling(window=20).mean()
        
        # Momentum
        momentum = data['Close'] / data['Close'].shift(10) - 1
        
        return {
            'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
            'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0,
            'signal_line': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0,
            'sma_20': float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else float(data['Close'].iloc[-1]),
            'sma_50': float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else float(data['Close'].iloc[-1]),
            'ema_12': float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else float(data['Close'].iloc[-1]),
            'ema_26': float(ema_26.iloc[-1]) if not pd.isna(ema_26.iloc[-1]) else float(data['Close'].iloc[-1]),
            'bb_upper': float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else float(data['Close'].iloc[-1]) * 1.02,
            'bb_lower': float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else float(data['Close'].iloc[-1]) * 0.98,
            'bb_middle': float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else float(data['Close'].iloc[-1]),
            'volume_ratio': float(data['Volume'].iloc[-1] / volume_avg.iloc[-1]) if volume_avg.iloc[-1] > 0 and not pd.isna(volume_avg.iloc[-1]) else 1.0,
            'volatility': float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 0.02,
            'momentum': float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else 0,
            'price': float(data['Close'].iloc[-1]),
            'volume': int(data['Volume'].iloc[-1])
        }
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def evaluar_senal_avanzada(indicators):
    """Sistema de puntuacion avanzado para senales con mejor scoring"""
    puntos_compra = 0
    puntos_venta = 0
    razones = []
    
    try:
        # RSI Analysis (peso: 2 puntos max)
        rsi = indicators['rsi']
        if rsi < 25:
            puntos_compra += 2
            razones.append(f"RSI extremadamente sobreventa ({rsi:.1f})")
        elif rsi < 35:
            puntos_compra += 1.5
            razones.append(f"RSI sobreventa ({rsi:.1f})")
        elif rsi > 75:
            puntos_venta += 2
            razones.append(f"RSI extremadamente sobrecompra ({rsi:.1f})")
        elif rsi > 65:
            puntos_venta += 1.5
            razones.append(f"RSI sobrecompra ({rsi:.1f})")
        
        # MACD Analysis (peso: 2 puntos max)
        macd = indicators['macd']
        signal_line = indicators['signal_line']
        macd_diff = macd - signal_line
        
        if macd > signal_line:
            if macd > 0 and macd_diff > 0.01:
                puntos_compra += 2
                razones.append("MACD fuertemente alcista")
            elif macd_diff > 0:
                puntos_compra += 1
                razones.append("MACD cruzando alcista")
        else:
            if macd < 0 and macd_diff < -0.01:
                puntos_venta += 2
                razones.append("MACD fuertemente bajista")
            elif macd_diff < 0:
                puntos_venta += 1
                razones.append("MACD cruzando bajista")
        
        # Trend Analysis (peso: 2 puntos max)
        precio = indicators['price']
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        ema_12 = indicators['ema_12']
        
        if precio > ema_12 > sma_20 > sma_50:
            puntos_compra += 2
            razones.append("Tendencia alcista muy fuerte")
        elif precio > sma_20 > sma_50:
            puntos_compra += 1.5
            razones.append("Tendencia alcista fuerte")
        elif precio > sma_20:
            puntos_compra += 1
            razones.append("Tendencia alcista corto plazo")
        elif precio < ema_12 < sma_20 < sma_50:
            puntos_venta += 2
            razones.append("Tendencia bajista muy fuerte")
        elif precio < sma_20 < sma_50:
            puntos_venta += 1.5
            razones.append("Tendencia bajista fuerte")
        elif precio < sma_20:
            puntos_venta += 1
            razones.append("Tendencia bajista corto plazo")
        
        # Bollinger Bands (peso: 1 punto max)
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        bb_middle = indicators['bb_middle']
        
        if precio <= bb_lower:
            puntos_compra += 1
            razones.append("Precio en banda inferior Bollinger")
        elif precio >= bb_upper:
            puntos_venta += 1
            razones.append("Precio en banda superior Bollinger")
        
        # Volume Analysis (peso: 1 punto max)
        volume_ratio = indicators['volume_ratio']
        if volume_ratio > 2:  # Volumen muy alto
            if puntos_compra > puntos_venta:
                puntos_compra += 1
                razones.append(f"Volumen excepcional confirma compra ({volume_ratio:.1f}x)")
            else:
                puntos_venta += 1
                razones.append(f"Volumen excepcional confirma venta ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.5:
            if puntos_compra > puntos_venta:
                puntos_compra += 0.5
                razones.append(f"Alto volumen apoya señal ({volume_ratio:.1f}x)")
            else:
                puntos_venta += 0.5
                razones.append(f"Alto volumen apoya señal ({volume_ratio:.1f}x)")
        
        # Momentum Analysis (peso: 1 punto max)
        momentum = indicators['momentum']
        if momentum > 0.05:  # 5% momentum positivo
            puntos_compra += 1
            razones.append(f"Momentum positivo fuerte ({momentum:.1%})")
        elif momentum < -0.05:
            puntos_venta += 1
            razones.append(f"Momentum negativo fuerte ({momentum:.1%})")
        
        # Volatility Check (filtro de riesgo)
        volatility = indicators['volatility']
        if volatility > 0.05:  # 5% volatilidad diaria
            razones.append(f"⚠️ Alta volatilidad ({volatility:.1%})")
            # Reducir confianza en mercados muy volátiles
            puntos_compra *= 0.8
            puntos_venta *= 0.8
        
        # Decision Final (umbral más inteligente)
        total_puntos = max(puntos_compra, puntos_venta)
        
        if puntos_compra >= 3.5 and puntos_compra > puntos_venta + 0.5:
            confianza = min(0.95, (puntos_compra / 8) + 0.3)  # Normalizado entre 0.3-0.95
            return "BUY", confianza, razones
        elif puntos_venta >= 3.5 and puntos_venta > puntos_compra + 0.5:
            confianza = min(0.95, (puntos_venta / 8) + 0.3)
            return "SELL", confianza, razones
        
        return None, 0, razones
        
    except Exception as e:
        logger.error(f"Error evaluating signal: {e}")
        return None, 0, ["Error en análisis"]

def analizar_sentimiento_noticias(symbol):
    """Análisis de sentimiento basado en noticias"""
    api_key = "TU_NEWSAPI_KEY_AQUI"  # Reemplaza con tu key
    try:
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': f'{symbol} stock',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10,
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        noticias = response.json().get('articles', [])
        
        # Palabras clave para análisis
        palabras_positivas = [
            'surge', 'soars', 'gains', 'profit', 'growth', 'bullish', 
            'upgrade', 'beat', 'strong', 'record', 'positive', 'rise',
            'outperform', 'breakthrough', 'expansion', 'innovation'
        ]
        
        palabras_negativas = [
            'falls', 'drops', 'decline', 'loss', 'bearish', 'downgrade', 
            'miss', 'weak', 'concern', 'negative', 'crash', 'plunge',
            'underperform', 'recession', 'layoffs', 'scandal'
        ]
        
        sentiment_score = 0
        noticias_analizadas = 0
        noticias_procesadas = []
        
        for noticia in noticias[:5]:  # Solo primeras 5
            titulo = noticia.get('title', '').lower()
            descripcion = noticia.get('description', '').lower()
            texto_completo = f"{titulo} {descripcion}"
            
            puntos_positivos = sum(1 for palabra in palabras_positivas if palabra in texto_completo)
            puntos_negativos = sum(1 for palabra in palabras_negativas if palabra in texto_completo)
            
            noticia_sentiment = puntos_positivos - puntos_negativos
            sentiment_score += noticia_sentiment
            noticias_analizadas += 1
            
            noticias_procesadas.append({
                'title': noticia.get('title', ''),
                'url': noticia.get('url', ''),
                'published': noticia.get('publishedAt', ''),
                'sentiment': 'POSITIVO' if noticia_sentiment > 0 else 'NEGATIVO' if noticia_sentiment < 0 else 'NEUTRAL',
                'score': noticia_sentiment
            })
        
        if noticias_analizadas > 0:
            sentiment_score = sentiment_score / noticias_analizadas
            
        return {
            'sentiment_score': round(sentiment_score, 2),
            'news_count': noticias_analizadas,
            'sentiment_label': 'POSITIVO' if sentiment_score > 0.5 else 'NEGATIVO' if sentiment_score < -0.5 else 'NEUTRAL',
            'news_articles': noticias_procesadas
        }
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment for {symbol}: {e}")
        return {
            'sentiment_score': 0,
            'news_count': 0,
            'sentiment_label': 'NO_DATA',
            'error': str(e),
            'news_articles': []
        }

def calcular_gestion_riesgo(precio, accion, confianza, volatility=0.02):
    """Gestion avanzada de riesgo con volatilidad"""
    try:
        balance_simulado = 10000  # $10,000 para paper trading
        riesgo_base = 0.02  # 2% base
        
        # Ajustar riesgo por volatilidad
        riesgo_ajustado_vol = riesgo_base * (1 + volatility * 2)  # Más riesgo = menos posición
        riesgo_ajustado_vol = min(riesgo_ajustado_vol, 0.05)  # Máximo 5%
        
        # Factor de confianza
        factor_confianza = min(confianza * 1.2, 1.0)
        riesgo_final = riesgo_ajustado_vol * factor_confianza
        
        # Stop loss y take profit dinámicos basados en volatilidad
        vol_multiplier = max(1, volatility * 20)  # Volatilidad como multiplicador
        
        if accion == "BUY":
            stop_loss = precio * (1 - 0.03 * vol_multiplier)  # 3% base * volatilidad
            take_profit = precio * (1 + 0.06 * vol_multiplier)  # 6% base * volatilidad
        else:
            stop_loss = precio * (1 + 0.03 * vol_multiplier)
            take_profit = precio * (1 - 0.06 * vol_multiplier)
        
        # Calcular cantidad de acciones
        riesgo_dolares = balance_simulado * riesgo_final
        riesgo_por_accion = abs(precio - (stop_loss if accion == "BUY" else stop_loss))
        cantidad = int(riesgo_dolares / riesgo_por_accion) if riesgo_por_accion > 0 else 1
        cantidad = max(1, min(cantidad, 100))  # Entre 1 y 100 acciones
        
        return {
            'position_size': cantidad,
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_amount': round(riesgo_dolares, 2),
            'risk_percent': round(riesgo_final * 100, 1),
            'volatility_factor': round(vol_multiplier, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating risk management: {e}")
        return {
            'position_size': 1,
            'stop_loss': round(precio * 0.95, 2),
            'take_profit': round(precio * 1.05, 2),
            'risk_amount': 200,
            'risk_percent': 2.0,
            'volatility_factor': 1.0
        }

@app.route('/analyze')
@cache_result(300)  # Cache por 5 minutos
def analyze_market():
    """ENDPOINT PRINCIPAL - Analisis completo con mejoras"""
    
    # Verificar horario
    market_open = es_horario_mercado()
    
    # Para testing, permitir override
    force_analysis = request.args.get('force', 'false').lower() == 'true'
    
    if not market_open and not force_analysis:
        return jsonify({
            "signals": [],
            "actionable_signals": 0,
            "total_analyzed": 0,
            "message": "Mercado cerrado - análisis pausado",
            "market_status": "CLOSED",
            "next_open": "9:30 AM ET"
        })
    
    signals = []
    errors = []
    total_analyzed = 0
    
    # Permitir filtrar por símbolo específico
    symbols_to_analyze = request.args.get('symbols', '').split(',') if request.args.get('symbols') else SYMBOLS
    symbols_to_analyze = [s.strip().upper() for s in symbols_to_analyze if s.strip()]
    
    for symbol in symbols_to_analyze:
        try:
            rate_limit()  # Rate limiting simple
            
            # Obtener datos históricos
            stock = yf.Ticker(symbol)
            data = stock.history(period='60d', interval='1d')
            
            if len(data) < 50:
                errors.append(f"{symbol}: Datos insuficientes")
                continue
                
            total_analyzed += 1
            
            # Calcular indicadores
            indicators = calcular_indicadores_avanzados(data)
            if not indicators:
                errors.append(f"{symbol}: Error en indicadores")
                continue
            
            # Evaluar señal
            action, confidence, reasons = evaluar_senal_avanzada(indicators)
            
            # Umbral de confianza configurable
            min_confidence = float(request.args.get('min_confidence', 0.4))
            
            if action and confidence >= min_confidence:
                # Gestión de riesgo
                risk_mgmt = calcular_gestion_riesgo(
                    indicators['price'], 
                    action, 
                    confidence,
                    indicators['volatility']
                )
                
                # Obtener información adicional
                info = stock.info
                sector = info.get('sector', 'Unknown')
                market_cap = info.get('marketCap', 0)
                
                signals.append({
                    "symbol": symbol,
                    "action": action,
                    "current_price": round(indicators['price'], 2),
                    "confidence": round(confidence, 3),
                    "sector": sector,
                    "market_cap": market_cap,
                    "indicators": {
                        "rsi": round(indicators['rsi'], 2),
                        "macd": round(indicators['macd'], 4),
                        "signal_line": round(indicators['signal_line'], 4),
                        "sma_20": round(indicators['sma_20'], 2),
                        "sma_50": round(indicators['sma_50'], 2),
                        "ema_12": round(indicators['ema_12'], 2),
                        "volume_ratio": round(indicators['volume_ratio'], 2),
                        "volatility": round(indicators['volatility'], 3),
                        "momentum": round(indicators['momentum'], 3)
                    },
                    "reasons": reasons,
                    "risk_management": risk_mgmt,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            error_msg = f"Error analizando {symbol}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue
    
    # Ordenar por confianza (mejores primero)
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    return jsonify({
        "signals": signals,
        "actionable_signals": len(signals),
        "total_analyzed": total_analyzed,
        "market_status": "OPEN" if market_open else "CLOSED",
        "analysis_time": datetime.now().isoformat(),
        "errors": errors[:5],  # Solo primeros 5 errores
        "cache_status": "cached" if f"analyze_market_{str(())}_{str({})}" in CACHE else "fresh"
    })

@app.route('/analyze/<symbol>')
def analyze_single_stock(symbol):
    """Analizar una acción específica en detalle"""
    try:
        symbol = symbol.upper()
        stock = yf.Ticker(symbol)
        
        # Obtener más datos para análisis detallado
        data = stock.history(period='3mo', interval='1d')
        info = stock.info
        
        if len(data) < 50:
            return jsonify({"error": "Datos insuficientes"}), 400
        
        # Indicadores técnicos
        indicators = calcular_indicadores_avanzados(data)
        if not indicators:
            return jsonify({"error": "Error calculando indicadores"}), 500
        
        # Análisis de sentimiento (opcional)
        sentiment_data = {}
        include_sentiment = request.args.get('sentiment', 'false').lower() == 'true'
        if include_sentiment:
            sentiment_data = analizar_sentimiento_noticias(symbol)
        
        # Señal
        action, confidence, reasons = evaluar_senal_avanzada(indicators)
        
        # Gestión de riesgo
        risk_mgmt = calcular_gestion_riesgo(
            indicators['price'], 
            action or "HOLD", 
            confidence,
            indicators['volatility']
        )
        
        # Datos históricos de precio (últimos 30 días)
        price_history = []
        recent_data = data.tail(30)
        for date, row in recent_data.iterrows():
            price_history.append({
                "date": date.strftime('%Y-%m-%d'),
                "price": round(row['Close'], 2),
                "volume": int(row['Volume'])
            })
        
        response_data = {
            "symbol": symbol,
            "company_name": info.get('longName', symbol),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "market_cap": info.get('marketCap', 0),
            "current_price": round(indicators['price'], 2),
            "action": action or "HOLD",
            "confidence": round(confidence, 3) if confidence else 0,
            "indicators": {
                "rsi": round(indicators['rsi'], 2),
                "macd": round(indicators['macd'], 4),
                "signal_line": round(indicators['signal_line'], 4),
                "sma_20": round(indicators['sma_20'], 2),
                "sma_50": round(indicators['sma_50'], 2),
                "ema_12": round(indicators['ema_12'], 2),
                "bb_upper": round(indicators['bb_upper'], 2),
                "bb_lower": round(indicators['bb_lower'], 2),
                "volume_ratio": round(indicators['volume_ratio'], 2),
                "volatility": round(indicators['volatility'], 3),
                "momentum": round(indicators['momentum'], 3)
            },
            "reasons": reasons,
            "risk_management": risk_mgmt,
            "price_history": price_history,
            "analysis_time": datetime.now().isoformat()
        }
        
        # Agregar datos de sentimiento si se solicitaron
        if include_sentiment and sentiment_data:
            response_data["sentiment"] = sentiment_data
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sentiment/<symbol>')
def analyze_sentiment(symbol):
    """Endpoint específico para análisis de sentimiento"""
    try:
        symbol = symbol.upper()
        sentiment_data = analizar_sentimiento_noticias(symbol)
        
        return jsonify({
            "symbol": symbol,
            "sentiment": sentiment_data,
            "analysis_time": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/portfolio/correlation')
@cache_result(600)  # Cache por 10 minutos
def portfolio_correlation():
    """Analisis de correlaciones del portafolio mejorado"""
    try:
        # Obtener datos de todas las acciones
        data = {}
        period = request.args.get('period', '30d')  # Configurable
        
        for symbol in SYMBOLS:
            try:
                rate_limit()
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period)
                if len(hist) > 20:
                    data[symbol] = hist['Close'].pct_change().dropna()
            except Exception as e:
                logger.error(f"Error getting data for {symbol}: {e}")
                continue
        
        # Calcular matriz de correlaciones
        correlations = []
        symbols_list = list(data.keys())
        correlation_matrix = {}
        
        for i in range(len(symbols_list)):
            symbol1 = symbols_list[i]
            correlation_matrix[symbol1] = {}
            
            for j in range(len(symbols_list)):
                symbol2 = symbols_list[j]
                
                if i == j:
                    correlation_matrix[symbol1][symbol2] = 1.0
                    continue
                
                if len(data[symbol1]) > 10 and len(data[symbol2]) > 10:
                    # Alinear fechas
                    common_dates = data[symbol1].index.intersection(data[symbol2].index)
                    if len(common_dates) > 10:
                        corr = data[symbol1].loc[common_dates].corr(data[symbol2].loc[common_dates])
                        correlation_matrix[symbol1][symbol2] = round(corr, 3)
                        
                        # Solo guardar correlaciones significativas para el resumen
                        if i < j and abs(corr) > 0.6:  # Umbral configurable
                            correlations.append({
                                "pair": [symbol1, symbol2],
                                "correlation": round(corr, 3),
                                "strength": "Fuerte" if abs(corr) > 0.8 else "Moderada"
                            })
        
        # Análisis de diversificación
        high_correlations = [c for c in correlations if abs(c["correlation"]) > 0.7]
        avg_correlation = np.mean([abs(c["correlation"]) for c in correlations]) if correlations else 0
        
        # Recomendaciones
        if avg_correlation > 0.7:
            recommendation = "⚠️ Portafolio con alta correlación - Considerar mayor diversificación"
            risk_level = "Alto"
        elif avg_correlation > 0.5:
            recommendation = "⚡ Diversificación moderada - Monitorear correlaciones"
            risk_level = "Medio"
        else:
            recommendation = "✅ Portafolio bien diversificado"
            risk_level = "Bajo"
        
        return jsonify({
            "correlation_matrix": correlation_matrix,
            "high_correlations": high_correlations,
            "average_correlation": round(avg_correlation, 3),
            "total_pairs_analyzed": len(correlations),
            "recommendation": recommendation,
            "risk_level": risk_level,
            "analysis_period": period,
            "analysis_date": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        return jsonify({
            "high_correlations": [],
            "recommendation": "Error en análisis de correlación",
            "error": str(e)
        }), 500

@app.route('/market/overview')
def market_overview():
    """Vista general del mercado"""
    try:
        # Índices principales
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX (Volatilidad)'
        }
        
        market_data = {}
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    
                    market_data[symbol] = {
                        'name': name,
                        'current': round(current, 2),
                        'change_percent': round(change, 2),
                        'trend': 'up' if change > 0 else 'down'
                    }
            except Exception as e:
                logger.error(f"Error getting {symbol}: {e}")
                continue
        
        # Análisis de sentimiento general
        positive_changes = sum(1 for data in market_data.values() if data['change_percent'] > 0)
        total_indices = len(market_data)
        
        if positive_changes / total_indices > 0.6:
            sentiment = "Optimista"
            sentiment_emoji = "📈"
        elif positive_changes / total_indices < 0.4:
            sentiment = "Pesimista"  
            sentiment_emoji = "📉"
        else:
            sentiment = "Neutral"
            sentiment_emoji = "➡️"
        
        return jsonify({
            "indices": market_data,
            "market_sentiment": sentiment,
            "sentiment_emoji": sentiment_emoji,
            "market_status": "OPEN" if es_horario_mercado() else "CLOSED",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in market overview: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """Health check mejorado"""
    try:
        # Test básico de conectividad con yfinance
        test_stock = yf.Ticker('AAPL')
        test_data = test_stock.history(period='1d')
        
        api_status = "OK" if len(test_data) > 0 else "WARNING"
        
        return jsonify({
            "status": "OK",
            "timestamp": datetime.now().isoformat(),
            "market_open": es_horario_mercado(),
            "symbols_count": len(SYMBOLS),
            "version": "2.1",
            "yfinance_api": api_status,
            "cache_size": len(CACHE),
            "features": {
                "technical_analysis": True,
                "risk_management": True,
                "correlation_analysis": True,
                "sentiment_analysis": True,
                "market_overview": True
            }
        })
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "ERROR",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

@app.route('/status')
def ping_endpoint():
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "market_open": es_horario_mercado(),
        "symbols_count": len(SYMBOLS),
        "version": "2.1"
    }

@app.route('/ping')
def ping():
    return {"message": "pong", "timestamp": datetime.now().isoformat()}

# Endpoint para limpiar cache manualmente
@app.route('/admin/clear-cache', methods=['POST'])
def clear_cache():
    """Limpiar cache manualmente"""
    global CACHE
    cache_size = len(CACHE)
    CACHE.clear()
    
    return jsonify({
        "message": f"Cache cleared - {cache_size} items removed",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
