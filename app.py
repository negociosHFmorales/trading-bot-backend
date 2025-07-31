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

# Símbolos prioritarios para análisis de sentimiento (para conservar API calls)
SENTIMENT_PRIORITY_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']

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

@cache_result(600)  # Cache por 10 minutos para sentimiento
def analizar_sentimiento_noticias(symbol, api_key="TU_NEWSAPI_KEY_AQUI"):
    """Análisis de sentimiento basado en noticias mejorado e integrado"""
    try:
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': f'{symbol} stock OR {symbol} earnings OR {symbol} revenue OR {symbol} quarter',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 12,
            'apiKey': api_key,
            'from': (datetime.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            logger.warning(f"NewsAPI returned status {response.status_code} for {symbol}")
            return {
                'sentiment_score': 0,
                'news_count': 0,
                'sentiment_label': 'NO_DATA',
                'confidence': 0,
                'news_articles': []
            }
        
        noticias = response.json().get('articles', [])
        
        # Palabras clave mejoradas y categorizadas
        palabras_muy_positivas = [
            'surge', 'soars', 'rally', 'breakthrough', 'earnings beat', 'revenue growth',
            'outperform', 'bullish', 'record high', 'strong momentum', 'upgrade'
        ]
        
        palabras_positivas = [
            'gains', 'profit', 'growth', 'beat', 'strong', 'positive', 'rise',
            'expansion', 'innovation', 'partnership', 'acquisition', 'dividend'
        ]
        
        palabras_muy_negativas = [
            'crash', 'plunge', 'disappointing', 'earnings miss', 'revenue decline',
            'underperform', 'bearish', 'selloff', 'recession', 'bankruptcy'
        ]
        
        palabras_negativas = [
            'falls', 'drops', 'decline', 'loss', 'downgrade', 'miss', 'weak',
            'concern', 'negative', 'volatility', 'uncertainty', 'lawsuit'
        ]
        
        sentiment_score = 0
        noticias_analizadas = 0
        noticias_procesadas = []
        
        for noticia in noticias[:10]:  # Primeras 10 noticias
            titulo = noticia.get('title', '').lower()
            descripcion = noticia.get('description', '').lower() if noticia.get('description') else ''
            
            # Peso mayor para el título (3x), menor para descripción (1x)
            puntos_muy_positivos_titulo = sum(3 for palabra in palabras_muy_positivas if palabra in titulo)
            puntos_positivos_titulo = sum(2 for palabra in palabras_positivas if palabra in titulo)
            puntos_muy_negativos_titulo = sum(3 for palabra in palabras_muy_negativas if palabra in titulo)
            puntos_negativos_titulo = sum(2 for palabra in palabras_negativas if palabra in titulo)
            
            puntos_muy_positivos_desc = sum(1 for palabra in palabras_muy_positivas if palabra in descripcion)
            puntos_positivos_desc = sum(0.5 for palabra in palabras_positivas if palabra in descripcion)
            puntos_muy_negativos_desc = sum(1 for palabra in palabras_muy_negativas if palabra in descripcion)
            puntos_negativos_desc = sum(0.5 for palabra in palabras_negativas if palabra in descripcion)
            
            puntos_totales = (
                puntos_muy_positivos_titulo + puntos_positivos_titulo + 
                puntos_muy_positivos_desc + puntos_positivos_desc
            ) - (
                puntos_muy_negativos_titulo + puntos_negativos_titulo + 
                puntos_muy_negativos_desc + puntos_negativos_desc
            )
            
            sentiment_score += puntos_totales
            noticias_analizadas += 1
            
            # Determinar sentimiento individual de la noticia
            if puntos_totales > 2:
                noticia_sentiment = 'MUY_POSITIVO'
            elif puntos_totales > 0:
                noticia_sentiment = 'POSITIVO'
            elif puntos_totales < -2:
                noticia_sentiment = 'MUY_NEGATIVO'
            elif puntos_totales < 0:
                noticia_sentiment = 'NEGATIVO'
            else:
                noticia_sentiment = 'NEUTRAL'
            
            noticias_procesadas.append({
                'title': noticia.get('title', ''),
                'url': noticia.get('url', ''),
                'published': noticia.get('publishedAt', ''),
                'sentiment': noticia_sentiment,
                'score': round(puntos_totales, 1),
                'source': noticia.get('source', {}).get('name', 'Unknown')
            })
        
        if noticias_analizadas > 0:
            sentiment_score = sentiment_score / noticias_analizadas
        
        # Determinar etiqueta y confianza
        if sentiment_score > 1.5:
            sentiment_label = 'MUY_POSITIVO'
            confidence = min(0.9, abs(sentiment_score) * 0.2)
        elif sentiment_score > 0.5:
            sentiment_label = 'POSITIVO'
            confidence = min(0.8, abs(sentiment_score) * 0.15)
        elif sentiment_score < -1.5:
            sentiment_label = 'MUY_NEGATIVO'
            confidence = min(0.9, abs(sentiment_score) * 0.2)
        elif sentiment_score < -0.5:
            sentiment_label = 'NEGATIVO'
            confidence = min(0.8, abs(sentiment_score) * 0.15)
        else:
            sentiment_label = 'NEUTRAL'
            confidence = 0.5
        
        return {
            'sentiment_score': round(sentiment_score, 3),
            'news_count': noticias_analizadas,
            'sentiment_label': sentiment_label,
            'confidence': round(confidence, 3),
            'news_articles': noticias_procesadas[:5],  # Solo las primeras 5 para respuesta
            'analysis_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment for {symbol}: {e}")
        return {
            'sentiment_score': 0,
            'news_count': 0,
            'sentiment_label': 'ERROR',
            'confidence': 0,
            'error': str(e),
            'news_articles': []
        }

def evaluar_senal_avanzada(indicators, sentiment_data=None):
    """Sistema de puntuacion avanzado con análisis de sentimiento integrado"""
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
        elif rsi < 40:
            puntos_compra += 1
            razones.append(f"RSI favorable compra ({rsi:.1f})")
        elif rsi > 60:
            puntos_venta += 1
            razones.append(f"RSI favorable venta ({rsi:.1f})")
        
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
            razones.append("Tendencia alcista fuerte (SMA)")
        elif precio > sma_20:
            puntos_compra += 1
            razones.append("Tendencia alcista corto plazo")
        elif precio < ema_12 < sma_20 < sma_50:
            puntos_venta += 2
            razones.append("Tendencia bajista muy fuerte")
        elif precio < sma_20 < sma_50:
            puntos_venta += 1.5
            razones.append("Tendencia bajista fuerte (SMA)")
        elif precio < sma_20:
            puntos_venta += 1
            razones.append("Tendencia bajista corto plazo")
        
        # Bollinger Bands (peso: 1 punto max)
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        
        if precio <= bb_lower:
            puntos_compra += 1
            razones.append("Precio en banda inferior (oportunidad)")
        elif precio >= bb_upper:
            puntos_venta += 1
            razones.append("Precio en banda superior (cuidado)")
        
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
                razones.append(f"Alto volumen confirma venta")
        
        # ANÁLISIS DE SENTIMIENTO INTEGRADO (peso: 2.5 puntos max)
        if sentiment_data and sentiment_data['news_count'] > 0:
            sentiment_score = sentiment_data['sentiment_score']
            sentiment_label = sentiment_data['sentiment_label']
            confidence = sentiment_data['confidence']
            news_count = sentiment_data['news_count']
            
            # Aplicar peso basado en confianza del sentimiento
            peso_sentimiento = confidence * 2.5
            
            if sentiment_label == 'MUY_POSITIVO':
                puntos_compra += peso_sentimiento
                razones.append(f"📈 Sentimiento muy positivo ({news_count} noticias, score: {sentiment_score:.1f})")
            elif sentiment_label == 'POSITIVO':
                puntos_compra += peso_sentimiento * 0.6
                razones.append(f"📊 Sentimiento positivo ({news_count} noticias)")
            elif sentiment_label == 'MUY_NEGATIVO':
                puntos_venta += peso_sentimiento
                razones.append(f"📉 Sentimiento muy negativo ({news_count} noticias, score: {sentiment_score:.1f})")
            elif sentiment_label == 'NEGATIVO':
                puntos_venta += peso_sentimiento * 0.6
                razones.append(f"📊 Sentimiento negativo ({news_count} noticias)")
            else:
                razones.append(f"📰 Sentimiento neutral ({news_count} noticias)")
        
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
        
        # Decision Final con umbral ajustado por sentimiento
        umbral_base = 3.0
        # Si tenemos datos de sentimiento muy confiables, bajamos el umbral
        if sentiment_data and sentiment_data.get('confidence', 0) > 0.7:
            umbral_base = 2.5
        
        if puntos_compra >= umbral_base and puntos_compra > puntos_venta + 0.5:
            confianza = min(0.95, (puntos_compra / 9) + 0.25)  # Normalizado por más factores
            return "BUY", confianza, razones
        elif puntos_venta >= umbral_base and puntos_venta > puntos_compra + 0.5:
            confianza = min(0.95, (puntos_venta / 9) + 0.25)
            return "SELL", confianza, razones
        
        return None, 0, razones
        
    except Exception as e:
        logger.error(f"Error evaluating signal: {e}")
        return None, 0, ["Error en análisis"]

def calcular_gestion_riesgo(precio, accion, confianza, volatility=0.02):
    """Gestion avanzada de riesgo con volatilidad"""
    try:
        balance_simulado = 10000  # $10,000 para paper trading
        riesgo_base = 0.02  # 2% base
        
        # Ajustar riesgo por volatilidad
        riesgo_ajustado_vol = riesgo_base * (1 + volatility * 2)
        riesgo_ajustado_vol = min(riesgo_ajustado_vol, 0.05)  # Máximo 5%
        
        # Factor de confianza
        factor_confianza = min(confianza * 1.2, 1.0)
        riesgo_final = riesgo_ajustado_vol * factor_confianza
        
        # Stop loss y take profit dinámicos basados en volatilidad
        vol_multiplier = max(1, volatility * 20)
        
        if accion == "BUY":
            stop_loss = precio * (1 - 0.03 * vol_multiplier)
            take_profit = precio * (1 + 0.06 * vol_multiplier)
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
    """ENDPOINT PRINCIPAL - Analisis completo con sentimiento integrado"""
    
    # Verificar horario
    market_open = es_horario_mercado()
    force_analysis = request.args.get('force', 'false').lower() == 'true'
    include_sentiment = request.args.get('sentiment', 'true').lower() == 'true'
    
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
    sentiment_analyzed = 0
    
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
            
            # ANÁLISIS DE SENTIMIENTO (solo para símbolos prioritarios o si se especifica)
            sentiment_data = None
            if include_sentiment and symbol in SENTIMENT_PRIORITY_SYMBOLS:
                try:
                    sentiment_data = analizar_sentimiento_noticias(symbol)
                    if sentiment_data and sentiment_data.get('news_count', 0) > 0:
                        sentiment_analyzed += 1
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed for {symbol}: {e}")
            
            # Evaluar señal con sentimiento integrado
            action, confidence, reasons = evaluar_senal_avanzada(indicators, sentiment_data)
            
            # Umbral de confianza configurable (más bajo si tenemos sentimiento)
            min_confidence = float(request.args.get('min_confidence', 0.35))
            if sentiment_data and sentiment_data.get('confidence', 0) > 0.6:
                min_confidence = max(0.25, min_confidence - 0.1)  # Reducir umbral con buen sentimiento
            
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
                
                signal_data = {
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
                }
                
                # Agregar datos de sentimiento si están disponibles
                if sentiment_data and sentiment_data.get('news_count', 0) > 0:
                    signal_data["sentiment"] = {
                        "sentiment_score": sentiment_data['sentiment_score'],
                        "sentiment_label": sentiment_data['sentiment_label'],
                        "news_count": sentiment_data['news_count'],
                        "confidence": sentiment_data['confidence'],
                        "top_news": sentiment_data.get('news_articles', [])[:3]  # Solo top 3
                    }
                
                signals.append(signal_data)
                
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
        "sentiment_analyzed": sentiment_analyzed,
        "market_status": "OPEN" if market_open else "CLOSED",
        "analysis_time": datetime.now().isoformat(),
        "errors": errors[:5],  # Solo primeros 5 errores
        "cache_status": "cached" if f"analyze_market_{str(())}_{str({})}" in CACHE else "fresh",
        "sentiment_enabled": include_sentiment
    })

@app.route('/analyze/<symbol>')
def analyze_single_stock(symbol):
    """Analizar una acción específica en detalle con sentimiento"""
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
        
        # Análisis de sentimiento (siempre incluido para análisis individual)
        sentiment_data = None
        include_sentiment = request.args.get('sentiment', 'true').lower() == 'true'
        if include_sentiment:
            try:
                sentiment_data = analizar_sentimiento_noticias(symbol)
            except Exception as e:
                logger.warning(f"Sentiment analysis failed for {symbol}: {e}")
                sentiment_data = {
                    'sentiment_score': 0,
                    'news_count': 0,
                    'sentiment_label': 'NO_DATA',
                    'confidence': 0,
                    'error': str(e)
                }
        
        # Señal con sentimiento integrado
        action, confidence, reasons = evaluar_senal_avanzada(indicators, sentiment_data)
        
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
        
        # Agregar datos de sentimiento completos
        if sentiment_data and sentiment_data.get('news_count', 0) > 0:
            response_data["sentiment"] = sentiment_data
        elif include_sentiment:
            response_data["sentiment"] = {
                "sentiment_score": 0,
                "sentiment_label": "NO_DATA",
                "news_count": 0,
                "confidence": 0,
                "message": "No hay datos de noticias disponibles"
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sentiment/<symbol>')
def analyze_sentiment_only(symbol):
    """Endpoint específico para análisis de sentimiento detallado"""
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

@app.route('/sentiment/batch')
def analyze_sentiment_batch():
    """Análisis de sentimiento para múltiples símbolos"""
    try:
        symbols_param = request.args.get('symbols', '')
        if not symbols_param:
            symbols_to_analyze = SENTIMENT_PRIORITY_SYMBOLS
        else:
            symbols_to_analyze = [s.strip().upper() for s in symbols_param.split(',') if s.strip()]
        
        batch_results = {}
        errors = []
        
        for symbol in symbols_to_analyze[:10]:  # Máximo 10 para evitar rate limiting
            try:
                rate_limit()
                sentiment_data = analizar_sentimiento_noticias(symbol)
                batch_results[symbol] = sentiment_data
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
                continue
        
        # Análisis agregado
        total_news = sum(data.get('news_count', 0) for data in batch_results.values())
        avg_sentiment = np.mean([data.get('sentiment_score', 0) for data in batch_results.values() if data.get('news_count', 0) > 0])
        
        positive_count = sum(1 for data in batch_results.values() if data.get('sentiment_label') in ['POSITIVO', 'MUY_POSITIVO'])
        negative_count = sum(1 for data in batch_results.values() if data.get('sentiment_label') in ['NEGATIVO', 'MUY_NEGATIVO'])
        
        market_sentiment = "OPTIMISTA" if positive_count > negative_count else "PESIMISTA" if negative_count > positive_count else "NEUTRAL"
        
        return jsonify({
            "symbols_analyzed": list(batch_results.keys()),
            "individual_results": batch_results,
            "market_overview": {
                "total_news_analyzed": total_news,
                "average_sentiment_score": round(avg_sentiment, 3) if not np.isnan(avg_sentiment) else 0,
                "market_sentiment": market_sentiment,
                "positive_stocks": positive_count,
                "negative_stocks": negative_count,
                "neutral_stocks": len(batch_results) - positive_count - negative_count
            },
            "errors": errors,
            "analysis_time": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {e}")
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
    """Vista general del mercado con sentimiento"""
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
        
        # Análisis de sentimiento general del mercado (opcional)
        include_sentiment = request.args.get('sentiment', 'false').lower() == 'true'
        market_sentiment_data = None
        
        if include_sentiment:
            try:
                # Analizar sentimiento de algunos índices principales
                sentiment_results = []
                for symbol in ['SPY', 'QQQ', 'DIA']:  # ETFs principales
                    try:
                        sentiment = analizar_sentimiento_noticias(symbol)
                        if sentiment.get('news_count', 0) > 0:
                            sentiment_results.append(sentiment)
                    except:
                        continue
                
                if sentiment_results:
                    avg_market_sentiment = np.mean([s['sentiment_score'] for s in sentiment_results])
                    total_market_news = sum(s['news_count'] for s in sentiment_results)
                    
                    market_sentiment_data = {
                        'average_score': round(avg_market_sentiment, 3),
                        'total_news': total_market_news,
                        'label': 'POSITIVO' if avg_market_sentiment > 0.5 else 'NEGATIVO' if avg_market_sentiment < -0.5 else 'NEUTRAL'
                    }
            except Exception as e:
                logger.warning(f"Error getting market sentiment: {e}")
        
        # Análisis de sentimiento basado en índices
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
        
        response = {
            "indices": market_data,
            "market_sentiment": sentiment,
            "sentiment_emoji": sentiment_emoji,
            "market_status": "OPEN" if es_horario_mercado() else "CLOSED",
            "timestamp": datetime.now().isoformat()
        }
        
        if market_sentiment_data:
            response["news_sentiment"] = market_sentiment_data
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in market overview: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """Health check mejorado con validación de APIs"""
    try:
        # Test básico de conectividad con yfinance
        test_stock = yf.Ticker('AAPL')
        test_data = test_stock.history(period='1d')
        yfinance_status = "OK" if len(test_data) > 0 else "WARNING"
        
        # Test básico de NewsAPI (solo si hay API key configurada)
        newsapi_status = "NOT_CONFIGURED"
        try:
            # Intentar una llamada simple para verificar conectividad
            test_sentiment = analizar_sentimiento_noticias('AAPL')
            if test_sentiment.get('sentiment_label') != 'ERROR':
                newsapi_status = "OK" if test_sentiment.get('news_count', 0) > 0 else "NO_DATA"
            else:
                newsapi_status = "ERROR"
        except:
            newsapi_status = "ERROR"
        
        return jsonify({
            "status": "OK",
            "timestamp": datetime.now().isoformat(),
            "market_open": es_horario_mercado(),
            "symbols_count": len(SYMBOLS),
            "priority_symbols_count": len(SENTIMENT_PRIORITY_SYMBOLS),
            "version": "2.2-sentiment",
            "apis": {
                "yfinance": yfinance_status,
                "newsapi": newsapi_status
            },
            "cache_size": len(CACHE),
            "features": {
                "technical_analysis": True,
                "risk_management": True,
                "correlation_analysis": True,
                "sentiment_analysis": True,
                "market_overview": True,
                "batch_sentiment": True
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
        "version": "2.2-sentiment",
        "sentiment_enabled": True
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

# Endpoint para configurar API key de NewsAPI
@app.route('/admin/config', methods=['POST'])
def update_config():
    """Actualizar configuración (como API keys)"""
    try:
        data = request.get_json()
        
        # En una implementación real, esto debería guardarse de forma segura
        # Por ahora solo mostramos cómo sería la estructura
        
        return jsonify({
            "message": "Configuración actualizada",
            "timestamp": datetime.now().isoformat(),
            "note": "Implementar almacenamiento seguro de API keys en producción"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    
    # Mostrar información de inicio
    print(f"🚀 Sistema de Trading con Análisis de Sentimiento v2.2")
    print(f"📊 Símbolos configurados: {len(SYMBOLS)}")
    print(f"📰 Símbolos prioritarios para sentimiento: {len(SENTIMENT_PRIORITY_SYMBOLS)}")
    print(f"🌐 Puerto: {port}")
    print(f"⚠️  Recuerda configurar tu NewsAPI key en la función analizar_sentimiento_noticias")
    
    app.run(host='0.0.0.0', port=port, debug=False)
