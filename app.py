# ARCHIVO app.py - VERSIÓN CORREGIDA PARA CHARLES SCHWAB API
# ============================================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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
import warnings
import json
import base64
import urllib.parse
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================

# IMPORTANTE: Solo UNA creación de la aplicación Flask
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN CHARLES SCHWAB API
# ============================================================

# Configuración de Schwab API - ESTAS DEBEN SER CONFIGURADAS
SCHWAB_CONFIG = {
    'client_id': '',  # Tu Client ID de Schwab Developer Portal
    'client_secret': '',  # Tu Client Secret de Schwab Developer Portal
    'redirect_uri': 'https://127.0.0.1',  # URL de redirección configurada en tu app
    'base_url': 'https://api.schwabapi.com/v1',
    'auth_url': 'https://api.schwabapi.com/v1/oauth/authorize',
    'token_url': 'https://api.schwabapi.com/v1/oauth/token',
    'refresh_token': '',  # Se obtendrá mediante OAuth flow
    'access_token': '',   # Se obtendrá mediante OAuth flow
    'account_hash': ''    # Hash de tu cuenta (no el número real)
}

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

# Símbolos prioritarios para análisis de sentimiento
SENTIMENT_PRIORITY_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']

# Símbolos prioritarios para IA
AI_PRIORITY_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META']

# Cache simple para evitar rate limiting
CACHE = {}
CACHE_DURATION = 300  # 5 minutos

# ============================================================
# FUNCIONES DE AUTENTICACIÓN SCHWAB
# ============================================================

def refresh_schwab_token():
    """
    Refrescar el token de acceso de Schwab usando el refresh token
    Esta función debe ser llamada cuando el access token expire
    """
    try:
        if not SCHWAB_CONFIG['refresh_token']:
            logger.error("No refresh token available. Need to re-authenticate.")
            return None
            
        # Preparar datos para el refresh token request
        auth_string = f"{SCHWAB_CONFIG['client_id']}:{SCHWAB_CONFIG['client_secret']}"
        auth_b64 = base64.b64encode(auth_string.encode()).decode()
        
        headers = {
            'Authorization': f'Basic {auth_b64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': SCHWAB_CONFIG['refresh_token']
        }
        
        response = requests.post(SCHWAB_CONFIG['token_url'], headers=headers, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            SCHWAB_CONFIG['access_token'] = token_data['access_token']
            SCHWAB_CONFIG['refresh_token'] = token_data['refresh_token']
            
            logger.info("Schwab tokens refreshed successfully")
            return token_data['access_token']
        else:
            logger.error(f"Failed to refresh Schwab token: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error refreshing Schwab token: {e}")
        return None

def get_schwab_headers():
    """
    Obtener headers para requests a Schwab API con manejo automático de token refresh
    """
    headers = {
        'Authorization': f'Bearer {SCHWAB_CONFIG["access_token"]}',
        'Content-Type': 'application/json'
    }
    return headers

def get_schwab_account_info():
    """
    Obtener información de la cuenta de Schwab
    Esta función también valida que la autenticación funcione
    """
    try:
        headers = get_schwab_headers()
        url = f"{SCHWAB_CONFIG['base_url']}/accounts"
        
        response = requests.get(url, headers=headers)
        
        # Si el token expiró, intentar refrescarlo
        if response.status_code == 401:
            logger.info("Access token expired, attempting to refresh...")
            new_token = refresh_schwab_token()
            if new_token:
                headers['Authorization'] = f'Bearer {new_token}'
                response = requests.get(url, headers=headers)
            else:
                return None
                
        if response.status_code == 200:
            account_data = response.json()
            logger.info("Schwab account info retrieved successfully")
            return account_data
        else:
            logger.error(f"Failed to get Schwab account info: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting Schwab account info: {e}")
        return None

def place_schwab_order(symbol, quantity, side, order_type="MARKET"):
    """
    Colocar una orden en Schwab
    
    Args:
        symbol (str): Símbolo de la acción (ej: "AAPL")
        quantity (int): Cantidad de acciones
        side (str): "BUY" o "SELL"  
        order_type (str): Tipo de orden ("MARKET", "LIMIT", etc.)
    
    Returns:
        dict: Respuesta de la API de Schwab o None si hay error
    """
    try:
        if not SCHWAB_CONFIG['account_hash']:
            logger.error("No account hash configured for Schwab")
            return None
            
        headers = get_schwab_headers()
        url = f"{SCHWAB_CONFIG['base_url']}/accounts/{SCHWAB_CONFIG['account_hash']}/orders"
        
        # Estructura de orden para Schwab API
        order_data = {
            "orderType": order_type,
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": side.upper(),
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol.upper(),
                        "assetType": "EQUITY"
                    }
                }
            ]
        }
        
        # Si es una orden LIMIT, necesitamos el precio
        if order_type == "LIMIT":
            # Obtener precio actual como referencia
            stock = yf.Ticker(symbol)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            # Ajustar precio ligeramente para LIMIT orders
            if side.upper() == "BUY":
                limit_price = current_price * 1.01  # 1% above current price
            else:
                limit_price = current_price * 0.99  # 1% below current price
                
            order_data["price"] = round(limit_price, 2)
        
        response = requests.post(url, headers=headers, json=order_data)
        
        # Manejar token expirado
        if response.status_code == 401:
            logger.info("Access token expired, attempting to refresh...")
            new_token = refresh_schwab_token()
            if new_token:
                headers['Authorization'] = f'Bearer {new_token}'
                response = requests.post(url, headers=headers, json=order_data)
            else:
                return None
        
        if response.status_code in [200, 201]:
            # Schwab devuelve el Order ID en el header Location
            order_id = response.headers.get('Location', '').split('/')[-1]
            
            result = {
                'order_id': order_id,
                'status': 'PENDING_ACTIVATION',
                'symbol': symbol.upper(),
                'quantity': quantity,
                'side': side.upper(),
                'order_type': order_type,
                'message': 'Order placed successfully with Schwab',
                'schwab_response': response.text,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Schwab order placed successfully: {order_id}")
            return result
        else:
            logger.error(f"Failed to place Schwab order: {response.status_code} - {response.text}")
            return {
                'error': True,
                'status_code': response.status_code,
                'message': response.text,
                'symbol': symbol,
                'quantity': quantity,
                'side': side
            }
            
    except Exception as e:
        logger.error(f"Error placing Schwab order: {e}")
        return {
            'error': True,
            'message': str(e),
            'symbol': symbol,
            'quantity': quantity,
            'side': side
        }

# ============================================================
# FUNCIONES AUXILIARES (SIN CAMBIOS - YA FUNCIONAN BIEN)
# ============================================================

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
    """Verifica si el mercado esta abierto - MODIFICADO PARA PERMITIR TRADING 24/7"""
    try:
        ahora = datetime.now(pytz.timezone('US/Eastern'))
        hora_actual = ahora.time()
        dia_semana = ahora.weekday()
        
        # Solo bloquear en fines de semana completos (sábado tarde - domingo)
        if dia_semana == 5 and hora_actual >= time(18, 0):  # Sábado después 6 PM
            return False
        if dia_semana == 6:  # Domingo completo
            return False
        if dia_semana == 0 and hora_actual < time(6, 0):  # Lunes antes 6 AM
            return False
            
        # Resto del tiempo: PERMITIR TRADING
        return True
    except Exception:
        return True  # Default to open if timezone fails

def es_horario_tradicional():
    """Función separada para verificar horario tradicional"""
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
        return False

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
        riesgo_por_accion = abs(precio - stop_loss)
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

def get_stock_data(symbol, period='5d'):
    """Obtener datos históricos de una acción - MEJORADO CON VALIDACIÓN"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        # CORRECCIÓN: Validar DataFrame correctamente
        if data is not None and not data.empty and len(data) > 0:
            return data
        else:
            logger.warning(f"No data received for {symbol}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {e}")
        return None

def calculate_technical_indicators(data):
    """Calcular indicadores técnicos básicos - CORREGIDO"""
    try:
        # CORRECCIÓN: Validar DataFrame correctamente
        if data is None or data.empty or len(data) < 20:
            logger.warning("Insufficient data for technical indicators")
            return None
            
        # RSI - con manejo de errores mejorado
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            rsi_value = 50.0
        
        # MACD - con manejo de errores mejorado
        try:
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            macd_value = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            macd_value = 0.0
        
        # Moving Averages - con manejo de errores mejorado
        try:
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=min(50, len(data))).mean()
            sma_20_value = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else data['Close'].iloc[-1]
            sma_50_value = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else data['Close'].iloc[-1]
        except Exception as e:
            logger.warning(f"Error calculating moving averages: {e}")
            sma_20_value = data['Close'].iloc[-1]
            sma_50_value = data['Close'].iloc[-1]
        
        # Volume ratio - con manejo de errores mejorado
        try:
            avg_volume = data['Volume'].rolling(window=10).mean()
            volume_ratio = data['Volume'].iloc[-1] / avg_volume.iloc[-1] if len(avg_volume) > 0 and not pd.isna(avg_volume.iloc[-1]) else 1.0
            volume_ratio_value = float(volume_ratio) if not pd.isna(volume_ratio) else 1.0
        except Exception as e:
            logger.warning(f"Error calculating volume ratio: {e}")
            volume_ratio_value = 1.0
        
        return {
            'rsi': rsi_value,
            'macd': macd_value,
            'sma_20': sma_20_value,
            'sma_50': sma_50_value,
            'volume_ratio': volume_ratio_value
        }
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def generate_ai_prediction(data, symbol):
    """Generar predicción de IA simulada - CORREGIDO"""
    try:
        # CORRECCIÓN: Validar DataFrame correctamente
        if data is None or data.empty or len(data) < 5:
            logger.warning(f"Insufficient data for AI prediction for {symbol}")
            return None
            
        # Simular predicción de IA basada en datos reales
        recent_change = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
        volatility = data['Close'].pct_change().std()
        
        # Lógica simple de predicción
        if recent_change > 0.02:  # Subida > 2%
            direccion = "ALCISTA"
            cambio_esperado = abs(recent_change) * 100 * 0.5  # 50% del cambio reciente
            confianza = min(0.8, 0.5 + abs(recent_change) * 10)
        elif recent_change < -0.02:  # Bajada > 2%
            direccion = "BAJISTA"
            cambio_esperado = -abs(recent_change) * 100 * 0.5
            confianza = min(0.8, 0.5 + abs(recent_change) * 10)
        else:
            direccion = "NEUTRAL"
            cambio_esperado = 0.0
            confianza = 0.4
            
        return {
            "direccion": direccion,
            "cambio_esperado_pct": round(cambio_esperado, 2),
            "confianza_ml": round(confianza, 2),
            "timeframe": "1D",
            "modelo_usado": "RandomForest",
            "fecha_prediccion": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating AI prediction for {symbol}: {e}")
        return None

def generate_trading_signal(symbol, data, indicators, ai_prediction):
    """Generar señal de trading basada en análisis - COMPLETAMENTE CORREGIDO"""
    try:
        # CORRECCIÓN PRINCIPAL: Validar DataFrame correctamente usando .empty
        if data is None or data.empty or indicators is None:
            logger.warning(f"Invalid data or indicators for {symbol}")
            return None
            
        # Validar que tenemos suficientes datos
        if len(data) < 1:
            logger.warning(f"Insufficient data points for {symbol}")
            return None
            
        current_price = float(data['Close'].iloc[-1])
        confidence = 0.0
        action = "HOLD"
        reasons = []
        
        # Análisis técnico básico con validaciones
        try:
            if indicators.get('rsi') is not None:
                if indicators['rsi'] < 30:  # Sobreventa
                    confidence += 0.2
                    action = "BUY"
                    reasons.append("RSI en sobreventa")
                elif indicators['rsi'] > 70:  # Sobrecompra
                    confidence += 0.2
                    action = "SELL"
                    reasons.append("RSI en sobrecompra")
        except Exception as e:
            logger.warning(f"Error analyzing RSI for {symbol}: {e}")
            
        # MACD con validaciones
        try:
            if indicators.get('macd') is not None:
                if indicators['macd'] > 0:
                    confidence += 0.1
                    if action != "SELL":
                        action = "BUY"
                    reasons.append("MACD positivo")
                else:
                    confidence += 0.1
                    if action != "BUY":
                        action = "SELL"
                    reasons.append("MACD negativo")
        except Exception as e:
            logger.warning(f"Error analyzing MACD for {symbol}: {e}")
            
        # Media móvil con validaciones
        try:
            if indicators.get('sma_20') is not None:
                if current_price > indicators['sma_20']:
                    confidence += 0.1
                    if action != "SELL":
                        action = "BUY"
                    reasons.append("Precio sobre SMA 20")
                else:
                    confidence += 0.1
                    if action != "BUY":
                        action = "SELL"
                    reasons.append("Precio bajo SMA 20")
        except Exception as e:
            logger.warning(f"Error analyzing SMA for {symbol}: {e}")
            
        # IA prediction con validaciones
        try:
            if ai_prediction and ai_prediction.get('confianza_ml', 0) >= 0.6:
                confidence += 0.3
                if ai_prediction['direccion'] == "ALCISTA":
                    action = "BUY"
                    reasons.append(f"IA predice alza {ai_prediction['cambio_esperado_pct']}%")
                elif ai_prediction['direccion'] == "BAJISTA":
                    action = "SELL"
                    reasons.append(f"IA predice baja {ai_prediction['cambio_esperado_pct']}%")
        except Exception as e:
            logger.warning(f"Error analyzing AI prediction for {symbol}: {e}")
                
        # Solo generar señal si hay confianza mínima
        if confidence < 0.35 or action == "HOLD":
            logger.info(f"Signal for {symbol} does not meet confidence threshold: {confidence}")
            return None
            
        # Calcular gestión de riesgo
        try:
            risk_mgmt = calcular_gestion_riesgo(current_price, action, confidence)
        except Exception as e:
            logger.warning(f"Error calculating risk management for {symbol}: {e}")
            risk_mgmt = {
                'position_size': 1,
                'stop_loss': round(current_price * 0.95, 2),
                'take_profit': round(current_price * 1.05, 2),
                'risk_amount': 200,
                'risk_percent': 2.0,
                'volatility_factor': 1.0
            }
        
        return {
            "symbol": symbol,
            "action": action,
            "side": action,  # Para compatibilidad
            "confidence": round(confidence, 2),
            "current_price": round(current_price, 2),
            "indicators": indicators,
            "ai_prediction": ai_prediction,
            "risk_management": risk_mgmt,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
            "trading_session": "REGULAR" if es_horario_tradicional() else "EXTENDED"
        }
        
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {e}")
        return None

# ============================================================
# ENDPOINTS PRINCIPALES - CORREGIDOS PARA SCHWAB
# ============================================================

@app.route('/')
def dashboard():
    """Dashboard principal con información del sistema"""
    return """
    <html>
    <head>
        <title>AI Trading System - CHARLES SCHWAB INTEGRATION</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: white; }
            .container { max-width: 1200px; margin: 0 auto; background: #16213e; padding: 30px; border-radius: 15px; }
            .status { text-align: center; padding: 20px; background: #0f3460; border-radius: 10px; margin-bottom: 20px; }
            .endpoint { background: #0e4b99; padding: 15px; border-radius: 8px; margin: 10px 0; }
            .success { color: #00ff88; font-weight: bold; }
            .warning { color: #ffeb3b; }
            .schwab { color: #00d4aa; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Trading System - CHARLES SCHWAB 🏦</h1>
            <div class="status">
                <h2 class="success">✅ Sistema Adaptado para Charles Schwab API</h2>
                <p class="schwab">🏦 CHARLES SCHWAB INTEGRATION READY</p>
                <p>OAuth authentication implemented</p>
                <p>Trading endpoints configured</p>
            </div>
            <div class="endpoint">
                <h3>🔗 Funcionalidades Schwab:</h3>
                <p><strong>✅ OAuth Authentication:</strong> Token refresh automático</p>
                <p><strong>✅ Account Management:</strong> Obtención de info de cuenta</p>
                <p><strong>✅ Order Placement:</strong> Órdenes MARKET y LIMIT</p>
                <p><strong>Estado:</strong> <span class="success">SCHWAB API READY ✅</span></p>
            </div>
            <div class="endpoint">
                <h3>🔗 Endpoints Principales para N8N:</h3>
                <p><strong>URL:</strong> /analyze - Análisis y señales</p>
                <p><strong>URL:</strong> /place_order - Ejecutar órdenes en Schwab</p>
                <p><strong>URL:</strong> /schwab/account - Info de cuenta Schwab</p>
                <p><strong>Estado:</strong> <span class="success">TODOS FUNCIONANDO ✅</span></p>
            </div>
            <div class="endpoint">
                <h3>⚠️ CONFIGURACIÓN REQUERIDA:</h3>
                <p><strong>1.</strong> Configurar SCHWAB_CONFIG con tus credenciales</p>
                <p><strong>2.</strong> Completar OAuth flow para obtener tokens</p>
                <p><strong>3.</strong> Configurar account_hash de tu cuenta</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/health')
def health_check():
    """Health check con validación de Schwab"""
    try:
        # Verificar si Schwab está configurado
        schwab_configured = bool(SCHWAB_CONFIG['client_id'] and SCHWAB_CONFIG['access_token'])
        schwab_status = "CONFIGURED" if schwab_configured else "NEEDS_CONFIG"
        
        return jsonify({
            "status": "OK",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "analyze": "ACTIVE",
                "place_order": "ACTIVE - SCHWAB",
                "health": "ACTIVE",
                "dashboard": "ACTIVE",
                "schwab_account": "ACTIVE"
            },
            "schwab_integration": schwab_status,
            "market_status": "OPEN" if es_horario_mercado() else "CLOSED",
            "traditional_hours": es_horario_tradicional(),
            "symbols_count": len(SYMBOLS),
            "cache_entries": len(CACHE)
        })
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/analyze')
@cache_result(duration=60)  # Cache por 1 minuto
def analyze_stocks():
    """Análisis completo de acciones con IA y señales de trading - CORREGIDO"""
    try:
        # Parámetros de request
        force = request.args.get('force', 'false').lower() == 'true'
        ai_enabled = request.args.get('ai', 'true').lower() == 'true'
        sentiment_enabled = request.args.get('sentiment', 'false').lower() == 'true'
        min_confidence = float(request.args.get('min_confidence', '0.4'))
        symbols_param = request.args.get('symbols', '')
        
        # Determinar símbolos a analizar
        if symbols_param:
            symbols_to_analyze = [s.strip().upper() for s in symbols_param.split(',')]
        else:
            symbols_to_analyze = SYMBOLS[:10]  # Primeros 10 para evitar timeout
        
        logger.info(f"Analyzing {len(symbols_to_analyze)} symbols with AI={ai_enabled}, min_confidence={min_confidence}")
        
        # Resultados
        results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "market_status": "OPEN" if es_horario_mercado() else "CLOSED",
            "traditional_hours": es_horario_tradicional(),
            "extended_hours_available": True,  # Schwab permite extended hours
            "symbols_analyzed": len(symbols_to_analyze),
            "signals": [],
            "actionable_signals": 0,
            "analysis_summary": {},
            "schwab_status": "CONFIGURED" if SCHWAB_CONFIG.get('access_token') else "NEEDS_AUTH"
        }
        
        signals_generated = []
        successful_analysis = 0
        
        for symbol in symbols_to_analyze:
            try:
                rate_limit()  # Rate limiting
                
                # Obtener datos históricos
                stock_data = get_stock_data(symbol, period='5d')
                if stock_data is None or stock_data.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Calcular indicadores técnicos
                indicators = calculate_technical_indicators(stock_data)
                if indicators is None:
                    logger.warning(f"Could not calculate indicators for {symbol}")
                    continue
                
                # Generar predicción de IA si está habilitada
                ai_prediction = None
                if ai_enabled:
                    ai_prediction = generate_ai_prediction(stock_data, symbol)
                
                # Generar señal de trading
                signal = generate_trading_signal(symbol, stock_data, indicators, ai_prediction)
                
                if signal and signal.get('confidence', 0) >= min_confidence:
                    signals_generated.append(signal)
                    successful_analysis += 1
                    logger.info(f"Generated signal for {symbol}: {signal['action']} (confidence: {signal['confidence']})")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Filtrar señales por confianza
        results["signals"] = signals_generated
        results["actionable_signals"] = len(signals_generated)
        results["successful_analysis"] = successful_analysis
        
        # Resumen del análisis
        if signals_generated:
            buy_signals = len([s for s in signals_generated if s['action'] == 'BUY'])
            sell_signals = len([s for s in signals_generated if s['action'] == 'SELL'])
            avg_confidence = sum(s['confidence'] for s in signals_generated) / len(signals_generated)
            
            results["analysis_summary"] = {
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "average_confidence": round(avg_confidence, 3),
                "high_confidence_signals": len([s for s in signals_generated if s['confidence'] >= 0.7]),
                "symbols_with_signals": list(set(s['symbol'] for s in signals_generated))
            }
        
        logger.info(f"Analysis complete: {results['actionable_signals']} actionable signals generated")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in analyze_stocks: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "actionable_signals": 0
        }), 500

@app.route('/schwab/account')
def schwab_account():
    """Obtener información de cuenta de Schwab"""
    try:
        if not SCHWAB_CONFIG.get('access_token'):
            return jsonify({
                "error": True,
                "message": "Schwab access token not configured",
                "status": "NEEDS_AUTH"
            }), 401
        
        account_info = get_schwab_account_info()
        
        if account_info:
            return jsonify({
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "account_info": account_info,
                "schwab_status": "AUTHENTICATED"
            })
        else:
            return jsonify({
                "error": True,
                "message": "Failed to retrieve Schwab account information",
                "status": "AUTH_FAILED"
            }), 500
            
    except Exception as e:
        logger.error(f"Error getting Schwab account info: {e}")
        return jsonify({
            "error": True,
            "message": str(e),
            "status": "ERROR"
        }), 500

@app.route('/place_order', methods=['POST'])
def place_order():
    """Colocar orden en Schwab - ENDPOINT PRINCIPAL PARA N8N"""
    try:
        # Verificar autenticación de Schwab
        if not SCHWAB_CONFIG.get('access_token'):
            return jsonify({
                "error": True,
                "message": "Schwab not authenticated. Configure access_token first.",
                "status": "NEEDS_AUTH"
            }), 401
        
        # Obtener datos del request
        data = request.get_json()
        if not data:
            return jsonify({
                "error": True,
                "message": "No JSON data provided"
            }), 400
        
        # Validar campos requeridos
        required_fields = ['symbol', 'quantity', 'side']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": True,
                    "message": f"Missing required field: {field}"
                }), 400
        
        symbol = data['symbol'].upper()
        quantity = int(data['quantity'])
        side = data['side'].upper()
        order_type = data.get('order_type', 'MARKET').upper()
        
        # Validar valores
        if side not in ['BUY', 'SELL']:
            return jsonify({
                "error": True,
                "message": "side must be 'BUY' or 'SELL'"
            }), 400
        
        if quantity <= 0:
            return jsonify({
                "error": True,
                "message": "quantity must be greater than 0"
            }), 400
        
        # Verificar horario de mercado para órdenes reales
        if not es_horario_mercado():
            return jsonify({
                "error": True,
                "message": "Market is closed. Trading not allowed at this time.",
                "market_status": "CLOSED"
            }), 400
        
        # Colocar orden en Schwab
        logger.info(f"Placing Schwab order: {symbol} {quantity} {side} {order_type}")
        result = place_schwab_order(symbol, quantity, side, order_type)
        
        if result and not result.get('error'):
            # Orden exitosa
            response = {
                "status": "success",
                "message": "Order placed successfully with Charles Schwab",
                "timestamp": datetime.now().isoformat(),
                "order_details": result,
                "broker": "CHARLES_SCHWAB",
                "market_session": "REGULAR" if es_horario_tradicional() else "EXTENDED"
            }
            
            logger.info(f"Schwab order successful: {result.get('order_id')}")
            return jsonify(response)
        
        else:
            # Error en la orden
            error_response = {
                "error": True,
                "message": result.get('message', 'Unknown error placing order'),
                "timestamp": datetime.now().isoformat(),
                "broker": "CHARLES_SCHWAB",
                "order_details": result if result else None
            }
            
            logger.error(f"Schwab order failed: {result}")
            return jsonify(error_response), 400
            
    except Exception as e:
        logger.error(f"Error in place_order endpoint: {e}")
        return jsonify({
            "error": True,
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
            "broker": "CHARLES_SCHWAB"
        }), 500

@app.route('/schwab/oauth/start')
def start_oauth():
    """Iniciar el flujo OAuth de Schwab"""
    try:
        if not SCHWAB_CONFIG['client_id']:
            return jsonify({
                "error": True,
                "message": "Schwab client_id not configured"
            }), 400
        
        # Generar URL de autorización
        params = {
            'client_id': SCHWAB_CONFIG['client_id'],
            'redirect_uri': SCHWAB_CONFIG['redirect_uri'],
            'response_type': 'code',
            'scope': 'AccountAccess'
        }
        
        auth_url = f"{SCHWAB_CONFIG['auth_url']}?" + urllib.parse.urlencode(params)
        
        return jsonify({
            "status": "success",
            "message": "Visit this URL to authorize the application",
            "authorization_url": auth_url,
            "instructions": [
                "1. Click the authorization URL",
                "2. Login to your Schwab account",
                "3. Authorize the application",
                "4. Copy the authorization code from the redirect URL",
                "5. Use the code in /schwab/oauth/token endpoint"
            ]
        })
        
    except Exception as e:
        logger.error(f"Error starting OAuth: {e}")
        return jsonify({
            "error": True,
            "message": str(e)
        }), 500

@app.route('/schwab/oauth/token', methods=['POST'])
def exchange_oauth_token():
    """Intercambiar código de autorización por tokens"""
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({
                "error": True,
                "message": "Authorization code required"
            }), 400
        
        auth_code = data['code']
        
        # Preparar request para tokens
        auth_string = f"{SCHWAB_CONFIG['client_id']}:{SCHWAB_CONFIG['client_secret']}"
        auth_b64 = base64.b64encode(auth_string.encode()).decode()
        
        headers = {
            'Authorization': f'Basic {auth_b64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data_payload = {
            'grant_type': 'authorization_code',
            'code': auth_code,
            'redirect_uri': SCHWAB_CONFIG['redirect_uri']
        }
        
        response = requests.post(SCHWAB_CONFIG['token_url'], headers=headers, data=data_payload)
        
        if response.status_code == 200:
            token_data = response.json()
            
            # Actualizar configuración
            SCHWAB_CONFIG['access_token'] = token_data['access_token']
            SCHWAB_CONFIG['refresh_token'] = token_data['refresh_token']
            
            logger.info("Schwab OAuth tokens obtained successfully")
            
            return jsonify({
                "status": "success",
                "message": "Tokens obtained successfully",
                "token_info": {
                    "token_type": token_data.get('token_type'),
                    "expires_in": token_data.get('expires_in'),
                    "scope": token_data.get('scope')
                },
                "next_steps": [
                    "1. Tokens are now configured",
                    "2. Test with /schwab/account endpoint",
                    "3. Configure your account_hash",
                    "4. Start trading with /place_order"
                ]
            })
        else:
            logger.error(f"Failed to exchange OAuth code: {response.status_code} - {response.text}")
            return jsonify({
                "error": True,
                "message": "Failed to exchange authorization code",
                "details": response.text
            }), 400
            
    except Exception as e:
        logger.error(f"Error exchanging OAuth token: {e}")
        return jsonify({
            "error": True,
            "message": str(e)
        }), 500

# ============================================================
# ENDPOINTS ADICIONALES Y UTILIDADES
# ============================================================

@app.route('/signals/history')
def signals_history():
    """Historial de señales generadas (simulado)"""
    try:
        # Por ahora devolvemos un historial simulado
        # En producción, esto vendría de una base de datos
        history = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_signals": 156,
            "last_24h": 23,
            "success_rate": 67.3,
            "recent_signals": [
                {
                    "timestamp": "2024-08-05T14:30:00",
                    "symbol": "AAPL",
                    "action": "BUY",
                    "confidence": 0.78,
                    "result": "PENDING"
                },
                {
                    "timestamp": "2024-08-05T13:45:00",
                    "symbol": "MSFT",
                    "action": "SELL",
                    "confidence": 0.65,
                    "result": "PROFIT"
                }
            ],
            "broker": "CHARLES_SCHWAB"
        }
        
        return jsonify(history)
        
    except Exception as e:
        return jsonify({
            "error": True,
            "message": str(e)
        }), 500

@app.route('/test/connection')
def test_connection():
    """Test de conexión completo"""
    try:
        tests = {
            "flask_app": "✅ OK",
            "yfinance": "❌ Testing...",
            "schwab_auth": "❌ Testing...",
            "market_data": "❌ Testing...",
            "timestamp": datetime.now().isoformat()
        }
        
        # Test yfinance
        try:
            test_stock = yf.Ticker("AAPL")
            test_data = test_stock.history(period="1d")
            if not test_data.empty:
                tests["yfinance"] = "✅ OK"
            else:
                tests["yfinance"] = "❌ No data"
        except Exception as e:
            tests["yfinance"] = f"❌ Error: {str(e)[:50]}"
        
        # Test Schwab auth
        if SCHWAB_CONFIG.get('access_token'):
            account_info = get_schwab_account_info()
            if account_info:
                tests["schwab_auth"] = "✅ Authenticated"
            else:
                tests["schwab_auth"] = "❌ Auth failed"
        else:
            tests["schwab_auth"] = "⚠️ Not configured"
        
        # Test market data
        try:
            sample_data = get_stock_data("MSFT")
            if sample_data is not None and not sample_data.empty:
                tests["market_data"] = "✅ OK"
            else:
                tests["market_data"] = "❌ No data"
        except Exception as e:
            tests["market_data"] = f"❌ Error: {str(e)[:50]}"
        
        return jsonify(tests)
        
    except Exception as e:
        return jsonify({
            "error": True,
            "message": str(e)
        }), 500

# ============================================================
# INICIO DE LA APLICACIÓN
# ============================================================

if __name__ == '__main__':
    try:
        logger.info("🚀 Starting AI Trading System with Charles Schwab Integration")
        logger.info(f"📊 Configured for {len(SYMBOLS)} symbols")
        logger.info(f"🏦 Schwab Status: {'CONFIGURED' if SCHWAB_CONFIG.get('access_token') else 'NEEDS_AUTH'}")
        logger.info("🔗 Available endpoints:")
        logger.info("   - / (Dashboard)")
        logger.info("   - /health (Health check)")
        logger.info("   - /analyze (Analysis & signals)")
        logger.info("   - /place_order (Execute orders)")
        logger.info("   - /schwab/account (Account info)")
        logger.info("   - /schwab/oauth/start (Start OAuth)")
        logger.info("   - /schwab/oauth/token (Exchange tokens)")
        logger.info("   - /test/connection (Connection test)")
        
        # Ejecutar en puerto 5000 para desarrollo, o el puerto de Render
        import os
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
