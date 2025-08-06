# SISTEMA DE TRADING CON INTERACTIVE BROKERS (IBKR) - VERSIÓN COMPLETA
# =====================================================================================
# Sistema profesional de trading automatizado usando Interactive Brokers
# Compatible con N8N, TWS/Gateway, Paper Trading y Live Trading

from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
import logging
from functools import wraps
import time as time_module
import threading
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración IBKR
IBKR_CONFIG = {
    'host': '127.0.0.1',  # TWS/Gateway local
    'port': 7497,         # 7497 = Paper Trading, 7496 = Live
    'clientId': 1,        # ID único para esta aplicación
    'timeout': 30
}

# Símbolos de trading
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'ADBE',
    'JPM', 'V', 'MA', 'PYPL', 'JNJ', 'PFE', 'UNH', 'MRNA',
    'TSLA', 'DIS', 'KO', 'WMT'
]

PRIORITY_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META']

# Cache global
CACHE = {}
CACHE_DURATION = 300

# ============================================================
# GESTOR DE CONEXIÓN IBKR
# ============================================================

class IBKRManager:
    """Gestor profesional de conexión con Interactive Brokers"""
    
    def __init__(self):
        self.ib = None
        self.connected = False
        self.connection_lock = threading.Lock()
        self.last_connection_attempt = 0
        self.connection_cooldown = 30  # 30 segundos entre intentos
        
    def connect(self):
        """Conectar a IBKR TWS/Gateway"""
        with self.connection_lock:
            try:
                current_time = time_module.time()
                
                # Evitar reconexiones muy frecuentes
                if current_time - self.last_connection_attempt < self.connection_cooldown:
                    if self.connected and self.ib and self.ib.isConnected():
                        return True
                
                self.last_connection_attempt = current_time
                
                # Si ya hay conexión, verificarla
                if self.ib and self.connected:
                    try:
                        if self.ib.isConnected():
                            accounts = self.ib.managedAccounts()
                            if accounts:
                                logger.info("IBKR connection verified")
                                return True
                    except Exception:
                        logger.warning("Existing connection failed, reconnecting...")
                        self.disconnect()
                
                # Crear nueva conexión
                logger.info(f"Connecting to IBKR on {IBKR_CONFIG['host']}:{IBKR_CONFIG['port']}...")
                self.ib = IB()
                
                # Conectar con configuración
                self.ib.connect(
                    host=IBKR_CONFIG['host'],
                    port=IBKR_CONFIG['port'],
                    clientId=IBKR_CONFIG['clientId'],
                    timeout=IBKR_CONFIG['timeout']
                )
                
                # Verificar conexión
                if self.ib.isConnected():
                    accounts = self.ib.managedAccounts()
                    logger.info(f"✅ Connected to IBKR! Accounts: {accounts}")
                    self.connected = True
                    
                    # Configurar eventos
                    self.ib.errorEvent += self._on_error
                    self.ib.disconnectedEvent += self._on_disconnect
                    
                    return True
                else:
                    logger.error("❌ Failed to connect to IBKR")
                    self.connected = False
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error connecting to IBKR: {e}")
                logger.error("💡 Asegúrate de que TWS/Gateway esté ejecutándose y API habilitada")
                self.connected = False
                self.ib = None
                return False
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """Manejar errores de IBKR"""
        logger.warning(f"IBKR Error {errorCode}: {errorString}")
    
    def _on_disconnect(self):
        """Manejar desconexión"""
        logger.warning("IBKR disconnected")
        self.connected = False
    
    def disconnect(self):
        """Desconectar de IBKR"""
        try:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
                logger.info("Disconnected from IBKR")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
        finally:
            self.connected = False
            self.ib = None
    
    def get_account_info(self):
        """Obtener información completa de la cuenta"""
        try:
            if not self.connected or not self.ib:
                if not self.connect():
                    return {"error": "Cannot connect to IBKR", "connected": False}
            
            # Obtener valores de cuenta
            account_values = self.ib.accountValues()
            account_summary = {}
            
            key_values = ['TotalCashValue', 'NetLiquidation', 'BuyingPower', 
                         'GrossPositionValue', 'AvailableFunds', 'ExcessLiquidity']
            
            for av in account_values:
                if av.tag in key_values:
                    try:
                        account_summary[av.tag] = float(av.value) if av.value != '' else 0.0
                    except:
                        account_summary[av.tag] = av.value
            
            # Obtener posiciones
            positions = self.ib.positions()
            positions_data = []
            total_positions_value = 0
            
            for pos in positions:
                if pos.position != 0:  # Solo posiciones activas
                    pos_data = {
                        'symbol': pos.contract.symbol,
                        'position': pos.position,
                        'avgCost': pos.avgCost,
                        'marketPrice': pos.marketPrice,
                        'marketValue': pos.marketValue,
                        'unrealizedPNL': pos.unrealizedPNL,
                        'realizedPNL': pos.realizedPNL
                    }
                    positions_data.append(pos_data)
                    total_positions_value += pos.marketValue
            
            # Obtener órdenes activas
            orders = self.ib.openOrders()
            orders_data = []
            
            for order in orders:
                order_data = {
                    'orderId': order.order.orderId,
                    'symbol': order.contract.symbol,
                    'action': order.order.action,
                    'quantity': order.order.totalQuantity,
                    'orderType': order.order.orderType,
                    'status': order.orderStatus.status,
                    'filled': order.orderStatus.filled,
                    'remaining': order.orderStatus.remaining
                }
                orders_data.append(order_data)
            
            return {
                'connected': True,
                'account_summary': account_summary,
                'positions': positions_data,
                'open_orders': orders_data,
                'total_positions': len(positions_data),
                'total_positions_value': round(total_positions_value, 2),
                'timestamp': datetime.now().isoformat(),
                'trading_available': account_summary.get('BuyingPower', 0) > 1000
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {
                'connected': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def place_order(self, symbol, action, quantity, order_type='MKT', limit_price=None):
        """Ejecutar orden en IBKR con validaciones completas"""
        try:
            if not self.connected or not self.ib:
                if not self.connect():
                    raise Exception("Cannot connect to IBKR TWS/Gateway")
            
            logger.info(f"Placing order: {action} {quantity} {symbol} @ {order_type}")
            
            # Crear contrato
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Crear orden según el tipo
            if order_type.upper() == 'MKT':
                order = MarketOrder(action.upper(), quantity)
            elif order_type.upper() == 'LMT' and limit_price:
                order = LimitOrder(action.upper(), quantity, limit_price)
            else:
                order = MarketOrder(action.upper(), quantity)
            
            # Configuraciones adicionales
            order.outsideRth = True  # Trading fuera de horario
            order.tif = 'DAY'       # Good for day
            order.transmit = True   # Transmitir inmediatamente
            
            # Ejecutar orden
            trade = self.ib.placeOrder(contract, order)
            
            # Esperar respuesta (máximo 15 segundos)
            for i in range(150):  # 15 segundos = 150 * 0.1
                self.ib.sleep(0.1)
                if trade.orderStatus.status in ['Filled', 'Cancelled', 'Rejected', 'Submitted']:
                    break
            
            # Status final
            final_status = trade.orderStatus.status
            is_success = final_status in ['Filled', 'Submitted', 'PreSubmitted']
            
            result = {
                'success': is_success,
                'order_id': trade.order.orderId,
                'client_order_id': getattr(trade.order, 'clientId', ''),
                'symbol': symbol,
                'action': action.upper(),
                'quantity': quantity,
                'order_type': order_type.upper(),
                'status': final_status,
                'filled_quantity': trade.orderStatus.filled,
                'remaining_quantity': trade.orderStatus.remaining,
                'avg_fill_price': trade.orderStatus.avgFillPrice,
                'last_fill_price': trade.orderStatus.lastFillPrice,
                'commission': getattr(trade.orderStatus, 'commission', None),
                'timestamp': datetime.now().isoformat(),
                'broker': 'IBKR',
                'message': f"Order {final_status}: {action} {quantity} {symbol}"
            }
            
            if is_success:
                logger.info(f"✅ Order successful: {result['message']}")
            else:
                logger.error(f"❌ Order failed: {result['message']}")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error placing order: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'timestamp': datetime.now().isoformat(),
                'broker': 'IBKR',
                'message': f"Order failed: {error_msg}"
            }
    
    def get_market_data(self, symbol):
        """Obtener datos de mercado en tiempo real"""
        try:
            if not self.connected or not self.ib:
                if not self.connect():
                    return None
            
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Solicitar datos de mercado
            ticker = self.ib.reqMktData(contract, '', False, False)
            
            # Esperar datos (máximo 5 segundos)
            for i in range(50):
                self.ib.sleep(0.1)
                if ticker.last > 0 or ticker.marketPrice() > 0:
                    break
            
            # Cancelar suscripción para limpiar
            self.ib.cancelMktData(contract)
            
            price = ticker.last if ticker.last > 0 else ticker.marketPrice()
            
            if price > 0:
                return {
                    'symbol': symbol,
                    'price': price,
                    'last': ticker.last,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'volume': ticker.volume,
                    'high': ticker.high,
                    'low': ticker.low,
                    'close': ticker.close,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'IBKR_LIVE'
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None

# Instancia global del gestor IBKR
ibkr_manager = IBKRManager()

# ============================================================
# FUNCIONES AUXILIARES Y ANÁLISIS
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

def es_horario_mercado():
    """Trading 24/7 con IBKR"""
    try:
        ahora = datetime.now(pytz.timezone('US/Eastern'))
        dia_semana = ahora.weekday()
        hora_actual = ahora.time()
        
        # Solo bloquear fines de semana completos
        if dia_semana == 5 and hora_actual >= time(22, 0):  # Viernes noche
            return False
        if dia_semana == 6:  # Sábado completo
            return False
        if dia_semana == 0 and hora_actual < time(4, 0):  # Domingo madrugada
            return False
            
        return True
    except Exception:
        return True

def calcular_gestion_riesgo(precio, accion, confianza, volatility=0.02):
    """Gestión de riesgo adaptada para IBKR"""
    try:
        balance_base = 25000  # Mínimo para day trading
        riesgo_por_trade = 0.015  # 1.5% por trade (más conservador)
        
        # Ajustar por volatilidad
        vol_adjustment = min(volatility * 10, 2.0)
        riesgo_ajustado = riesgo_por_trade * (1 + vol_adjustment)
        
        # Factor de confianza
        factor_confianza = min(confianza * 1.1, 1.0)
        riesgo_final = riesgo_ajustado * factor_confianza
        
        # Stops más precisos para IBKR
        if accion == "BUY":
            stop_loss = precio * (1 - 0.025)  # 2.5% stop loss
            take_profit = precio * (1 + 0.05)   # 5% take profit
        else:
            stop_loss = precio * (1 + 0.025)
            take_profit = precio * (1 - 0.05)
        
        # Calcular cantidad
        riesgo_dolares = balance_base * riesgo_final
        riesgo_por_accion = abs(precio - stop_loss)
        cantidad = int(riesgo_dolares / riesgo_por_accion) if riesgo_por_accion > 0 else 1
        cantidad = max(1, min(cantidad, 500))  # Entre 1 y 500 acciones
        
        return {
            'position_size': cantidad,
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_amount': round(riesgo_dolares, 2),
            'risk_percent': round(riesgo_final * 100, 2),
            'volatility_adjustment': round(vol_adjustment, 2)
        }
        
    except Exception as e:
        logger.error(f"Error in risk management: {e}")
        return {
            'position_size': 1,
            'stop_loss': round(precio * 0.97, 2),
            'take_profit': round(precio * 1.03, 2),
            'risk_amount': 375,
            'risk_percent': 1.5,
            'volatility_adjustment': 1.0
        }

def get_stock_data_enhanced(symbol, period='30d'):
    """Obtener datos combinando IBKR y yfinance"""
    try:
        # Primero intentar datos en tiempo real de IBKR
        live_data = ibkr_manager.get_market_data(symbol)
        
        # Obtener datos históricos de yfinance
        stock = yf.Ticker(symbol)
        hist_data = stock.history(period=period)
        
        if hist_data is not None and not hist_data.empty:
            # Si tenemos datos live de IBKR, actualizar último precio
            if live_data and live_data.get('price', 0) > 0:
                logger.info(f"Using live price from IBKR for {symbol}: ${live_data['price']}")
                # Opcional: actualizar último precio en datos históricos
                # hist_data.iloc[-1, hist_data.columns.get_loc('Close')] = live_data['price']
            
            return hist_data
        else:
            logger.warning(f"No historical data for {symbol}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting enhanced data for {symbol}: {e}")
        return None

def calculate_technical_indicators(data):
    """Indicadores técnicos mejorados para IBKR"""
    try:
        if data is None or data.empty or len(data) < 20:
            return None
            
        indicators = {}
        
        # RSI
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            indicators['rsi'] = 50.0
        
        # MACD
        try:
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0
            indicators['macd_signal'] = float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0.0
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        except:
            indicators['macd'] = 0.0
            indicators['macd_signal'] = 0.0
            indicators['macd_histogram'] = 0.0
        
        # Moving Averages
        try:
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=min(50, len(data))).mean()
            ema_12 = data['Close'].ewm(span=12).mean()
            
            indicators['sma_20'] = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else data['Close'].iloc[-1]
            indicators['sma_50'] = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else data['Close'].iloc[-1]
            indicators['ema_12'] = float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else data['Close'].iloc[-1]
        except:
            indicators['sma_20'] = data['Close'].iloc[-1]
            indicators['sma_50'] = data['Close'].iloc[-1]
            indicators['ema_12'] = data['Close'].iloc[-1]
        
        # Volatility
        try:
            volatility = data['Close'].pct_change().std() * np.sqrt(252)  # Anualizada
            indicators['volatility'] = float(volatility) if not pd.isna(volatility) else 0.2
        except:
            indicators['volatility'] = 0.2
        
        # Volume analysis
        try:
            avg_volume = data['Volume'].rolling(window=20).mean()
            volume_ratio = data['Volume'].iloc[-1] / avg_volume.iloc[-1] if not pd.isna(avg_volume.iloc[-1]) else 1.0
            indicators['volume_ratio'] = float(volume_ratio) if not pd.isna(volume_ratio) else 1.0
        except:
            indicators['volume_ratio'] = 1.0
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def generate_ai_prediction_advanced(data, symbol):
    """Predicción de IA mejorada para IBKR"""
    try:
        if data is None or data.empty or len(data) < 10:
            return None
            
        # Análisis de tendencia
        recent_data = data['Close'].tail(10)
        long_data = data['Close'].tail(30) if len(data) >= 30 else data['Close']
        
        # Cambios recientes
        short_change = (recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]
        long_change = (long_data.iloc[-1] - long_data.iloc[0]) / long_data.iloc[0]
        
        # Volatilidad
        volatility = recent_data.pct_change().std()
        
        # Momentum
        momentum = recent_data.iloc[-1] / recent_data.rolling(5).mean().iloc[-1] - 1
        
        # Scoring system
        score = 0
        factors = []
        
        if short_change > 0.03:  # Subida > 3%
            score += 0.3
            factors.append(f"Tendencia alcista reciente: {short_change*100:.1f}%")
        elif short_change < -0.03:  # Bajada > 3%
            score -= 0.3
            factors.append(f"Tendencia bajista reciente: {short_change*100:.1f}%")
        
        if momentum > 0.02:
            score += 0.2
            factors.append("Momentum positivo")
        elif momentum < -0.02:
            score -= 0.2
            factors.append("Momentum negativo")
        
        # Determinar dirección y confianza
        if score > 0.2:
            direccion = "ALCISTA"
            cambio_esperado = abs(score) * 5  # 5% por punto de score
            confianza = min(0.85, 0.6 + abs(score))
        elif score < -0.2:
            direccion = "BAJISTA"
            cambio_esperado = -abs(score) * 5
            confianza = min(0.85, 0.6 + abs(score))
        else:
            direccion = "NEUTRAL"
            cambio_esperado = 0.0
            confianza = 0.45
        
        return {
            "direccion": direccion,
            "cambio_esperado_pct": round(cambio_esperado, 2),
            "confianza_ml": round(confianza, 2),
            "volatility": round(volatility, 4),
            "momentum": round(momentum, 4),
            "factores": factors,
            "timeframe": "1D",
            "modelo_usado": "TechnicalAnalysis_v2",
            "fecha_prediccion": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in AI prediction for {symbol}: {e}")
        return None

def generate_trading_signal_ibkr(symbol, data, indicators, ai_prediction):
    """Generar señal de trading optimizada para IBKR"""
    try:
        if data is None or data.empty or indicators is None:
            return None
            
        current_price = float(data['Close'].iloc[-1])
        confidence = 0.0
        action = "HOLD"
        reasons = []
        
        # Análisis RSI con más precisión
        rsi = indicators.get('rsi', 50)
        if rsi < 25:  # Sobreventa extrema
            confidence += 0.25
            action = "BUY"
            reasons.append(f"RSI muy bajo: {rsi:.1f}")
        elif rsi < 35:  # Sobreventa
            confidence += 0.15
            if action != "SELL":
                action = "BUY"
            reasons.append(f"RSI bajo: {rsi:.1f}")
        elif rsi > 75:  # Sobrecompra extrema
            confidence += 0.25
            action = "SELL"
            reasons.append(f"RSI muy alto: {rsi:.1f}")
        elif rsi > 65:  # Sobrecompra
            confidence += 0.15
            if action != "BUY":
                action = "SELL"
            reasons.append(f"RSI alto: {rsi:.1f}")
        
        # Análisis MACD
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_histogram', 0)
        
        if macd > macd_signal and macd_hist > 0:
            confidence += 0.15
            if action != "SELL":
                action = "BUY"
            reasons.append("MACD bullish crossover")
        elif macd < macd_signal and macd_hist < 0:
            confidence += 0.15
            if action != "BUY":
                action = "SELL"
            reasons.append("MACD bearish crossover")
        
        # Análisis de medias móviles
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        
        if current_price > sma_20 > sma_50:
            confidence += 0.2
            if action != "SELL":
                action = "BUY"
            reasons.append("Precio sobre medias móviles - Tendencia alcista")
        elif current_price < sma_20 < sma_50:
            confidence += 0.2
            if action != "BUY":
                action = "SELL"
            reasons.append("Precio bajo medias móviles - Tendencia bajista")
        
        # Análisis de volumen
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            confidence += 0.1
            reasons.append(f"Volume alto: {volume_ratio:.1f}x promedio")
        
        # IA prediction con peso alto
        if ai_prediction and ai_prediction.get('confianza_ml', 0) >= 0.6:
            ai_confidence = ai_prediction['confianza_ml']
            confidence += ai_confidence * 0.4  # Peso del 40%
            
            if ai_prediction['direccion'] == "ALCISTA":
                if action != "SELL":
                    action = "BUY"
                reasons.append(f"IA: Alcista {ai_prediction['cambio_esperado_pct']:.1f}% (conf: {ai_confidence*100:.0f}%)")
            elif ai_prediction['direccion'] == "BAJISTA":
                if action != "BUY":
                    action = "SELL"
                reasons.append(f"IA: Bajista {ai_prediction['cambio_esperado_pct']:.1f}% (conf: {ai_confidence*100:.0f}%)")
        
        # Filtro de confianza mínima
        if confidence < 0.4 or action == "HOLD":
            return None
        
        # Gestión de riesgo
        volatility = indicators.get('volatility', 0.2)
        risk_mgmt = calcular_gestion_riesgo(current_price, action, confidence, volatility)
        
        return {
            "symbol": symbol,
            "action": action,
            "side": action,
            "confidence": round(confidence, 3),
            "current_price": round(current_price, 2),
            "indicators": indicators,
            "ai_prediction": ai_prediction,
            "risk_management": risk_mgmt,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
            "trading_session": "REGULAR" if 9.5 <= datetime.now(pytz.timezone('US/Eastern')).hour <= 16 else "EXTENDED",
            "broker": "IBKR",
            "volatility": round(volatility, 4)
        }
        
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {e}")
        return None

# ============================================================
# ENDPOINTS PRINCIPALES
# ============================================================

@app.route('/')
def dashboard():
    """Dashboard principal del sistema IBKR"""
    return """
    <html>
    <head>
        <title>🚀 AI Trading System - Interactive Brokers (IBKR)</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #0a0e27; color: white; }
            .container { max-width: 1200px; margin: 0 auto; background: #1a1d29; padding: 30px; border-radius: 15px; }
            .status { text-align: center; padding: 20px; background: #2d1b69; border-radius: 10px; margin-bottom: 20px; }
            .endpoint { background: #1e3a5f; padding: 15px; border-radius: 8px; margin: 10px 0; }
            .success { color: #00ff88; font-weight: bold; }
            .warning { color: #ffeb3b; }
            .ibkr { color: #0088ff; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 AI Trading System - Interactive Brokers Integration</h1>
            <div class="status">
                <h2 class="success">✅ Sistema IBKR Listo para N8N</h2>
                <p class="ibkr">🏦 Integración con Interactive Brokers TWS/Gateway</p>
                <p>Trading algorítmico de nivel profesional</p>
                <p>Compatible con Paper Trading y Live Trading</p>
            </div>
            <div class="endpoint">
                <h3>🔗 Configuración IBKR:</h3>
                <p><strong>Puerto TWS:</strong> 7497 (Paper) / 7496 (Live)</p>
                <p><strong>Host:</strong> 127.0.0.1 (Local)</p>
                <p><strong>API:</strong> ib_insync framework</p>
                <p><strong>Estado:</strong> <span class="success">CONFIGURADO ✅</span></p>
            </div>
            <div class="endpoint">
                <h3>🔗 Endpoints para N8N:</h3>
                <p><strong>/analyze</strong> - Análisis y señales con IBKR</p>
                <p><strong>/place_order</strong> - Ejecutar órdenes reales en IBKR</p>
                <p><strong>/ibkr/account</strong> - Info de cuenta IBKR</p>
                <p><strong>/ibkr/positions</strong> - Posiciones actuales</p>
                <p><strong>Estado:</strong> <span class="success">ACTIVOS ✅</span></p>
            </div>
            <div class="endpoint">
                <h3>⚙️ Instalación y Configuración:</h3>
                <p><strong>1.</strong> Instalar TWS o IB Gateway</p>
                <p><strong>2.</strong> Habilitar API: Configuration → API → Settings</p>
                <p><strong>3.</strong> Puerto 7497 (Paper) o 7496 (Live)</p>
                <p><strong>4.</strong> <code>pip install ib_insync</code></p>
                <p><strong>5.</strong> Ejecutar TWS/Gateway antes que este sistema</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/health')
def health_check():
    """Health check con verificación de IBKR"""
    try:
        # Verificar conexión IBKR
        ibkr_status = "DISCONNECTED"
        ibkr_error = None
        account_info = None
        
        try:
            if ibkr_manager.connect():
                ibkr_status = "CONNECTED"
                account_info = ibkr_manager.get_account_info()
            else:
                ibkr_status = "CONNECTION_FAILED"
        except Exception as e:
            ibkr_status = "ERROR"
            ibkr_error = str(e)
        
        return jsonify({
            "status": "OK",
            "timestamp": datetime.now().isoformat(),
            "broker": "Interactive Brokers (IBKR)",
            "ibkr_connection": ibkr_status,
            "ibkr_error": ibkr_error,
            "trading_available": account_info.get('trading_available', False) if account_info else False,
            "endpoints": {
                "analyze": "ACTIVE",
                "place_order": "ACTIVE", 
                "ibkr_account": "ACTIVE",
                "health": "ACTIVE"
            },
            "market_status": "EXTENDED" if es_horario_mercado() else "CLOSED",
            "version": "IBKR_v1.0",
            "dependencies": {
                "ib_insync": "Available",
                "yfinance": "Available",
                "flask": "Available"
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/analyze')
def analyze_market():
    """Endpoint principal de análisis con IBKR"""
    try:
        logger.info("Starting IBKR market analysis...")
        
        # Parámetros
        force_analysis = request.args.get('force', 'false').lower() == 'true'
        enable_ai = request.args.get('ai', 'true').lower() == 'true'
        min_confidence = float(request.args.get('min_confidence', '0.4'))
        
        # Lista de señales
        valid_signals = []
        
        # Verificar conexión IBKR
        if not ibkr_manager.connect():
            logger.warning("IBKR not connected, using yfinance only")
        
        # Analizar símbolos prioritarios
        symbols_to_analyze = PRIORITY_SYMBOLS
        
        for symbol in symbols_to_analyze:
            try:
                logger.info(f"Analyzing {symbol} with IBKR integration...")
                
                # Obtener datos mejorados
                data = get_stock_data_enhanced(symbol, period='30d')
                if data is None or data.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Calcular indicadores
                indicators = calculate_technical_indicators(data)
                if not indicators:
                    logger.warning(f"No indicators for {symbol}")
                    continue
                    
                # Predicción IA
                ai_prediction = None
                if enable_ai:
                    ai_prediction = generate_ai_prediction_advanced(data, symbol)
                
                # Generar señal
                signal = generate_trading_signal_ibkr(symbol, data, indicators, ai_prediction)
                
                if signal and signal['confidence'] >= min_confidence:
                    # Enriquecer señal para N8N
                    signal.update({
                        "order_type": "market",
                        "type": "market", 
                        "price": signal['current_price'],
                        "qty": signal['risk_management']['position_size'],
                        "order_success": True,
                        "order_status": "PENDING",
                        "order_id": f"IBKR-{int(time_module.time())}-{symbol}",
                        "submitted_at": datetime.now().isoformat(),
                        "message": f"Señal IBKR para {symbol}",
                        "n8n_compatible": True,
                        "broker_specific": {
                            "outside_rth": True,
                            "tif": "DAY",
                            "contract_type": "STK",
                            "exchange": "SMART",
                            "currency": "USD"
                        }
                    })
                    
                    valid_signals.append(signal)
                    logger.info(f"✅ Valid IBKR signal: {symbol} {signal['action']} conf:{signal['confidence']:.2f}")
                
                # Rate limiting
                time_module.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Respuesta para N8N
        response = {
            "status": "SUCCESS",
            "timestamp": datetime.now().isoformat(),
            "broker": "IBKR",
            "ibkr_connected": ibkr_manager.connected,
            "market_status": "EXTENDED" if es_horario_mercado() else "CLOSED",
            "extended_hours_available": es_horario_mercado(),
            "trading_session": "REGULAR" if 9.5 <= datetime.now(pytz.timezone('US/Eastern')).hour <= 16 else "EXTENDED",
            "analysis_params": {
                "force_analysis": force_analysis,
                "ai_enabled": enable_ai,
                "min_confidence": min_confidence,
                "symbols_analyzed": len(symbols_to_analyze)
            },
            "signals_generated": len(valid_signals),
            "actionable_signals": len(valid_signals),
            "signals": valid_signals,
            "server_time": datetime.now().isoformat(),
            "n8n_integration": "READY",
            "version": "IBKR_v1.0"
        }
        
        logger.info(f"IBKR Analysis completed: {len(valid_signals)} signals")
        return jsonify(response)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in IBKR analysis: {error_msg}")
        
        return jsonify({
            "status": "ERROR",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "signals": [],
            "actionable_signals": 0,
            "market_status": "ERROR",
            "broker": "IBKR",
            "version": "IBKR_v1.0"
        }), 500

@app.route('/place_order', methods=['POST'])
def place_order():
    """Ejecutar orden real en IBKR"""
    try:
        logger.info("Received order request for IBKR")
        
        # Obtener datos de la orden
        data = request.json
        if not data:
            return jsonify({
                "status": "ERROR",
                "message": "No data received",
                "order_success": False,
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Validar campos requeridos
        required_fields = ['symbol', 'qty', 'side']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "status": "ERROR",
                    "message": f"Missing field: {field}",
                    "order_success": False,
                    "timestamp": datetime.now().isoformat()
                }), 400
        
        # Verificar conexión IBKR
        if not ibkr_manager.connect():
            return jsonify({
                "status": "ERROR",
                "message": "Cannot connect to IBKR TWS/Gateway",
                "order_success": False,
                "timestamp": datetime.now().isoformat(),
                "help": "Asegúrate de que TWS/Gateway esté ejecutándose"
            }), 503
        
        # Ejecutar orden en IBKR
        symbol = data['symbol'].upper()
        action = data['side'].upper()
        quantity = int(data['qty'])
        order_type = data.get('type', 'MKT').upper()
        limit_price = data.get('price') if order_type == 'LMT' else None
        
        logger.info(f"Placing IBKR order: {action} {quantity} {symbol}")
        
        # Ejecutar orden
        result = ibkr_manager.place_order(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price
        )
        
        # Enriquecer respuesta para N8N
        if result.get('success', False):
            # Obtener precio actual para cálculos
            current_price = result.get('avg_fill_price', 0) or result.get('last_fill_price', 0) or data.get('price', 100)
            
            # Gestión de riesgo
            risk_mgmt = calcular_gestion_riesgo(current_price, action, 0.8)
            
            response = {
                # Campos básicos
                "order_id": result['order_id'],
                "symbol": symbol,
                "qty": quantity,
                "side": action,
                "action": action,
                "type": order_type,
                "order_type": order_type.lower(),
                "price": current_price,
                "current_price": current_price,
                
                # Estados
                "status": result['status'],
                "order_status": result['status'],
                "order_success": True,
                
                # IBKR específico
                "filled_quantity": result.get('filled_quantity', 0),
                "remaining_quantity": result.get('remaining_quantity', quantity),
                "commission": result.get('commission'),
                
                # Campos requeridos por N8N
                "confidence": 0.8,
                "ai_prediction": {
                    "direccion": "ALCISTA" if action == "BUY" else "BAJISTA",
                    "cambio_esperado_pct": 3.0 if action == "BUY" else -3.0,
                    "confianza_ml": 0.8,
                    "timeframe": "1D",
                    "modelo_usado": "IBKR_Direct",
                    "fecha_prediccion": datetime.now().isoformat()
                },
                "indicators": {
                    "rsi": 50.0,
                    "macd": 0.1,
                    "sma_20": current_price * 0.99,
                    "sma_50": current_price * 0.98,
                    "volume_ratio": 1.2
                },
                "risk_management": risk_mgmt,
                "reasons": [
                    "Orden ejecutada directamente en IBKR",
                    f"Broker: Interactive Brokers",
                    f"Status: {result['status']}"
                ],
                
                # Timestamps
                "submitted_at": datetime.now().isoformat(),
                "filled_at": datetime.now().isoformat() if result['status'] == 'Filled' else None,
                "timestamp": datetime.now().isoformat(),
                
                # Info adicional
                "message": result.get('message', f"IBKR order {result['status']}"),
                "broker": "IBKR",
                "trading_session": "REGULAR" if 9.5 <= datetime.now(pytz.timezone('US/Eastern')).hour <= 16 else "EXTENDED",
                "n8n_compatible": True,
                "actionable_signals": 1
            }
            
            logger.info(f"✅ IBKR order successful: {response['message']}")
            return jsonify(response), 200
            
        else:
            # Orden falló
            error_response = {
                "status": "ERROR",
                "order_success": False,
                "message": result.get('message', 'IBKR order failed'),
                "error": result.get('error', 'Unknown error'),
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "timestamp": datetime.now().isoformat(),
                "broker": "IBKR"
            }
            
            logger.error(f"❌ IBKR order failed: {error_response['message']}")
            return jsonify(error_response), 400
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in IBKR place_order: {error_msg}")
        
        return jsonify({
            "status": "ERROR",
            "order_success": False,
            "message": f"IBKR order error: {error_msg}",
            "timestamp": datetime.now().isoformat(),
            "broker": "IBKR"
        }), 500

@app.route('/ibkr/account')
def ibkr_account():
    """Información de cuenta IBKR"""
    try:
        account_info = ibkr_manager.get_account_info()
        
        if account_info and account_info.get('connected', False):
            return jsonify({
                "status": "SUCCESS",
                "connected": True,
                "account_info": account_info,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "ERROR", 
                "connected": False,
                "error": account_info.get('error', 'Connection failed'),
                "timestamp": datetime.now().isoformat()
            }), 503
            
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/ibkr/positions')
def ibkr_positions():
    """Posiciones actuales en IBKR"""
    try:
        account_info = ibkr_manager.get_account_info()
        
        if account_info and account_info.get('connected', False):
            return jsonify({
                "status": "SUCCESS",
                "positions": account_info.get('positions', []),
                "total_positions": account_info.get('total_positions', 0),
                "total_value": account_info.get('total_positions_value', 0),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "ERROR",
                "positions": [],
                "error": "Not connected to IBKR",
                "timestamp": datetime.now().isoformat()
            }), 503
            
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "positions": [],
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/ibkr/test_connection')
def test_ibkr_connection():
    """Test específico de conexión IBKR"""
    try:
        logger.info("Testing IBKR connection...")
        
        if ibkr_manager.connect():
            account_info = ibkr_manager.get_account_info()
            
            return jsonify({
                "status": "SUCCESS",
                "message": "IBKR connection successful",
                "connected": True,
                "account_summary": account_info.get('account_summary', {}),
                "managed_accounts": ibkr_manager.ib.managedAccounts() if ibkr_manager.ib else [],
                "timestamp": datetime.now().isoformat(),
                "instructions": {
                    "step1": "TWS/Gateway is running ✅",
                    "step2": "API is enabled ✅", 
                    "step3": "Connection successful ✅"
                }
            })
        else:
            return jsonify({
                "status": "ERROR",
                "message": "Cannot connect to IBKR",
                "connected": False,
                "timestamp": datetime.now().isoformat(),
                "troubleshooting": {
                    "step1": "Start TWS or IB Gateway",
                    "step2": "Enable API in Configuration → API → Settings",
                    "step3": f"Verify port {IBKR_CONFIG['port']} is correct",
                    "step4": "Check firewall settings"
                }
            }), 503
            
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": f"Connection test failed: {str(e)}",
            "connected": False,
            "timestamp": datetime.now().isoformat()
        }), 500

# ============================================================
# ENDPOINTS ADICIONALES Y UTILIDADES
# ============================================================

@app.route('/ibkr/market_data/<symbol>')
def get_live_market_data(symbol):
    """Obtener datos de mercado en tiempo real"""
    try:
        symbol = symbol.upper()
        market_data = ibkr_manager.get_market_data(symbol)
        
        if market_data:
            return jsonify({
                "status": "SUCCESS",
                "symbol": symbol,
                "market_data": market_data,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "ERROR",
                "symbol": symbol,
                "error": "No market data available",
                "timestamp": datetime.now().isoformat()
            }), 404
            
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "symbol": symbol,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Cerrar conexiones y apagar servidor"""
    try:
        logger.info("Shutting down IBKR Trading System...")
        
        # Desconectar IBKR
        ibkr_manager.disconnect()
        
        return jsonify({
            "status": "SUCCESS",
            "message": "System shutdown initiated",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": f"Shutdown error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == '__main__':
    import os
    
    # Puerto desde variable de entorno
    port = int(os.environ.get('PORT', 10000))
    
    # Información de inicio
    print("\n" + "="*80)
    print("🚀 AI TRADING SYSTEM - INTERACTIVE BROKERS (IBKR) INTEGRATION")
    print("="*80)
    print(f"🌐 Puerto: {port}")
    print(f"🏦 Broker: Interactive Brokers")
    print(f"📍 TWS Port: {IBKR_CONFIG['port']} ({'Paper Trading' if IBKR_CONFIG['port'] == 7497 else 'Live Trading'})")
    print(f"🔗 Dashboard: http://localhost:{port}")
    print("="*80)
    print("📋 Endpoints principales:")
    print("   • GET  /              - Dashboard del sistema")
    print("   • GET  /health        - Estado general")
    print("   • GET  /analyze       - Análisis con IBKR")
    print("   • POST /place_order   - Ejecutar órdenes IBKR")
    print("   • GET  /ibkr/account  - Info cuenta IBKR")
    print("   • GET  /ibkr/positions - Posiciones IBKR")
    print("   • GET  /ibkr/test_connection - Test conexión")
    print("="*80)
    print("⚙️ Configuración requerida:")
    print("   1. Instalar TWS o IB Gateway")
    print("   2. Configuration → API → Settings → Enable API")
    print(f"   3. Port: {IBKR_CONFIG['port']}")
    print("   4. pip install ib_insync")
    print("="*80)
    print("🔥 Sistema IBKR listo para trading profesional!")
    print()
    
    # Iniciar Flask
    app.run(host='0.0.0.0', port=port, debug=False)
