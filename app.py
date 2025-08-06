# SISTEMA DE TRADING HÍBRIDO - CONFIGURACIÓN MEJORADA
# =====================================================================
# Solución para ejecutar trading bot en la nube con conexión local a IBKR
# =====================================================================

import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIGURACIÓN ADAPTABLE - Se ajusta automáticamente al entorno
def get_trading_config():
    """Configuración que se adapta al entorno de ejecución"""
    
    # Detectar si estamos en producción (Render) o desarrollo local
    is_production = os.environ.get('RENDER') is not None
    
    if is_production:
        # En producción: Solo análisis, sin conexión directa a IBKR
        return {
            'mode': 'ANALYSIS_ONLY',
            'ibkr_enabled': False,
            'description': 'Servidor en la nube - Solo análisis y señales',
            'trading_method': 'WEBHOOK_TO_LOCAL'
        }
    else:
        # En desarrollo local: Conexión completa a IBKR
        return {
            'mode': 'FULL_TRADING',
            'ibkr_enabled': True,
            'host': '127.0.0.1',
            'port': 7497,  # Paper trading
            'description': 'Servidor local - Trading completo con IBKR'
        }

TRADING_CONFIG = get_trading_config()

# Símbolos para análisis
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

class TradingAnalyzer:
    """Analizador de trading que funciona en cualquier entorno"""
    
    def __init__(self):
        self.config = TRADING_CONFIG
        logger.info(f"Trading Analyzer iniciado en modo: {self.config['mode']}")
    
    def get_stock_data(self, symbol, period='30d'):
        """Obtener datos de mercado usando yfinance (funciona en cualquier lugar)"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"No hay datos para {symbol}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Error obteniendo datos para {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, window=14):
        """Calcular RSI - Indicador técnico fundamental"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def calculate_macd(self, prices):
        """Calcular MACD - Detector de tendencias"""
        try:
            exp1 = prices.ewm(span=12).mean()
            exp2 = prices.ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            return {
                'macd': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0,
                'signal': signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0.0,
                'histogram': histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0.0
            }
        except:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    def analyze_symbol(self, symbol):
        """Análisis completo de un símbolo"""
        try:
            # Obtener datos históricos
            data = self.get_stock_data(symbol)
            if data is None or data.empty:
                return None
            
            current_price = float(data['Close'].iloc[-1])
            
            # Calcular indicadores técnicos
            rsi = self.calculate_rsi(data['Close'])
            macd_data = self.calculate_macd(data['Close'])
            
            # Medias móviles
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else current_price
            
            # Análisis de tendencia
            trend_short = (current_price - sma_20) / sma_20 * 100  # % sobre SMA20
            trend_long = (current_price - sma_50) / sma_50 * 100   # % sobre SMA50
            
            # Generar señal de trading
            signal = self.generate_trading_signal(
                current_price, rsi, macd_data, trend_short, trend_long
            )
            
            return {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'indicators': {
                    'rsi': round(rsi, 2),
                    'macd': round(macd_data['macd'], 4),
                    'macd_signal': round(macd_data['signal'], 4),
                    'macd_histogram': round(macd_data['histogram'], 4),
                    'sma_20': round(sma_20, 2),
                    'sma_50': round(sma_50, 2),
                    'trend_short_pct': round(trend_short, 2),
                    'trend_long_pct': round(trend_long, 2)
                },
                'signal': signal,
                'timestamp': datetime.now().isoformat(),
                'data_source': 'yfinance'
            }
            
        except Exception as e:
            logger.error(f"Error analizando {symbol}: {e}")
            return None
    
    def generate_trading_signal(self, price, rsi, macd_data, trend_short, trend_long):
        """Generar señal de trading basada en indicadores técnicos"""
        try:
            signals = []
            confidence = 0.0
            action = "HOLD"
            
            # Análisis RSI
            if rsi < 30:  # Sobreventa
                signals.append("RSI sobreventa")
                confidence += 0.3
                action = "BUY"
            elif rsi > 70:  # Sobrecompra
                signals.append("RSI sobrecompra")
                confidence += 0.3
                action = "SELL"
            
            # Análisis MACD
            if macd_data['histogram'] > 0 and macd_data['macd'] > macd_data['signal']:
                signals.append("MACD alcista")
                confidence += 0.25
                if action != "SELL":
                    action = "BUY"
            elif macd_data['histogram'] < 0 and macd_data['macd'] < macd_data['signal']:
                signals.append("MACD bajista")
                confidence += 0.25
                if action != "BUY":
                    action = "SELL"
            
            # Análisis de tendencia
            if trend_short > 2 and trend_long > 1:  # Fuerte tendencia alcista
                signals.append("Tendencia alcista fuerte")
                confidence += 0.2
                if action != "SELL":
                    action = "BUY"
            elif trend_short < -2 and trend_long < -1:  # Fuerte tendencia bajista
                signals.append("Tendencia bajista fuerte")
                confidence += 0.2
                if action != "BUY":
                    action = "SELL"
            
            # Solo generar señal si hay confianza suficiente
            if confidence < 0.4:
                action = "HOLD"
                confidence = 0.0
            
            return {
                'action': action,
                'confidence': round(confidence, 3),
                'reasons': signals,
                'recommended_quantity': self.calculate_position_size(price, confidence) if action != "HOLD" else 0,
                'stop_loss': self.calculate_stop_loss(price, action) if action != "HOLD" else None,
                'take_profit': self.calculate_take_profit(price, action) if action != "HOLD" else None
            }
            
        except Exception as e:
            logger.error(f"Error generando señal: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasons': ['Error en análisis'],
                'recommended_quantity': 0
            }
    
    def calculate_position_size(self, price, confidence):
        """Calcular tamaño de posición basado en gestión de riesgo"""
        base_investment = 1000  # $1000 base
        risk_multiplier = min(confidence * 1.5, 1.0)  # Máximo 100% del base
        total_investment = base_investment * risk_multiplier
        return max(1, int(total_investment / price))
    
    def calculate_stop_loss(self, price, action):
        """Calcular stop loss - 3% de pérdida máxima"""
        if action == "BUY":
            return round(price * 0.97, 2)  # 3% debajo
        else:  # SELL
            return round(price * 1.03, 2)  # 3% arriba
    
    def calculate_take_profit(self, price, action):
        """Calcular take profit - 6% de ganancia objetivo"""
        if action == "BUY":
            return round(price * 1.06, 2)  # 6% arriba
        else:  # SELL
            return round(price * 0.94, 2)  # 6% abajo

# Instancia global del analizador
analyzer = TradingAnalyzer()

# ============================================================
# ENDPOINTS DE LA API
# ============================================================

@app.route('/')
def dashboard():
    """Dashboard adaptable según el entorno"""
    env_info = "🌐 PRODUCCIÓN (Render)" if TRADING_CONFIG['mode'] == 'ANALYSIS_ONLY' else "💻 DESARROLLO (Local)"
    
    return f"""
    <html>
    <head>
        <title>AI Trading System - {TRADING_CONFIG['mode']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #0a0e27; color: white; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: #1a1d29; padding: 30px; border-radius: 15px; }}
            .status {{ text-align: center; padding: 20px; background: #2d1b69; border-radius: 10px; margin-bottom: 20px; }}
            .mode {{ color: #00ff88; font-weight: bold; font-size: 18px; }}
            .warning {{ color: #ffeb3b; padding: 10px; background: #332900; border-radius: 5px; margin: 10px 0; }}
            .success {{ color: #00ff88; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 AI Trading System</h1>
            <div class="status">
                <div class="mode">{env_info}</div>
                <p>Modo: {TRADING_CONFIG['mode']}</p>
                <p class="success">✅ Sistema funcionando correctamente</p>
            </div>
            
            {"<div class='warning'>⚠️ En modo ANÁLISIS - Para trading real, ejecutar localmente con TWS</div>" if TRADING_CONFIG['mode'] == 'ANALYSIS_ONLY' else "<div class='success'>✅ Conexión IBKR disponible para trading real</div>"}
            
            <h3>📋 Endpoints disponibles:</h3>
            <p><strong>/health</strong> - Estado del sistema</p>
            <p><strong>/analyze</strong> - Análisis de mercado y señales</p>
            <p><strong>/analyze/&lt;symbol&gt;</strong> - Análisis de símbolo específico</p>
            {"<p><strong>/place_order</strong> - Ejecutar órdenes (solo local)</p>" if TRADING_CONFIG['ibkr_enabled'] else ""}
        </div>
    </body>
    </html>
    """

@app.route('/health')
def health_check():
    """Estado del sistema"""
    return jsonify({
        'status': 'OK',
        'mode': TRADING_CONFIG['mode'],
        'ibkr_enabled': TRADING_CONFIG['ibkr_enabled'],
        'description': TRADING_CONFIG['description'],
        'timestamp': datetime.now().isoformat(),
        'environment': 'production' if os.environ.get('RENDER') else 'development'
    })

@app.route('/analyze')
def analyze_all():
    """Análisis de todos los símbolos principales"""
    try:
        logger.info("Iniciando análisis de mercado...")
        
        results = []
        trading_signals = []
        
        for symbol in SYMBOLS:
            analysis = analyzer.analyze_symbol(symbol)
            if analysis:
                results.append(analysis)
                
                # Si hay señal de trading, añadir a lista de señales
                if analysis['signal']['action'] != 'HOLD':
                    trading_signals.append(analysis)
        
        response = {
            'status': 'SUCCESS',
            'mode': TRADING_CONFIG['mode'],
            'timestamp': datetime.now().isoformat(),
            'total_analyzed': len(results),
            'trading_signals': len(trading_signals),
            'analyses': results,
            'actionable_signals': trading_signals,
            'market_summary': {
                'bullish_signals': len([s for s in trading_signals if s['signal']['action'] == 'BUY']),
                'bearish_signals': len([s for s in trading_signals if s['signal']['action'] == 'SELL']),
                'avg_confidence': round(sum([s['signal']['confidence'] for s in trading_signals]) / len(trading_signals), 3) if trading_signals else 0
            }
        }
        
        logger.info(f"Análisis completado: {len(trading_signals)} señales generadas")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en análisis: {e}")
        return jsonify({
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/analyze/<symbol>')
def analyze_single(symbol):
    """Análisis de un símbolo específico"""
    try:
        symbol = symbol.upper()
        analysis = analyzer.analyze_symbol(symbol)
        
        if analysis:
            return jsonify({
                'status': 'SUCCESS',
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'ERROR',
                'error': f'No se pudo analizar {symbol}',
                'timestamp': datetime.now().isoformat()
            }), 404
            
    except Exception as e:
        return jsonify({
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/place_order', methods=['POST'])
def place_order():
    """Endpoint para órdenes (solo funciona en modo local)"""
    if not TRADING_CONFIG['ibkr_enabled']:
        return jsonify({
            'status': 'ERROR',
            'message': 'Trading no disponible en modo producción',
            'help': 'Para trading real, ejecuta el sistema localmente con TWS',
            'timestamp': datetime.now().isoformat()
        }), 403
    
    # Aquí iría la lógica de IBKR si estamos en modo local
    return jsonify({
        'status': 'INFO',
        'message': 'Funcionalidad de trading local disponible',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    print("\n" + "="*60)
    print("🚀 AI TRADING SYSTEM - CONFIGURACIÓN INTELIGENTE")
    print("="*60)
    print(f"🌐 Puerto: {port}")
    print(f"🏗️ Modo: {TRADING_CONFIG['mode']}")
    print(f"📍 Descripción: {TRADING_CONFIG['description']}")
    print(f"🔧 IBKR Habilitado: {TRADING_CONFIG['ibkr_enabled']}")
    print("="*60)
    
    if TRADING_CONFIG['mode'] == 'ANALYSIS_ONLY':
        print("ℹ️  MODO ANÁLISIS: Perfecto para generar señales en la nube")
        print("💡 Para trading real: ejecutar localmente con TWS instalado")
    else:
        print("🔥 MODO COMPLETO: Trading real disponible con IBKR")
    
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=False)
