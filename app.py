from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time
import pytz

app = Flask(__name__)

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

def es_horario_mercado():
    """Verifica si el mercado esta abierto"""
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

def calcular_indicadores_avanzados(data):
    """Calcula multiples indicadores tecnicos"""
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
    
    # Bollinger Bands
    bb_middle = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    
    # Volumen promedio
    volume_avg = data['Volume'].rolling(window=20).mean()
    
    return {
        'rsi': rsi.iloc[-1],
        'macd': macd.iloc[-1],
        'signal_line': signal_line.iloc[-1],
        'sma_20': sma_20.iloc[-1],
        'sma_50': sma_50.iloc[-1],
        'bb_upper': bb_upper.iloc[-1],
        'bb_lower': bb_lower.iloc[-1],
        'bb_middle': bb_middle.iloc[-1],
        'volume_ratio': data['Volume'].iloc[-1] / volume_avg.iloc[-1] if volume_avg.iloc[-1] > 0 else 1,
        'price': data['Close'].iloc[-1]
    }

def evaluar_senal_avanzada(indicators):
    """Sistema de puntuacion avanzado para senales"""
    puntos_compra = 0
    puntos_venta = 0
    razones = []
    
    # RSI Analysis
    if indicators['rsi'] < 30:
        puntos_compra += 2
        razones.append(f"RSI sobreventa ({indicators['rsi']:.1f})")
    elif indicators['rsi'] > 70:
        puntos_venta += 2
        razones.append(f"RSI sobrecompra ({indicators['rsi']:.1f})")
    elif indicators['rsi'] < 40:
        puntos_compra += 1
        razones.append(f"RSI favorable compra ({indicators['rsi']:.1f})")
    elif indicators['rsi'] > 60:
        puntos_venta += 1
        razones.append(f"RSI favorable venta ({indicators['rsi']:.1f})")
    
    # MACD Analysis
    if indicators['macd'] > indicators['signal_line']:
        if indicators['macd'] > 0:
            puntos_compra += 2
            razones.append("MACD fuertemente alcista")
        else:
            puntos_compra += 1
            razones.append("MACD cruzando alcista")
    else:
        if indicators['macd'] < 0:
            puntos_venta += 2
            razones.append("MACD fuertemente bajista")
        else:
            puntos_venta += 1
            razones.append("MACD cruzando bajista")
    
    # SMA Trend Analysis
    precio = indicators['price']
    sma_20 = indicators['sma_20']
    sma_50 = indicators['sma_50']
    
    if precio > sma_20 > sma_50:
        puntos_compra += 2
        razones.append("Tendencia alcista fuerte (SMA)")
    elif precio > sma_20:
        puntos_compra += 1
        razones.append("Tendencia alcista corto plazo")
    elif precio < sma_20 < sma_50:
        puntos_venta += 2
        razones.append("Tendencia bajista fuerte (SMA)")
    elif precio < sma_20:
        puntos_venta += 1
        razones.append("Tendencia bajista corto plazo")
    
    # Bollinger Bands
    if precio <= indicators['bb_lower']:
        puntos_compra += 1
        razones.append("Precio en banda inferior (oportunidad)")
    elif precio >= indicators['bb_upper']:
        puntos_venta += 1
        razones.append("Precio en banda superior (cuidado)")
    
    # Volume Analysis
    if indicators['volume_ratio'] > 1.5:
        if puntos_compra > puntos_venta:
            puntos_compra += 1
            razones.append("Alto volumen confirma compra")
        else:
            puntos_venta += 1
            razones.append("Alto volumen confirma venta")
    
    # Decision Final (requiere al menos 3 puntos)
    if puntos_compra >= 3 and puntos_compra > puntos_venta:
        confianza = min(0.95, puntos_compra * 0.15)
        return "BUY", confianza, razones
    elif puntos_venta >= 3 and puntos_venta > puntos_compra:
        confianza = min(0.95, puntos_venta * 0.15)
        return "SELL", confianza, razones
    
    return None, 0, razones

def calcular_gestion_riesgo(precio, accion, confianza):
    """Gestion avanzada de riesgo"""
    balance_simulado = 10000  # $10,000 para paper trading
    riesgo_por_operacion = 0.02  # 2% maximo por operacion
    
    # Tamano de posicion basado en confianza
    factor_confianza = min(confianza * 1.5, 1.0)
    riesgo_ajustado = riesgo_por_operacion * factor_confianza
    
    # Stop loss y take profit dinamicos
    if accion == "BUY":
        stop_loss = precio * 0.95  # 5% stop loss
        take_profit = precio * 1.10  # 10% take profit
    else:
        stop_loss = precio * 1.05
        take_profit = precio * 0.90
    
    # Calcular cantidad de acciones
    riesgo_dolares = balance_simulado * riesgo_ajustado
    cantidad = int(riesgo_dolares / (precio * 0.05))  # Basado en 5% de riesgo por accion
    cantidad = max(1, min(cantidad, 50))  # Entre 1 y 50 acciones
    
    return {
        'position_size': cantidad,
        'stop_loss': round(stop_loss, 2),
        'take_profit': round(take_profit, 2),
        'risk_amount': round(riesgo_dolares, 2),
        'risk_percent': round(riesgo_ajustado * 100, 1)
    }

@app.route('/analyze')
def analyze_market():
    """ENDPOINT PRINCIPAL - Analisis completo"""
    
    # Verificar horario (para testing, siempre retorna True)
    # Cambiar a es_horario_mercado() para produccion
    market_open = True  # es_horario_mercado()
    
    if not market_open:
        return jsonify({
            "signals": [],
            "actionable_signals": 0,
            "total_analyzed": 0,
            "message": "Mercado cerrado - analisis pausado",
            "market_status": "CLOSED"
        })
    
    signals = []
    total_analyzed = 0
    
    for symbol in SYMBOLS:
        try:
            # Obtener datos historicos
            stock = yf.Ticker(symbol)
            data = stock.history(period='60d', interval='1d')
            
            if len(data) < 50:
                continue
                
            total_analyzed += 1
            
            # Calcular indicadores
            indicators = calcular_indicadores_avanzados(data)
            if not indicators:
                continue
            
            # Evaluar senal
            action, confidence, reasons = evaluar_senal_avanzada(indicators)
            
            if action and confidence > 0.4:  # Solo senales con >40% confianza
                # Gestion de riesgo
                risk_mgmt = calcular_gestion_riesgo(indicators['price'], action, confidence)
                
                signals.append({
                    "symbol": symbol,
                    "action": action,
                    "current_price": round(indicators['price'], 2),
                    "confidence": round(confidence, 3),
                    "indicators": {
                        "rsi": round(indicators['rsi'], 2),
                        "macd": round(indicators['macd'], 4),
                        "signal_line": round(indicators['signal_line'], 4),
                        "sma_20": round(indicators['sma_20'], 2),
                        "sma_50": round(indicators['sma_50'], 2),
                        "volume_ratio": round(indicators['volume_ratio'], 2)
                    },
                    "reasons": reasons,
                    "risk_management": risk_mgmt,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            print(f"Error analizando {symbol}: {e}")
            continue
    
    # Ordenar por confianza (mejores primero)
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    return jsonify({
        "signals": signals,
        "actionable_signals": len(signals),
        "total_analyzed": total_analyzed,
        "market_status": "OPEN",
        "analysis_time": datetime.now().isoformat()
    })

@app.route('/portfolio/correlation')
def portfolio_correlation():
    """Analisis de correlaciones del portafolio"""
    try:
        # Obtener datos de todas las acciones
        data = {}
        for symbol in SYMBOLS[:10]:  # Primeras 10 para no sobrecargar
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='30d')
                if len(hist) > 20:
                    data[symbol] = hist['Close'].pct_change().dropna()
            except:
                continue
        
        # Calcular correlaciones
        correlations = []
        symbols_list = list(data.keys())
        
        for i in range(len(symbols_list)):
            for j in range(i+1, len(symbols_list)):
                symbol1, symbol2 = symbols_list[i], symbols_list[j]
                if len(data[symbol1]) > 10 and len(data[symbol2]) > 10:
                    # Alinear fechas
                    common_dates = data[symbol1].index.intersection(data[symbol2].index)
                    if len(common_dates) > 10:
                        corr = data[symbol1].loc[common_dates].corr(data[symbol2].loc[common_dates])
                        if abs(corr) > 0.7:  # Solo correlaciones altas
                            correlations.append({
                                "pair": [symbol1, symbol2],
                                "correlation": round(corr, 3)
                            })
        
        recommendation = "Portafolio bien diversificado"
        if len(correlations) > 3:
            recommendation = "Cuidado: Muchas correlaciones altas detectadas"
        
        return jsonify({
            "high_correlations": correlations,
            "recommendation": recommendation,
            "analysis_date": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "high_correlations": [],
            "recommendation": "Error en analisis de correlacion",
            "error": str(e)
        })

@app.route('/health')
def health_check():
    """Health check para UptimeRobot"""
    return jsonify({
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "market_open": True,  # es_horario_mercado() para produccion
        "symbols_count": len(SYMBOLS),
        "version": "2.0"
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
