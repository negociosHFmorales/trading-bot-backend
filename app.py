from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time
import pytz

app = Flask(_name_)

# 🎯 CONFIGURACIÓN DE ACCIONES (20 acciones diversificadas)
SYMBOLS = [
    # Tecnología
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'ADBE',
    # Finanzas
    'JPM', 'V', 'MA', 'PYPL',
    # Salud
    'JNJ', 'PFE', 'UNH', 'MRNA',
    # Otros sectores
    'TSLA', 'DIS', 'KO', 'WMT'
]

def es_horario_mercado():
    """Verifica si el mercado está abierto"""
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
    """Calcula múltiples indicadores técnicos"""
    if len(data) < 50:
        return None
        
    # 📊 RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 📈 MACD (Moving Average Convergence Divergence)
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9).mean()
    
    # 📉 SMA (Simple Moving Averages)
    sma_20 = data['Close'].rolling(window=20).mean()
    sma_50 = data['Close'].rolling(window=50).mean()
    
    # 📊 Bollinger Bands
    bb_middle = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    
    # 📈 Volumen promedio
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

def evaluar_señal_avanzada(indicators):
    """Sistema de puntuación avanzado para señales"""
    puntos_compra = 0
    puntos_venta = 0
    razones = []
    
    # 🎯 RSI Analysis
    if indicators['rsi'] < 30:
        puntos_compra += 2
        razones.append(f"🔥 RSI sobreventa ({indicators['rsi']:.1f})")
    elif indicators['rsi'] > 70:
        puntos_venta += 2
        razones.append(f"⚠️ RSI sobrecompra ({indicators['rsi']:.1f})")
    elif indicators['rsi'] < 40:
        puntos_compra += 1
        razones.append(f"📈 RSI favorable compra ({indicators['rsi']:.1f})")
    elif indicators['rsi'] > 60:
        puntos_venta += 1
        razones.append(f"📉 RSI favorable venta ({indicators['rsi']:.1f})")
    
    # 🎯 MACD Analysis
    if indicators['macd'] > indicators['signal_line']:
        if indicators['macd'] > 0:
            puntos_compra += 2
            razones.append("🚀 MACD fuertemente alcista")
        else:
            puntos_compra += 1
            razones.append("📈 MACD cruzando alcista")
    else:
        if indicators['macd'] < 0:
            puntos_venta += 2
            razones.append("📉 MACD fuertemente bajista")
        else:
            puntos_venta += 1
            razones.append("⚠️ MACD cruzando bajista")
    
    # 🎯 SMA Trend Analysis
    precio = indicators['price']
    sma_20 = indicators['sma_20']
    sma_50 = indicators['sma_50']
    
    if precio > sma_20 > sma_50:
        puntos_compra += 2
        razones.append("🔥 Tendencia alcista fuerte (SMA)")
    elif precio > sma_20:
        puntos_compra += 1
        razones.append("📈 Tendenc
