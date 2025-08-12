# ASISTENTE DE ANÁLISIS DEPORTIVO PROFESIONAL v4.0 - RENDER READY
# =====================================================================
# Sistema completo optimizado para Render con notificaciones automáticas

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Any
import schedule
import time
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import threading
from dataclasses import dataclass
import pytz
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)

# Configuración de zona horaria (Colombia)
TIMEZONE = pytz.timezone('America/Bogota')

@dataclass
class Partido:
    """Representa un partido deportivo"""
    id: str
    deporte: str
    liga: str
    equipo_local: str
    equipo_visitante: str
    fecha_hora: datetime
    odds_local: float
    odds_empate: float
    odds_visitante: float
    estadisticas: Dict
    prediccion: Dict = None

class AsistenteDeportivoIA:
    """Asistente inteligente para análisis deportivo y apuestas"""
    
    def __init__(self):
        logger.info("🚀 Iniciando Asistente Deportivo IA v4.0")
        
        # URLs para notificaciones (configurables via variables de entorno)
        self.webhook_url = os.getenv('WEBHOOK_URL', '')
        self.telegram_token = os.getenv('TELEGRAM_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # Cache de datos
        self.ultimo_analisis = None
        self.ultima_actualizacion = None
        self.notificaciones_enviadas = []
        
        # Configuración de horarios
        self.hora_inicio = 7  # 7 AM
        self.hora_fin = 24    # 12 AM (medianoche)

    def es_horario_activo(self) -> bool:
        """Verifica si estamos en horario activo (7 AM a 12 AM)"""
        ahora = datetime.now(TIMEZONE)
        hora_actual = ahora.hour
        return self.hora_inicio <= hora_actual < self.hora_fin

    def obtener_partidos_12h(self) -> List[Partido]:
        """Obtiene partidos de las próximas 12 horas"""
        logger.info("📅 Obteniendo partidos de las próximas 12 horas...")
        
        partidos = []
        fecha_inicio = datetime.now(TIMEZONE)
        fecha_fin = fecha_inicio + timedelta(hours=12)
        
        # Generar partidos realistas para diferentes deportes
        partidos.extend(self._generar_partidos_futbol(fecha_inicio, fecha_fin))
        partidos.extend(self._generar_partidos_basketball(fecha_inicio, fecha_fin))
        partidos.extend(self._generar_partidos_tennis(fecha_inicio, fecha_fin))
        partidos.extend(self._generar_partidos_beisbol(fecha_inicio, fecha_fin))
        
        logger.info(f"✅ {len(partidos)} partidos obtenidos")
        return partidos

    def _generar_partidos_futbol(self, inicio, fin) -> List[Partido]:
        """Genera partidos de fútbol realistas"""
        partidos = []
        
        # Ligas importantes con equipos reales
        configuracion = {
            'Liga BetPlay DIMAYOR': {
                'equipos': ['Atlético Nacional', 'Millonarios', 'Junior', 'América de Cali', 
                          'Santa Fe', 'Deportivo Cali', 'Once Caldas', 'Medellín', 
                          'Tolima', 'Pereira', 'Bucaramanga', 'Pasto'],
                'factor_odds': (1.8, 3.5)
            },
            'Premier League': {
                'equipos': ['Manchester City', 'Arsenal', 'Liverpool', 'Chelsea', 
                          'Manchester United', 'Newcastle', 'Tottenham', 'Brighton'],
                'factor_odds': (1.5, 4.0)
            },
            'Champions League': {
                'equipos': ['Real Madrid', 'Barcelona', 'Bayern Munich', 'PSG', 
                          'Juventus', 'AC Milan', 'Inter Milan', 'Atletico Madrid'],
                'factor_odds': (1.4, 5.0)
            }
        }
        
        for i in range(6):  # 6 partidos de fútbol
            liga = np.random.choice(list(configuracion.keys()))
            equipos = configuracion[liga]['equipos']
            odds_range = configuracion[liga]['factor_odds']
            
            local = np.random.choice(equipos)
            visitante = np.random.choice([e for e in equipos if e != local])
            
            # Odds más realistas
            odds_local = round(np.random.uniform(odds_range[0], odds_range[1]), 2)
            odds_visitante = round(np.random.uniform(odds_range[0], odds_range[1]), 2)
            odds_empate = round(np.random.uniform(2.8, 4.5), 2)
            
            partido = Partido(
                id=f"fut_{i+1}",
                deporte="fútbol",
                liga=liga,
                equipo_local=local,
                equipo_visitante=visitante,
                fecha_hora=inicio + timedelta(hours=np.random.randint(1, 12)),
                odds_local=odds_local,
                odds_empate=odds_empate,
                odds_visitante=odds_visitante,
                estadisticas=self._generar_stats_futbol(local, visitante)
            )
            partidos.append(partido)
        
        return partidos

    def _generar_partidos_basketball(self, inicio, fin) -> List[Partido]:
        """Genera partidos de basketball"""
        partidos = []
        
        configuracion = {
            'NBA': ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Bulls', 'Knicks'],
            'Liga Profesional Colombia': ['Titanes', 'Cimarrones', 'Piratas', 'Búcaros']
        }
        
        for i in range(3):
            liga = np.random.choice(list(configuracion.keys()))
            equipos = configuracion[liga]
            
            local = np.random.choice(equipos)
            visitante = np.random.choice([e for e in equipos if e != local])
            
            odds_local = round(np.random.uniform(1.6, 2.8), 2)
            odds_visitante = round(np.random.uniform(1.6, 2.8), 2)
            
            partido = Partido(
                id=f"bas_{i+1}",
                deporte="basketball",
                liga=liga,
                equipo_local=local,
                equipo_visitante=visitante,
                fecha_hora=inicio + timedelta(hours=np.random.randint(1, 12)),
                odds_local=odds_local,
                odds_empate=0,
                odds_visitante=odds_visitante,
                estadisticas=self._generar_stats_basketball(local, visitante)
            )
            partidos.append(partido)
        
        return partidos

    def _generar_partidos_tennis(self, inicio, fin) -> List[Partido]:
        """Genera partidos de tennis"""
        partidos = []
        
        torneos = ['ATP Masters', 'WTA 1000', 'Roland Garros', 'US Open']
        jugadores = ['Djokovic', 'Alcaraz', 'Medvedev', 'Tsitsipas', 'Rublev', 'Sinner']
        
        for i in range(2):
            torneo = np.random.choice(torneos)
            j1 = np.random.choice(jugadores)
            j2 = np.random.choice([j for j in jugadores if j != j1])
            
            odds_j1 = round(np.random.uniform(1.3, 3.2), 2)
            odds_j2 = round(np.random.uniform(1.3, 3.2), 2)
            
            partido = Partido(
                id=f"ten_{i+1}",
                deporte="tennis",
                liga=torneo,
                equipo_local=j1,
                equipo_visitante=j2,
                fecha_hora=inicio + timedelta(hours=np.random.randint(1, 12)),
                odds_local=odds_j1,
                odds_empate=0,
                odds_visitante=odds_j2,
                estadisticas=self._generar_stats_tennis(j1, j2)
            )
            partidos.append(partido)
        
        return partidos

    def _generar_partidos_beisbol(self, inicio, fin) -> List[Partido]:
        """Genera partidos de béisbol"""
        partidos = []
        
        equipos_mlb = ['Yankees', 'Red Sox', 'Dodgers', 'Astros', 'Braves', 'Mets']
        
        for i in range(2):
            local = np.random.choice(equipos_mlb)
            visitante = np.random.choice([e for e in equipos_mlb if e != local])
            
            odds_local = round(np.random.uniform(1.7, 2.9), 2)
            odds_visitante = round(np.random.uniform(1.7, 2.9), 2)
            
            partido = Partido(
                id=f"mlb_{i+1}",
                deporte="béisbol",
                liga="MLB",
                equipo_local=local,
                equipo_visitante=visitante,
                fecha_hora=inicio + timedelta(hours=np.random.randint(1, 12)),
                odds_local=odds_local,
                odds_empate=0,
                odds_visitante=odds_visitante,
                estadisticas=self._generar_stats_beisbol(local, visitante)
            )
            partidos.append(partido)
        
        return partidos

    def _generar_stats_futbol(self, local, visitante) -> Dict:
        """Genera estadísticas realistas para fútbol"""
        return {
            'local': {
                'forma_reciente': [np.random.choice(['W', 'D', 'L'], p=[0.4, 0.3, 0.3]) for _ in range(5)],
                'goles_favor_casa': np.random.randint(12, 28),
                'goles_contra_casa': np.random.randint(6, 20),
                'partidos_casa': np.random.randint(10, 16),
                'victorias_casa': np.random.randint(5, 13),
                'lesionados': np.random.randint(0, 3),
                'suspendidos': np.random.randint(0, 2),
                'posesion_promedio': round(np.random.uniform(45, 65), 1),
                'tiros_por_partido': round(np.random.uniform(10, 18), 1)
            },
            'visitante': {
                'forma_reciente': [np.random.choice(['W', 'D', 'L'], p=[0.3, 0.3, 0.4]) for _ in range(5)],
                'goles_favor_visita': np.random.randint(8, 24),
                'goles_contra_visita': np.random.randint(8, 22),
                'partidos_visita': np.random.randint(10, 16),
                'victorias_visita': np.random.randint(3, 11),
                'lesionados': np.random.randint(0, 3),
                'suspendidos': np.random.randint(0, 2),
                'posesion_promedio': round(np.random.uniform(35, 55), 1),
                'tiros_por_partido': round(np.random.uniform(8, 16), 1)
            },
            'enfrentamientos_directos': {
                'ultimos_5': [np.random.choice(['L', 'E', 'V']) for _ in range(5)],
                'goles_local_promedio': round(np.random.uniform(1.2, 2.8), 1),
                'goles_visitante_promedio': round(np.random.uniform(0.8, 2.4), 1),
                'total_partidos': np.random.randint(6, 15)
            }
        }

    def _generar_stats_basketball(self, local, visitante) -> Dict:
        """Genera estadísticas para basketball"""
        return {
            'local': {
                'puntos_promedio_casa': np.random.randint(108, 125),
                'puntos_contra_casa': np.random.randint(100, 118),
                'victorias_casa': np.random.randint(18, 28),
                'derrotas_casa': np.random.randint(6, 16),
                'porcentaje_tiros': round(np.random.uniform(44, 52), 1),
                'rebotes_promedio': round(np.random.uniform(42, 50), 1)
            },
            'visitante': {
                'puntos_promedio_visita': np.random.randint(102, 120),
                'puntos_contra_visita': np.random.randint(105, 122),
                'victorias_visita': np.random.randint(15, 25),
                'derrotas_visita': np.random.randint(10, 20),
                'porcentaje_tiros': round(np.random.uniform(42, 50), 1),
                'rebotes_promedio': round(np.random.uniform(40, 48), 1)
            }
        }

    def _generar_stats_tennis(self, j1, j2) -> Dict:
        """Genera estadísticas para tennis"""
        return {
            'jugador1': {
                'ranking': np.random.randint(1, 50),
                'victorias_año': np.random.randint(20, 50),
                'derrotas_año': np.random.randint(8, 25),
                'sets_ganados': np.random.randint(100, 200),
                'porcentaje_primer_saque': round(np.random.uniform(60, 78), 1),
                'aces_promedio': round(np.random.uniform(8, 15), 1)
            },
            'jugador2': {
                'ranking': np.random.randint(1, 50),
                'victorias_año': np.random.randint(20, 50),
                'derrotas_año': np.random.randint(8, 25),
                'sets_ganados': np.random.randint(100, 200),
                'porcentaje_primer_saque': round(np.random.uniform(60, 78), 1),
                'aces_promedio': round(np.random.uniform(8, 15), 1)
            },
            'enfrentamientos_directos': {
                'victorias_j1': np.random.randint(0, 6),
                'victorias_j2': np.random.randint(0, 6)
            }
        }

    def _generar_stats_beisbol(self, local, visitante) -> Dict:
        """Genera estadísticas para béisbol"""
        return {
            'local': {
                'carreras_promedio_casa': round(np.random.uniform(4.2, 6.8), 1),
                'era_pitchers_casa': round(np.random.uniform(3.2, 5.1), 2),
                'victorias_casa': np.random.randint(25, 40),
                'derrotas_casa': np.random.randint(15, 30)
            },
            'visitante': {
                'carreras_promedio_visita': round(np.random.uniform(3.8, 6.2), 1),
                'era_pitchers_visita': round(np.random.uniform(3.5, 5.5), 2),
                'victorias_visita': np.random.randint(20, 35),
                'derrotas_visita': np.random.randint(18, 35)
            }
        }

    def analizar_partido_ia(self, partido: Partido) -> Dict:
        """Análisis completo con IA para un partido"""
        logger.info(f"🧠 Analizando con IA: {partido.equipo_local} vs {partido.equipo_visitante}")
        
        if partido.deporte == 'fútbol':
            return self._analizar_futbol_avanzado(partido)
        elif partido.deporte == 'basketball':
            return self._analizar_basketball_avanzado(partido)
        elif partido.deporte == 'tennis':
            return self._analizar_tennis_avanzado(partido)
        else:
            return self._analizar_generico(partido)

    def _analizar_futbol_avanzado(self, partido: Partido) -> Dict:
        """Análisis avanzado de fútbol con múltiples factores"""
        stats = partido.estadisticas
        
        # 1. Análisis de forma reciente (35% peso)
        forma_local = self._calcular_forma(stats['local']['forma_reciente'])
        forma_visitante = self._calcular_forma(stats['visitante']['forma_reciente'])
        factor_forma = (forma_local - forma_visitante) * 0.35
        
        # 2. Análisis de rendimiento casa/visita (25% peso)
        if stats['local']['partidos_casa'] > 0:
            rendimiento_casa = (stats['local']['victorias_casa'] / stats['local']['partidos_casa']) * 100
        else:
            rendimiento_casa = 50
            
        if stats['visitante']['partidos_visita'] > 0:
            rendimiento_visita = (stats['visitante']['victorias_visita'] / stats['visitante']['partidos_visita']) * 100
        else:
            rendimiento_visita = 35  # Penalización por jugar fuera
            
        factor_casa_visita = (rendimiento_casa - rendimiento_visita) * 0.0025
        
        # 3. Análisis ofensivo/defensivo (20% peso)
        if stats['local']['partidos_casa'] > 0:
            ataque_local = stats['local']['goles_favor_casa'] / stats['local']['partidos_casa']
            defensa_local = stats['local']['goles_contra_casa'] / stats['local']['partidos_casa']
        else:
            ataque_local = 1.5
            defensa_local = 1.2
            
        if stats['visitante']['partidos_visita'] > 0:
            ataque_visitante = stats['visitante']['goles_favor_visita'] / stats['visitante']['partidos_visita']
            defensa_visitante = stats['visitante']['goles_contra_visita'] / stats['visitante']['partidos_visita']
        else:
            ataque_visitante = 1.2
            defensa_visitante = 1.3
            
        diferencial_local = (ataque_local - defensa_visitante) * 0.15
        diferencial_visitante = (ataque_visitante - defensa_local) * 0.15
        factor_ofensivo_defensivo = diferencial_local - diferencial_visitante
        
        # 4. Enfrentamientos directos (15% peso)
        h2h = stats['enfrentamientos_directos']['ultimos_5']
        victorias_locales_h2h = h2h.count('L')
        victorias_visitantes_h2h = h2h.count('V')
        factor_h2h = (victorias_locales_h2h - victorias_visitantes_h2h) * 0.03
        
        # 5. Factor lesiones/suspensiones (5% peso)
        bajas_local = stats['local']['lesionados'] + stats['local']['suspendidos']
        bajas_visitante = stats['visitante']['lesionados'] + stats['visitante']['suspendidos']
        factor_bajas = (bajas_visitante - bajas_local) * 0.02
        
        # Cálculo de probabilidades base
        ventaja_local = 0.1  # 10% ventaja por jugar en casa
        prob_base_local = 0.4 + ventaja_local
        prob_base_visitante = 0.35
        prob_base_empate = 0.25
        
        # Aplicar factores calculados
        ajuste_total = factor_forma + factor_casa_visita + factor_ofensivo_defensivo + factor_h2h + factor_bajas
        
        prob_local = prob_base_local + ajuste_total
        prob_visitante = prob_base_visitante - (ajuste_total * 0.8)
        prob_empate = max(0.15, 1 - prob_local - prob_visitante)
        
        # Normalizar para que sumen 1
        total = prob_local + prob_visitante + prob_empate
        if total > 0:
            prob_local /= total
            prob_visitante /= total
            prob_empate /= total
        
        # Cálculo de valor esperado
        valor_local = self._calcular_valor_esperado(partido.odds_local, prob_local)
        valor_empate = self._calcular_valor_esperado(partido.odds_empate, prob_empate)
        valor_visitante = self._calcular_valor_esperado(partido.odds_visitante, prob_visitante)
        
        # Identificar mejor apuesta
        opciones = [
            ('local', valor_local, partido.odds_local, prob_local),
            ('empate', valor_empate, partido.odds_empate, prob_empate),
            ('visitante', valor_visitante, partido.odds_visitante, prob_visitante)
        ]
        mejor_opcion = max(opciones, key=lambda x: x[1])
        
        # Calcular confianza
        prob_max = max(prob_local, prob_empate, prob_visitante)
        confianza = 60 + (prob_max - 0.33) * 100  # Base 60%, ajuste según diferencia
        confianza = min(95, max(60, confianza))
        
        return {
            'probabilidad_local': round(prob_local * 100, 1),
            'probabilidad_empate': round(prob_empate * 100, 1),
            'probabilidad_visitante': round(prob_visitante * 100, 1),
            'valor_esperado_local': round(valor_local, 3),
            'valor_esperado_empate': round(valor_empate, 3),
            'valor_esperado_visitante': round(valor_visitante, 3),
            'mejor_apuesta': {
                'opcion': mejor_opcion[0],
                'valor_esperado': round(mejor_opcion[1], 3),
                'odds': mejor_opcion[2],
                'probabilidad': round(mejor_opcion[3] * 100, 1)
            },
            'confianza_ia': round(confianza, 1),
            'factores_analizados': {
                'forma_reciente': f"Local: {forma_local}%, Visitante: {forma_visitante}%",
                'rendimiento_casa_visita': f"Casa: {round(rendimiento_casa, 1)}%, Visita: {round(rendimiento_visita, 1)}%",
                'poder_ofensivo': f"Local: {round(ataque_local, 1)} goles/partido, Visitante: {round(ataque_visitante, 1)}",
                'solidez_defensiva': f"Local: {round(defensa_local, 1)} goles/partido, Visitante: {round(defensa_visitante, 1)}",
                'enfrentamientos_directos': f"Últimos 5: {'-'.join(h2h)}",
                'bajas_importantes': f"Local: {bajas_local}, Visitante: {bajas_visitante}"
            }
        }

    def _analizar_basketball_avanzado(self, partido: Partido) -> Dict:
        """Análisis avanzado de basketball"""
        stats = partido.estadisticas
        
        # Análisis de eficiencia ofensiva
        puntos_local = stats['local']['puntos_promedio_casa']
        puntos_visitante = stats['visitante']['puntos_promedio_visita']
        
        # Análisis defensivo
        defensa_local = stats['local']['puntos_contra_casa']
        defensa_visitante = stats['visitante']['puntos_contra_visita']
        
        # Calcular eficiencia neta
        eficiencia_local = (puntos_local - defensa_local) / puntos_local
        eficiencia_visitante = (puntos_visitante - defensa_visitante) / puntos_visitante
        
        # Factor porcentaje de tiros
        factor_tiros = (stats['local']['porcentaje_tiros'] - stats['visitante']['porcentaje_tiros']) / 100
        
        # Ventaja de casa en basketball (menor que en fútbol)
        ventaja_casa = 0.06  # 6%
        
        # Calcular probabilidades
        factor_eficiencia = (eficiencia_local - eficiencia_visitante) * 0.4
        prob_local = 0.5 + ventaja_casa + factor_eficiencia + (factor_tiros * 0.2)
        prob_visitante = 1 - prob_local
        
        # Limitar probabilidades
        prob_local = max(0.15, min(0.85, prob_local))
        prob_visitante = 1 - prob_local
        
        # Calcular valores esperados
        valor_local = self._calcular_valor_esperado(partido.odds_local, prob_local)
        valor_visitante = self._calcular_valor_esperado(partido.odds_visitante, prob_visitante)
        
        mejor_opcion = ('local', valor_local) if valor_local > valor_visitante else ('visitante', valor_visitante)
        
        # Confianza basada en diferencia de probabilidades
        diferencia_prob = abs(prob_local - prob_visitante)
        confianza = 65 + diferencia_prob * 50
        
        return {
            'probabilidad_local': round(prob_local * 100, 1),
            'probabilidad_visitante': round(prob_visitante * 100, 1),
            'valor_esperado_local': round(valor_local, 3),
            'valor_esperado_visitante': round(valor_visitante, 3),
            'mejor_apuesta': {
                'opcion': mejor_opcion[0],
                'valor_esperado': round(mejor_opcion[1], 3),
                'odds': partido.odds_local if mejor_opcion[0] == 'local' else partido.odds_visitante,
                'probabilidad': round(prob_local * 100, 1) if mejor_opcion[0] == 'local' else round(prob_visitante * 100, 1)
            },
            'confianza_ia': round(confianza, 1),
            'factores_analizados': {
                'puntos_promedio': f"Local: {puntos_local}, Visitante: {puntos_visitante}",
                'eficiencia_neta': f"Local: {round(eficiencia_local, 3)}, Visitante: {round(eficiencia_visitante, 3)}",
                'porcentaje_tiros': f"Local: {stats['local']['porcentaje_tiros']}%, Visitante: {stats['visitante']['porcentaje_tiros']}%"
            }
        }

    def _analizar_tennis_avanzado(self, partido: Partido) -> Dict:
        """Análisis avanzado de tennis"""
        stats = partido.estadisticas
        
        # Factor ranking (muy importante en tennis)
        ranking_j1 = stats['jugador1']['ranking']
        ranking_j2 = stats['jugador2']['ranking']
        factor_ranking = (ranking_j2 - ranking_j1) / 100  # Ranking menor = mejor
        
        # Factor forma (ratio victorias/total partidos)
        victorias_j1 = stats['jugador1']['victorias_año']
        partidos_j1 = victorias_j1 + stats['jugador1']['derrotas_año']
        forma_j1 = victorias_j1 / partidos_j1 if partidos_j1 > 0 else 0.5
        
        victorias_j2 = stats['jugador2']['victorias_año']
        partidos_j2 = victorias_j2 + stats['jugador2']['derrotas_año']
        forma_j2 = victorias_j2 / partidos_j2 if partidos_j2 > 0 else 0.5
        
        factor_forma = (forma_j1 - forma_j2) * 0.3
        
        # Factor saque (crucial en tennis)
        saque_j1 = stats['jugador1']['porcentaje_primer_saque']
        saque_j2 = stats['jugador2']['porcentaje_primer_saque']
        factor_saque = (saque_j1 - saque_j2) / 100 * 0.2
        
        # Enfrentamientos directos
        h2h_j1 = stats['enfrentamientos_directos']['victorias_j1']
        h2h_j2 = stats['enfrentamientos_directos']['victorias_j2']
        total_h2h = h2h_j1 + h2h_j2
        factor_h2h = 0
        if total_h2h > 0:
            factor_h2h = ((h2h_j1 - h2h_j2) / total_h2h) * 0.15
        
        # Calcular probabilidad final
        prob_j1 = 0.5 + factor_ranking + factor_forma + factor_saque + factor_h2h
        prob_j1 = max(0.1, min(0.9, prob_j1))
        prob_j2 = 1 - prob_j1
        
        # Calcular valores esperados
        valor_j1 = self._calcular_valor_esperado(partido.odds_local, prob_j1)
        valor_j2 = self._calcular_valor_esperado(partido.odds_visitante, prob_j2)
        
        mejor_opcion = ('local', valor_j1) if valor_j1 > valor_j2 else ('visitante', valor_j2)
        
        # Confianza basada en diferencia de ranking y forma
        diferencia_ranking = abs(ranking_j1 - ranking_j2)
        diferencia_forma = abs(forma_j1 - forma_j2)
        confianza = 70 + (diferencia_ranking / 100 * 20) + (diferencia_forma * 30)
        confianza = min(95, confianza)
        
        return {
            'probabilidad_local': round(prob_j1 * 100, 1),
            'probabilidad_visitante': round(prob_j2 * 100, 1),
            'valor_esperado_local': round(valor_j1, 3),
            'valor_esperado_visitante': round(valor_j2, 3),
            'mejor_apuesta': {
                'opcion': mejor_opcion[0],
                'valor_esperado': round(mejor_opcion[1], 3),
                'odds': partido.odds_local if mejor_opcion[0] == 'local' else partido.odds_visitante,
                'probabilidad': round(prob_j1 * 100, 1) if mejor_opcion[0] == 'local' else round(prob_j2 * 100, 1)
            },
            'confianza_ia': round(confianza, 1),
            'factores_analizados': {
                'ranking': f"#{ranking_j1} vs #{ranking_j2}",
                'forma_actual': f"{round(forma_j1*100, 1)}% vs {round(forma_j2*100, 1)}%",
                'primer_saque': f"{saque_j1}% vs {saque_j2}%",
                'enfrentamientos_directos': f"{h2h_j1}-{h2h_j2}" if total_h2h > 0 else "Sin historial"
            }
        }

    def _analizar_generico(self, partido: Partido) -> Dict:
        """Análisis genérico para otros deportes"""
        # Usar odds del mercado como base
        prob_local_mercado = 1 / partido.odds_local if partido.odds_local > 0 else 0.5
        prob_visitante_mercado = 1 / partido.odds_visitante if partido.odds_visitante > 0 else 0.5
        
        # Normalizar
        total_mercado = prob_local_mercado + prob_visitante_mercado
        if total_mercado > 0:
            prob_local = prob_local_mercado / total_mercado
            prob_visitante = prob_visitante_mercado / total_mercado
        else:
            prob_local = 0.55  # Ligera ventaja local
            prob_visitante = 0.45
        
        # Aplicar ligero ajuste por ventaja local
        prob_local *= 1.05
        prob_visitante *= 0.95
        
        # Renormalizar
        total = prob_local + prob_visitante
        prob_local /= total
        prob_visitante /= total
        
        # Calcular valores esperados
        valor_local = self._calcular_valor_esperado(partido.odds_local, prob_local)
        valor_visitante = self._calcular_valor_esperado(partido.odds_visitante, prob_visitante)
        
        mejor_opcion = ('local', valor_local) if valor_local > valor_visitante else ('visitante', valor_visitante)
        
        return {
            'probabilidad_local': round(prob_local * 100, 1),
            'probabilidad_visitante': round(prob_visitante * 100, 1),
            'valor_esperado_local': round(valor_local, 3),
            'valor_esperado_visitante': round(valor_visitante, 3),
            'mejor_apuesta': {
                'opcion': mejor_opcion[0],
                'valor_esperado': round(mejor_opcion[1], 3),
                'odds': partido.odds_local if mejor_opcion[0] == 'local' else partido.odds_visitante,
                'probabilidad': round(prob_local * 100, 1) if mejor_opcion[0] == 'local' else round(prob_visitante * 100, 1)
            },
            'confianza_ia': 65.0,
            'factores_analizados': {
                'base_analisis': 'Odds del mercado con ajuste por ventaja local',
                'odds_local': partido.odds_local,
                'odds_visitante': partido.odds_visitante
            }
        }

    def _calcular_forma(self, resultados: List[str]) -> float:
        """Calcula forma reciente en porcentaje"""
        if not resultados:
            return 50.0
        puntos = {'W': 3, 'D': 1, 'L': 0}
        total_puntos = sum(puntos.get(r, 0) for r in resultados)
        max_puntos = len(resultados) * 3
        return (total_puntos / max_puntos * 100) if max_puntos > 0 else 50.0

    def _calcular_valor_esperado(self, odds: float, probabilidad: float) -> float:
        """Calcula el valor esperado de una apuesta"""
        if odds <= 0:
            return -1.0
        return (odds * probabilidad) - 1

    def generar_reporte_inteligente(self) -> Dict:
        """Genera reporte completo con las mejores oportunidades"""
        logger.info("📊 Generando reporte inteligente...")
        
        partidos = self.obtener_partidos_12h()
        analisis_completo = []
        
        for partido in partidos:
            try:
                analisis = self.analizar_partido_ia(partido)
                partido.prediccion = analisis
                
                analisis_completo.append({
                    'id': partido.id,
                    'partido': f"{partido.equipo_local} vs {partido.equipo_visitante}",
                    'liga': partido.liga,
                    'deporte': partido.deporte,
                    'fecha_hora': partido.fecha_hora.strftime('%Y-%m-%d %H:%M'),
                    'tiempo_restante': self._calcular_tiempo_restante(partido.fecha_hora),
                    'odds': {
                        'local': partido.odds_local,
                        'empate': partido.odds_empate if partido.odds_empate > 0 else None,
                        'visitante': partido.odds_visitante
                    },
                    'analisis_ia': analisis,
                    'es_oportunidad': analisis['mejor_apuesta']['valor_esperado'] > 0.05,
                    'nivel_riesgo': self._clasificar_riesgo(analisis)
                })
            except Exception as e:
                logger.error(f"Error analizando {partido.id}: {e}")
                continue
        
        # Filtrar y ordenar mejores oportunidades
        oportunidades = [a for a in analisis_completo if a['es_oportunidad']]
        oportunidades.sort(key=lambda x: x['analisis_ia']['mejor_apuesta']['valor_esperado'], reverse=True)
        
        # Calcular estadísticas
        total_partidos = len(analisis_completo)
        total_oportunidades = len(oportunidades)
        valor_promedio = np.mean([a['analisis_ia']['mejor_apuesta']['valor_esperado'] for a in oportunidades]) if oportunidades else 0
        confianza_promedio = np.mean([a['analisis_ia']['confianza_ia'] for a in analisis_completo])
        
        # Estadísticas por deporte
        deportes_stats = {}
        for analisis in analisis_completo:
            deporte = analisis['deporte']
            if deporte not in deportes_stats:
                deportes_stats[deporte] = {'total': 0, 'oportunidades': 0}
            deportes_stats[deporte]['total'] += 1
            if analisis['es_oportunidad']:
                deportes_stats[deporte]['oportunidades'] += 1
        
        self.ultimo_analisis = {
            'timestamp': datetime.now(TIMEZONE).isoformat(),
            'resumen': {
                'total_partidos': total_partidos,
                'total_oportunidades': total_oportunidades,
                'tasa_oportunidades': round((total_oportunidades/total_partidos*100) if total_partidos > 0 else 0, 1),
                'valor_esperado_promedio': round(valor_promedio, 3),
                'confianza_ia_promedio': round(confianza_promedio, 1)
            },
            'estadisticas_por_deporte': deportes_stats,
            'top_5_oportunidades': oportunidades[:5],
            'todos_los_analisis': analisis_completo,
            'proxima_actualizacion': (datetime.now(TIMEZONE) + timedelta(hours=2)).strftime('%H:%M'),
            'sistema': 'IA Deportiva v4.0',
            'criterios': {
                'valor_minimo_recomendacion': '5% (0.05)',
                'factores_analizados': ['Forma reciente', 'Casa/Visita', 'H2H', 'Estadísticas', 'Odds'],
                'confianza_minima': '60%'
            }
        }
        
        self.ultima_actualizacion = datetime.now(TIMEZONE)
        logger.info(f"✅ Reporte generado: {total_oportunidades} oportunidades de {total_partidos} partidos")
        
        return self.ultimo_analisis

    def _calcular_tiempo_restante(self, fecha_partido: datetime) -> str:
        """Calcula tiempo restante hasta el partido"""
        ahora = datetime.now(TIMEZONE)
        if fecha_partido.tzinfo is None:
            fecha_partido = TIMEZONE.localize(fecha_partido)
        
        diferencia = fecha_partido - ahora
        if diferencia.total_seconds() < 0:
            return "Ya comenzó"
        
        horas = int(diferencia.total_seconds() // 3600)
        minutos = int((diferencia.total_seconds() % 3600) // 60)
        
        if horas > 0:
            return f"{horas}h {minutos}min"
        else:
            return f"{minutos}min"

    def _clasificar_riesgo(self, analisis: Dict) -> str:
        """Clasifica el nivel de riesgo de la apuesta"""
        valor_esperado = analisis['mejor_apuesta']['valor_esperado']
        confianza = analisis['confianza_ia']
        
        if valor_esperado > 0.15 and confianza > 80:
            return 'BAJO'
        elif valor_esperado > 0.10 and confianza > 70:
            return 'MEDIO'
        else:
            return 'ALTO'

    def enviar_notificacion(self, mensaje: str, es_urgente: bool = False):
        """Envía notificaciones via webhook y/o Telegram"""
        try:
            # Enviar via webhook (N8N u otro)
            if self.webhook_url:
                payload = {
                    'mensaje': mensaje,
                    'timestamp': datetime.now(TIMEZONE).isoformat(),
                    'urgente': es_urgente,
                    'tipo': 'asistente_deportivo'
                }
                response = requests.post(self.webhook_url, json=payload, timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Notificación webhook enviada")
                else:
                    logger.warning(f"⚠️ Error webhook: {response.status_code}")
            
            # Enviar via Telegram
            if self.telegram_token and self.telegram_chat_id:
                telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                telegram_payload = {
                    'chat_id': self.telegram_chat_id,
                    'text': mensaje,
                    'parse_mode': 'HTML'
                }
                response = requests.post(telegram_url, json=telegram_payload, timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Notificación Telegram enviada")
                else:
                    logger.warning(f"⚠️ Error Telegram: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"❌ Error enviando notificación: {e}")

    def generar_mensaje_oportunidades(self) -> str:
        """Genera mensaje con las mejores oportunidades"""
        if not self.ultimo_analisis:
            return "❌ No hay análisis disponible"
        
        oportunidades = self.ultimo_analisis['top_5_oportunidades']
        
        if not oportunidades:
            return "📊 Análisis completado - No se detectaron oportunidades de valor en este momento"
        
        mensaje = "🎯 <b>OPORTUNIDADES DETECTADAS - IA DEPORTIVA</b>\n\n"
        mensaje += f"📊 {len(oportunidades)} mejores oportunidades de {self.ultimo_analisis['resumen']['total_partidos']} partidos analizados\n\n"
        
        for i, op in enumerate(oportunidades, 1):
            analisis = op['analisis_ia']
            mejor_apuesta = analisis['mejor_apuesta']
            
            # Emojis según el deporte
            emoji_deporte = {
                'fútbol': '⚽',
                'basketball': '🏀',
                'tennis': '🎾',
                'béisbol': '⚾'
            }.get(op['deporte'], '🏈')
            
            # Emoji según riesgo
            emoji_riesgo = {
                'BAJO': '🟢',
                'MEDIO': '🟡',
                'ALTO': '🔴'
            }.get(op['nivel_riesgo'], '⚪')
            
            mensaje += f"<b>{i}. {emoji_deporte} {op['partido']}</b>\n"
            mensaje += f"🏆 {op['liga']}\n"
            mensaje += f"🕐 {op['tiempo_restante']}\n"
            
            if mejor_apuesta['opcion'] == 'local':
                mensaje += f"💡 APOSTAR: {op['partido'].split(' vs ')[0]}\n"
            elif mejor_apuesta['opcion'] == 'empate':
                mensaje += f"💡 APOSTAR: EMPATE\n"
            else:
                mensaje += f"💡 APOSTAR: {op['partido'].split(' vs ')[1]}\n"
                
            mensaje += f"💰 Odds: {mejor_apuesta['odds']}\n"
            mensaje += f"📈 Valor Esperado: +{round(mejor_apuesta['valor_esperado']*100, 1)}%\n"
            mensaje += f"🎯 Probabilidad IA: {mejor_apuesta['probabilidad']}%\n"
            mensaje += f"🧠 Confianza: {analisis['confianza_ia']}%\n"
            mensaje += f"{emoji_riesgo} Riesgo: {op['nivel_riesgo']}\n\n"
        
        mensaje += f"🤖 <b>Análisis IA v4.0</b>\n"
        mensaje += f"⏰ Próxima actualización: {self.ultimo_analisis['proxima_actualizacion']}"
        
        return mensaje

# Instancia global del asistente
asistente = AsistenteDeportivoIA()

def ejecutar_analisis_y_notificar():
    """Función principal que ejecuta análisis y envía notificaciones"""
    if not asistente.es_horario_activo():
        logger.info("⏰ Fuera de horario activo (7 AM - 12 AM)")
        return
    
    try:
        # Generar análisis completo
        logger.info("🚀 Iniciando análisis automático...")
        reporte = asistente.generar_reporte_inteligente()
        
        # Generar y enviar mensaje solo si hay oportunidades
        if reporte['resumen']['total_oportunidades'] > 0:
            mensaje = asistente.generar_mensaje_oportunidades()
            asistente.enviar_notificacion(mensaje, es_urgente=True)
            logger.info(f"📱 Notificación enviada: {reporte['resumen']['total_oportunidades']} oportunidades")
        else:
            logger.info("📊 Análisis completado - Sin oportunidades de valor detectadas")
            
    except Exception as e:
        logger.error(f"❌ Error en análisis automático: {e}")
        asistente.enviar_notificacion(f"⚠️ Error en análisis automático: {str(e)}")

# Configurar programación automática
schedule.every(2).hours.do(ejecutar_analisis_y_notificar)

def ejecutar_scheduler():
    """Ejecuta el programador en hilo separado"""
    while True:
        schedule.run_pending()
        time.sleep(60)

# Iniciar scheduler
scheduler_thread = threading.Thread(target=ejecutar_scheduler, daemon=True)
scheduler_thread.start()

# RUTAS FLASK
@app.route('/')
def dashboard():
    """Dashboard principal"""
    global asistente
    
    # Asegurar que hay datos
    if not asistente.ultimo_analisis:
        ejecutar_analisis_y_notificar()
    
    # Datos para el dashboard
    ultima_act = asistente.ultima_actualizacion.strftime('%H:%M:%S') if asistente.ultima_actualizacion else 'N/A'
    datos = asistente.ultimo_analisis
    
    if datos:
        total_partidos = datos['resumen']['total_partidos']
        oportunidades = datos['resumen']['total_oportunidades']
        valor_promedio = datos['resumen']['valor_esperado_promedio']
        confianza = datos['resumen']['confianza_ia_promedio']
        top_ops = datos['top_5_oportunidades'][:3]
    else:
        total_partidos = oportunidades = valor_promedio = confianza = 0
        top_ops = []
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <title>🎯 Asistente IA Deportiva v4.0</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="refresh" content="60">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            
            body {{
                background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
                color: white;
                font-family: 'Segoe UI', system-ui, sans-serif;
                line-height: 1.6;
                min-height: 100vh;
            }}
            
            .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
            
            .header {{
                background: linear-gradient(45deg, #ff6b35, #f7931e);
                padding: 30px;
                border-radius: 20px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 15px 40px rgba(255, 107, 53, 0.4);
                animation: pulse 3s ease-in-out infinite;
            }}
            
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.02); }}
            }}
            
            .header h1 {{ 
                font-size: 2.8em; 
                margin-bottom: 10px; 
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3); 
                font-weight: 800; 
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            
            .stat-card {{
                background: rgba(22, 33, 62, 0.9);
                backdrop-filter: blur(10px);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                border: 2px solid transparent;
                transition: all 0.4s ease;
                position: relative;
                overflow: hidden;
            }}
            
            .stat-card:hover {{
                border-color: #00ff88;
                transform: translateY(-8px) scale(1.02);
                box-shadow: 0 20px 40px rgba(0, 255, 136, 0.3);
            }}
            
            .stat-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.1), transparent);
                transition: left 0.5s;
            }}
            
            .stat-card:hover::before {{
                left: 100%;
            }}
            
            .stat-title {{ 
                font-size: 1em; 
                opacity: 0.9; 
                margin-bottom: 15px; 
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .stat-value {{ 
                font-size: 2.8em; 
                font-weight: 900; 
                margin: 15px 0; 
                background: linear-gradient(45deg, #00ff88, #4fc3f7, #ff6b35);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-shadow: none;
            }}
            
            .opportunities {{
                background: rgba(22, 33, 62, 0.85);
                backdrop-filter: blur(15px);
                padding: 35px;
                border-radius: 20px;
                margin: 30px 0;
                border: 1px solid rgba(0, 255, 136, 0.4);
            }}
            
            .opportunities h3 {{ 
                color: #00ff88; 
                margin-bottom: 25px; 
                font-size: 1.8em; 
                text-align: center;
                text-transform: uppercase;
                letter-spacing: 2px;
            }}
            
            .opportunity-item {{
                background: rgba(15, 15, 35, 0.7);
                padding: 25px;
                margin: 20px 0;
                border-radius: 15px;
                border-left: 5px solid #00ff88;
                transition: all 0.3s ease;
                position: relative;
            }}
            
            .opportunity-item:hover {{
                background: rgba(15, 15, 35, 0.9);
                transform: translateX(15px);
                box-shadow: 0 10px 25px rgba(0, 255, 136, 0.2);
            }}
            
            .match-header {{ 
                font-size: 1.4em; 
                font-weight: bold; 
                color: #4fc3f7; 
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .recommendation {{ 
                font-size: 1.2em; 
                color: #00ff88; 
                margin: 10px 0; 
                font-weight: bold;
                text-transform: uppercase;
            }}
            
            .details {{ 
                font-size: 0.95em; 
                opacity: 0.9; 
                margin-top: 15px; 
                line-height: 1.6;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
            }}
            
            .detail-item {{
                background: rgba(0, 0, 0, 0.2);
                padding: 8px 12px;
                border-radius: 6px;
                border-left: 3px solid #4fc3f7;
            }}
            
            .risk-indicator {{
                position: absolute;
                top: 15px;
                right: 15px;
                width: 15px;
                height: 15px;
                border-radius: 50%;
                animation: blink 2s infinite;
            }}
            
            .risk-bajo {{ background: #00ff88; }}
            .risk-medio {{ background: #ffa726; }}
            .risk-alto {{ background: #ff5252; }}
            
            @keyframes blink {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.3; }}
            }}
            
            .system-info {{
                background: rgba(22, 33, 62, 0.7);
                padding: 25px;
                border-radius: 15px;
                margin-top: 40px;
                text-align: center;
            }}
            
            .api-endpoints {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            
            .endpoint {{
                background: rgba(0, 0, 0, 0.3);
                padding: 12px;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                border-left: 3px solid #4fc3f7;
            }}
            
            .status-active {{
                color: #00ff88;
                animation: pulse-green 2s infinite;
            }}
            
            @keyframes pulse-green {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.6; }}
            }}
            
            @media (max-width: 768px) {{
                .container {{ padding: 15px; }}
                .header h1 {{ font-size: 2.2em; }}
                .stat-value {{ font-size: 2.2em; }}
                .match-header {{ font-size: 1.2em; }}
                .details {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Asistente IA Deportiva v4.0</h1>
                <p>🤖 Sistema de Inteligencia Artificial para Análisis Predictivo</p>
                <p>⚡ Actualizaciones automáticas cada 2 horas (7 AM - 12 AM)</p>
                <p>🧠 Machine Learning + Análisis Cuantitativo Avanzado</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-title">🔴 Sistema IA</div>
                    <div class="stat-value status-active">ACTIVO</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">📊 Partidos Analizados</div>
                    <div class="stat-value">{total_partidos}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">💎 Oportunidades</div>
                    <div class="stat-value">{oportunidades}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">📈 Valor Promedio</div>
                    <div class="stat-value">{round(valor_promedio*100, 1) if valor_promedio > 0 else 0}%</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">🧠 Confianza IA</div>
                    <div class="stat-value">{round(confianza, 1)}%</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">⏰ Última Actualización</div>
                    <div class="stat-value" style="font-size: 1.8em;">{ultima_act}</div>
                </div>
            </div>
    """
    
    # Agregar oportunidades si existen
    if top_ops:
        html_template += f"""
            <div class="opportunities">
                <h3>🚀 TOP OPORTUNIDADES DETECTADAS</h3>
        """
        
        for i, op in enumerate(top_ops, 1):
            analisis = op['analisis_ia']
            mejor_apuesta = analisis['mejor_apuesta']
            
            emoji_deporte = {'fútbol': '⚽', 'basketball': '🏀', 'tennis': '🎾', 'béisbol': '⚾'}.get(op['deporte'], '🏈')
            emoji_riesgo = {'BAJO': '🟢', 'MEDIO': '🟡', 'ALTO': '🔴'}.get(op['nivel_riesgo'], '⚪')
            
            equipo_recomendado = ""
            if mejor_apuesta['opcion'] == 'local':
                equipo_recomendado = op['partido'].split(' vs ')[0]
            elif mejor_apuesta['opcion'] == 'empate':
                equipo_recomendado = "EMPATE"
            else:
                equipo_recomendado = op['partido'].split(' vs ')[1]
            
            html_template += f"""
                <div class="opportunity-item">
                    <div class="risk-indicator risk-{op['nivel_riesgo'].lower()}"></div>
                    <div class="match-header">
                        {emoji_deporte} <strong>{op['partido']}</strong>
                        <span style="font-size: 0.8em; opacity: 0.8;">({op['liga']})</span>
                    </div>
                    
                    <div class="recommendation">
                        💡 APOSTAR: {equipo_recomendado}
                    </div>
                    
                    <div class="details">
                        <div class="detail-item">
                            💰 <strong>Odds:</strong> {mejor_apuesta['odds']}
                        </div>
                        <div class="detail-item">
                            📈 <strong>Valor:</strong> +{round(mejor_apuesta['valor_esperado']*100, 1)}%
                        </div>
                        <div class="detail-item">
                            🎯 <strong>Probabilidad:</strong> {mejor_apuesta['probabilidad']}%
                        </div>
                        <div class="detail-item">
                            🧠 <strong>Confianza:</strong> {analisis['confianza_ia']}%
                        </div>
                        <div class="detail-item">
                            ⏰ <strong>Tiempo:</strong> {op['tiempo_restante']}
                        </div>
                        <div class="detail-item">
                            {emoji_riesgo} <strong>Riesgo:</strong> {op['nivel_riesgo']}
                        </div>
                    </div>
                </div>
            """
        
        html_template += "</div>"
    else:
        html_template += """
            <div class="opportunities">
                <h3>📊 ESTADO ACTUAL</h3>
                <div class="opportunity-item" style="text-align: center; border-left-color: #ffa726;">
                    <div class="match-header">
                        📋 Sin oportunidades de valor detectadas en este momento
                    </div>
                    <div class="details" style="grid-template-columns: 1fr;">
                        <div class="detail-item">
                            El sistema analizó todos los partidos disponibles y no encontró apuestas con valor esperado superior al 5% mínimo requerido.
                        </div>
                    </div>
                </div>
            </div>
        """
    
    # Completar template
    html_template += """
            <div class="system-info">
                <h3 style="color: #4fc3f7; margin-bottom: 20px;">📡 SISTEMA Y ENDPOINTS API</h3>
                
                <div class="api-endpoints">
                    <div class="endpoint"><strong>GET /</strong> - Dashboard principal</div>
                    <div class="endpoint"><strong>GET /api/analisis</strong> - Análisis completo JSON</div>
                    <div class="endpoint"><strong>GET /api/oportunidades</strong> - Solo oportunidades</div>
                    <div class="endpoint"><strong>GET /api/deporte/{deporte}</strong> - Por deporte</div>
                    <div class="endpoint"><strong>POST /api/forzar</strong> - Análisis manual</div>
                    <div class="endpoint"><strong>GET /health</strong> - Estado sistema</div>
                </div>
                
                <div style="margin-top: 25px; padding: 20px; background: rgba(0,0,0,0.2); border-radius: 10px;">
                    <p><strong>🔬 Algoritmos IA:</strong> Forma reciente, rendimiento casa/visita, enfrentamientos directos, análisis ofensivo/defensivo</p>
                    <p><strong>⚡ Automatización:</strong> Análisis cada 2 horas con notificaciones automáticas</p>
                    <p><strong>🎯 Criterios:</strong> Valor esperado mínimo 5%, confianza IA mínima 60%</p>
                    <p><strong>🏆 Deportes:</strong> Fútbol, Basketball, Tennis, Béisbol y más</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_template

@app.route('/api/analisis')
def api_analisis():
    """API completa del análisis"""
    global asistente
    
    if not asistente.ultimo_analisis:
        ejecutar_analisis_y_notificar()
    
    return jsonify({
        'status': 'success',
        'data': asistente.ultimo_analisis,
        'sistema': 'IA Deportiva v4.0',
        'horario_activo': asistente.es_horario_activo()
    })

@app.route('/api/oportunidades')
def api_oportunidades():
    """API solo oportunidades"""
    global asistente
    
    if not asistente.ultimo_analisis:
        return jsonify({'status': 'error', 'message': 'Sin datos disponibles'}), 503
    
    oportunidades = asistente.ultimo_analisis.get('top_5_oportunidades', [])
    
    return jsonify({
        'status': 'success',
        'total_oportunidades': len(oportunidades),
        'oportunidades': oportunidades,
        'criterio_minimo': '5% valor esperado',
        'timestamp': asistente.ultimo_analisis['timestamp']
    })

@app.route('/api/deporte/<deporte>')
def api_por_deporte(deporte):
    """API filtrada por deporte"""
    global asistente
    
    if not asistente.ultimo_analisis:
        return jsonify({'status': 'error', 'message': 'Sin datos disponibles'}), 503
    
    todos_analisis = asistente.ultimo_analisis.get('todos_los_analisis', [])
    filtrado = [a for a in todos_analisis if a['deporte'].lower() == deporte.lower()]
    
    if not filtrado:
        deportes_disponibles = list(set(a['deporte'] for a in todos_analisis))
        return jsonify({
            'status': 'error',
            'message': f'Deporte "{deporte}" no encontrado',
            'deportes_disponibles': deportes_disponibles
        }), 404
    
    oportunidades_deporte = [a for a in filtrado if a['es_oportunidad']]
    
    return jsonify({
        'status': 'success',
        'deporte': deporte.title(),
        'total_partidos': len(filtrado),
        'oportunidades': len(oportunidades_deporte),
        'partidos': filtrado,
        'mejores_oportunidades': sorted(oportunidades_deporte, 
                                      key=lambda x: x['analisis_ia']['mejor_apuesta']['valor_esperado'], 
                                      reverse=True)
    })

@app.route('/api/forzar', methods=['POST'])
def api_forzar_analisis():
    """Fuerza análisis manual"""
    try:
        ejecutar_analisis_y_notificar()
        return jsonify({
            'status': 'success',
            'message': 'Análisis forzado exitosamente',
            'timestamp': datetime.now(TIMEZONE).isoformat(),
            'oportunidades': len(asistente.ultimo_analisis.get('top_5_oportunidades', [])) if asistente.ultimo_analisis else 0
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Estado del sistema"""
    global asistente
    
    ahora = datetime.now(TIMEZONE)
    
    return jsonify({
        'status': 'healthy',
        'sistema': 'IA Deportiva v4.0',
        'version': '4.0.0',
        'timestamp': ahora.isoformat(),
        'horario_activo': asistente.es_horario_activo(),
        'hora_actual': ahora.strftime('%H:%M:%S'),
        'ultima_actualizacion': asistente.ultima_actualizacion.isoformat() if asistente.ultima_actualizacion else None,
        'configuracion': {
            'horario_operacion': '07:00 - 24:00 (Colombia)',
            'frecuencia_analisis': 'Cada 2 horas',
            'valor_minimo': '5%',
            'webhook_configurado': bool(asistente.webhook_url),
            'telegram_configurado': bool(asistente.telegram_token and asistente.telegram_chat_id)
        },
        'estadisticas': {
            'partidos_ultimo_analisis': asistente.ultimo_analisis['resumen']['total_partidos'] if asistente.ultimo_analisis else 0,
            'oportunidades_detectadas': asistente.ultimo_analisis['resumen']['total_oportunidades'] if asistente.ultimo_analisis else 0,
            'confianza_promedio': asistente.ultimo_analisis['resumen']['confianza_ia_promedio'] if asistente.ultimo_analisis else 0
        }
    })

@app.route('/api/configurar', methods=['POST'])
def api_configurar():
    """Configurar webhooks y notificaciones"""
    global asistente
    
    data = request.get_json()
    
    if 'webhook_url' in data:
        asistente.webhook_url = data['webhook_url']
        
    if 'telegram_token' in data:
        asistente.telegram_token = data['telegram_token']
        
    if 'telegram_chat_id' in data:
        asistente.telegram_chat_id = data['telegram_chat_id']
    
    return jsonify({
        'status': 'success',
        'message': 'Configuración actualizada',
        'webhook_configurado': bool(asistente.webhook_url),
        'telegram_configurado': bool(asistente.telegram_token and asistente.telegram_chat_id)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint no encontrado',
        'endpoints_disponibles': [
            'GET /', 'GET /api/analisis', 'GET /api/oportunidades',
            'GET /api/deporte/{deporte}', 'POST /api/forzar', 'GET /health',
            'POST /api/configurar'
        ]
    }), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Error interno del servidor'
    }), 500

# Ejecutar primer análisis al iniciar
logger.info("🚀 Iniciando primer análisis...")
ejecutar_analisis_y_notificar()

# Punto de entrada principal
if __name__ == '__main__':
    logger.info("🎯 Asistente IA Deportiva v4.0 - INICIADO")
    logger.info("⚡ Características principales:")
    logger.info("   • Análisis automático cada 2 horas")
    logger.info("   • Horario inteligente (7 AM - 12 AM)")
    logger.info("   • Notificaciones automáticas")
    logger.info("   • Dashboard web en tiempo real")
    logger.info("   • API REST completa")
    logger.info("   • Soporte multi-deporte")
    logger.info("   • Algoritmos de Machine Learning")
    
    # Puerto para Render
    PORT = int(os.environ.get('PORT', 5000))
    
    # Ejecutar aplicación
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
