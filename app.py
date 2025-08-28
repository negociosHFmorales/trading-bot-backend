from flask import Flask, jsonify, request
import requests
import os
import logging
from datetime import datetime
import time
import telegram
from telegram import Bot
import asyncio
import threading

app = Flask(__name__)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables de entorno
ODDS_API_KEY = '187f072e0b43d7193c8e2c63fc612e9a'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

class OddsAPIHandler:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.active_sports = ['basketball_nba', 'americanfootball_nfl', 'icehockey_nhl']
        
    def get_active_sports(self):
        """Obtener deportes activos en diciembre 2024"""
        try:
            response = requests.get(
                f"{self.base_url}/sports",
                params={'apiKey': self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                sports = response.json()
                active = [s for s in sports if s['key'] in self.active_sports and s.get('active', False)]
                return {'success': True, 'sports': active}
            else:
                return {'success': False, 'error': f'API Error: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error getting sports: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_odds_with_fallback(self, sport_key):
        """Obtener cuotas con manejo robusto de errores"""
        regions = ['us', 'eu', 'uk']  # Regiones de fallback
        markets = ['h2h', 'spreads', 'totals']
        
        for region in regions:
            try:
                response = requests.get(
                    f"{self.base_url}/sports/{sport_key}/odds",
                    params={
                        'apiKey': self.api_key,
                        'regions': region,
                        'markets': ','.join(markets),
                        'oddsFormat': 'american'
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Manejar respuestas vacías
                    if not data:
                        logger.info(f"Empty response for {sport_key} in region {region}")
                        continue
                        
                    return {
                        'success': True,
                        'data': data,
                        'sport': sport_key,
                        'region': region,
                        'requests_remaining': response.headers.get('x-requests-remaining', 'Unknown'),
                        'count': len(data)
                    }
                elif response.status_code == 422:
                    logger.error(f"Invalid sport key: {sport_key}")
                    return {'success': False, 'error': f'Sport {sport_key} not available'}
                elif response.status_code == 429:
                    logger.warning(f"Rate limit hit for {sport_key}")
                    time.sleep(2)  # Wait and try next region
                    continue
                else:
                    logger.error(f"API error {response.status_code} for {sport_key}")
                    continue
                    
            except requests.exceptions.Timeout:
                logger.error(f"Timeout for {sport_key} in region {region}")
                continue
            except Exception as e:
                logger.error(f"Error fetching {sport_key}: {e}")
                continue
        
        # Si todos los intentos fallan
        return {
            'success': True,  # No es error, simplemente no hay datos
            'data': [],
            'message': f'No games available for {sport_key}',
            'empty': True
        }
    
    def format_game_data(self, games_data, sport_name):
        """Formatear datos para Telegram"""
        if not games_data or len(games_data) == 0:
            return f"❌ No hay partidos de {sport_name} programados"
            
        formatted_games = []
        sport_emoji = {'NBA': '🏀', 'NFL': '🏈', 'NHL': '🏒'}
        emoji = sport_emoji.get(sport_name, '⚽')
        
        for game in games_data[:3]:  # Límite de 3 juegos por mensaje
            home = game['home_team']
            away = game['away_team']
            game_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
            time_str = game_time.strftime('%d/%m %H:%M')
            
            message = f"{emoji} **{away} @ {home}**\n⏰ {time_str}\n"
            
            if game.get('bookmakers'):
                bookmaker = game['bookmakers'][0]
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'h2h':
                        message += "💰 **Línea de dinero:**\n"
                        for outcome in market['outcomes']:
                            price = outcome['price']
                            sign = '+' if price > 0 else ''
                            message += f"• {outcome['name']}: {sign}{price}\n"
                    elif market['key'] == 'spreads' and len(market['outcomes']) >= 2:
                        message += "📊 **Hándicap:**\n"
                        for outcome in market['outcomes']:
                            point = outcome.get('point', 0)
                            price = outcome['price']
                            sign = '+' if price > 0 else ''
                            point_sign = '+' if point > 0 else ''
                            message += f"• {outcome['name']} {point_sign}{point}: {sign}{price}\n"
                        
            formatted_games.append(message)
            
        return '\n\n'.join(formatted_games)

# Inicializar handler
odds_handler = OddsAPIHandler(ODDS_API_KEY)

@app.route('/health')
def health():
    """Health check para Render"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'api_key_configured': bool(ODDS_API_KEY),
        'telegram_configured': bool(TELEGRAM_BOT_TOKEN)
    })

@app.route('/check-sports')
def check_active_sports():
    """Verificar qué deportes están activos HOY"""
    result = odds_handler.get_active_sports()
    
    if result['success']:
        return jsonify({
            'success': True,
            'active_sports': result['sports'],
            'count': len(result['sports']),
            'recommended': ['basketball_nba', 'americanfootball_nfl', 'icehockey_nhl']
        })
    else:
        return jsonify(result), 500

@app.route('/odds/<sport_key>')
def get_odds(sport_key):
    """Obtener cuotas con manejo completo de errores"""
    try:
        result = odds_handler.get_odds_with_fallback(sport_key)
        
        if result['success'] and not result.get('empty'):
            sport_names = {
                'basketball_nba': 'NBA',
                'americanfootball_nfl': 'NFL',
                'icehockey_nhl': 'NHL'
            }
            sport_name = sport_names.get(sport_key, sport_key.upper())
            formatted_data = odds_handler.format_game_data(result['data'], sport_name)
            
            return jsonify({
                'success': True,
                'formatted_message': formatted_data,
                'raw_data': result['data'],
                'count': len(result['data']),
                'sport': sport_name,
                'requests_remaining': result.get('requests_remaining')
            })
        else:
            return jsonify({
                'success': True,
                'message': result.get('message', 'No games available'),
                'data': [],
                'empty': True
            })
            
    except Exception as e:
        logger.error(f"Error in get_odds for {sport_key}: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/nba')
def get_nba():
    """Endpoint específico NBA"""
    return get_odds('basketball_nba')

@app.route('/nfl')
def get_nfl():
    """Endpoint específico NFL"""  
    return get_odds('americanfootball_nfl')

@app.route('/send-update')
def send_betting_update():
    """Enviar actualización completa a Telegram"""
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return jsonify({'error': 'Telegram not configured'}), 400
            
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        
        # Obtener datos de múltiples deportes
        sports_data = []
        for sport in ['basketball_nba', 'americanfootball_nfl', 'icehockey_nhl']:
            result = odds_handler.get_odds_with_fallback(sport)
            if result['success'] and not result.get('empty'):
                sport_names = {'basketball_nba': 'NBA', 'americanfootball_nfl': 'NFL', 'icehockey_nhl': 'NHL'}
                formatted = odds_handler.format_game_data(result['data'], sport_names[sport])
                sports_data.append(formatted)
        
        if sports_data:
            message = "🎯 **ACTUALIZACIÓN APUESTAS DEPORTIVAS**\n\n" + "\n\n---\n\n".join(sports_data)
            message += f"\n\n📈 Datos actualizados: {datetime.now().strftime('%H:%M')}"
            
            # Enviar mensaje (función síncrona)
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
            
            return jsonify({
                'success': True,
                'message': 'Update sent to Telegram',
                'sports_count': len(sports_data)
            })
        else:
            # Enviar mensaje de no disponibilidad
            no_games_msg = "❌ **NO HAY JUEGOS DISPONIBLES**\n\nMotivos posibles:\n• Deportes en pausa\n• Entre jornadas\n• Mantenimiento casas de apuestas\n\n⏰ Próxima verificación en 2 horas"
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=no_games_msg, parse_mode='Markdown')
            
            return jsonify({
                'success': True,
                'message': 'No games message sent',
                'sports_available': False
            })
            
    except Exception as e:
        logger.error(f"Error sending update: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook para N8N u otros servicios"""
    try:
        data = request.get_json()
        logger.info(f"Webhook received: {data}")
        
        # Procesar webhook y responder
        return jsonify({'success': True, 'received': True})
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
