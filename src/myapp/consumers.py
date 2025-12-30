import json
import asyncio
import logging
import time
import urllib.parse
import re
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.tokens import UntypedToken
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from myapp.models import Users, SeriesConfig, WorkerShard
from myapp.services.influx_manager import InfluxManager
from datetime import datetime
from django.utils import timezone 
from typing import Dict, List
from channels.layers import get_channel_layer
from django.core.cache import cache
from asgiref.sync import sync_to_async

from myapp.services.content.integrator_service import get_integrator_service
from myapp.services.mongo_manager import get_mongo_manager

logger = logging.getLogger(__name__)

# Initialize global variable at module level
_influx_manager = None

def get_influx_manager():
    """Get or create global InfluxManager instance"""
    global _influx_manager
    if _influx_manager is None:
        _influx_manager = InfluxManager()
    return _influx_manager

def sanitize_group_name(name):
    """Sanitize group name to contain only valid characters"""
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\-_.]', '_', name)
    # Ensure length is under 100 characters
    if len(sanitized) > 99:
        sanitized = sanitized[:99]
    return sanitized


class PriceConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer with correct closed-candle logic and error handling"""
    
    async def connect(self):
        # Parse URL parameters
        url_kwargs = self.scope['url_route']['kwargs']
        self.exchange = url_kwargs.get('exchange', '').lower()
        self.symbol = url_kwargs.get('symbol', '').upper()
        self.timeframe = url_kwargs.get('timeframe', '1h')
        
        # Validate parameters
        if not self.exchange or not self.symbol:
            await self.close(code=4000)
            return
        
        # Sanitize group name to avoid invalid characters
        group_name_raw = f'price_{self.exchange}_{self.symbol}_{self.timeframe}'
        self.room_group_name = sanitize_group_name(group_name_raw)
        
        # Accept connection BEFORE doing any async operations that might send messages
        await self.accept()
        
        # Check if series is configured
        try:
            @database_sync_to_async
            def get_series_config():
                # Try to find the series config by matching exchange name and symbol
                from myapp.models import Exchange
                try:
                    # First, try to find the exchange
                    exchange = Exchange.objects.filter(name__iexact=self.exchange, isactive=1, isdeleted=0).first()
                    if not exchange:
                        return None
                    
                    # Then find the series config
                    return SeriesConfig.objects.filter(
                        exchange=exchange,
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        isactive=1,
                        isdeleted=0
                    ).first()
                except Exception as e:
                    logger.error(f"Error finding series config: {e}")
                    return None
            
            self.series_config = await get_series_config()
            
            if not self.series_config:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Series not configured: {self.exchange}/{self.symbol}/{self.timeframe}'
                }))
                await self.close(code=4004)
                return
            
        except Exception as e:
            logger.error(f"Error checking series config: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Configuration error: {str(e)}'
            }))
            await self.close(code=4005)
            return
        
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        # Send connection confirmation with series info
        await self.send(text_data=json.dumps({
            'type': 'connection',
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'series_config': {
                'backfill_completed': self.series_config.backfill_completed,
                'last_ts_ms': self.series_config.last_ts_ms,
                'earliest_start_ms': self.series_config.earliest_start_ms,
            },
            'status': 'connected'
        }))
        
        # Start data feed
        self.data_task = asyncio.create_task(self.data_feed())
    
    async def disconnect(self, close_code):
        # Cancel data feed
        if hasattr(self, 'data_task'):
            self.data_task.cancel()
        
        # Leave room group
        if hasattr(self, 'room_group_name'):
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )

    async def data_feed(self):
        """data feed with correct closed-candle logic and better error handling"""
        try:
            # Use global InfluxManager instance
            influx_manager = get_influx_manager()
            
            if not influx_manager.client:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'InfluxDB not available'
                }))
                return
            
            # Send initial historical data
            await self.send_historical_data(influx_manager)
            
            # Start real-time feed with logic
            error_count = 0
            max_errors = 10
            
            while error_count < max_errors:
                try:
                    # Get latest data from InfluxDB with closed-candle filtering
                    await self.send_latest_closed_candles(influx_manager)
                    
                    # Reset error count on success
                    error_count = 0
                    
                    # Dynamic sleep interval based on timeframe
                    sleep_seconds = self.get_update_interval()
                    await asyncio.sleep(sleep_seconds)
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error in data feed: {e}")
                    
                    # Exponential backoff on errors
                    backoff_seconds = min(2 ** error_count, 300)  # Cap at 5 minutes
                    await asyncio.sleep(backoff_seconds)
            
            # Max errors reached
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Data feed terminated due to repeated errors'
            }))
                    
        except asyncio.CancelledError:
            logger.info(f"data feed cancelled for {self.symbol}")
        except Exception as e:
            logger.error(f"Fatal error in data feed: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Data feed error: {str(e)}'
            }))
    
    async def send_historical_data(self, influx_manager):
        """Send recent historical data"""
        try:
            import pandas as pd
            from concurrent.futures import ThreadPoolExecutor
            
            def get_historical():
                query = f'''
                    from(bucket: "{influx_manager.bucket}")
                    |> range(start: -24h)
                    |> filter(fn: (r) => r._measurement == "ohlcv")
                    |> filter(fn: (r) => r.exchange == "{self.exchange}")
                    |> filter(fn: (r) => r.symbol == "{self.symbol}")
                    |> filter(fn: (r) => r.timeframe == "{self.timeframe}")
                    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                    |> sort(columns: ["_time"])
                    |> limit(n: 1000)
                '''
                
                result = influx_manager.query_api.query_data_frame(query, org=influx_manager.org)
                return result
            
            # Run query in executor
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                df = await loop.run_in_executor(executor, get_historical)
            
            if not df.empty:
                # Convert to list of candles
                candles = []
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                
                if all(col in df.columns for col in required_columns):
                    for _, row in df.iterrows():
                        candle = {
                            'timestamp': int(pd.to_datetime(row['_time']).timestamp() * 1000),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']),
                        }
                        candles.append(candle)
                
                await self.send(text_data=json.dumps({
                    'type': 'historical_data',
                    'symbol': self.symbol,
                    'timeframe': self.timeframe,
                    'data': candles[-100:],  # Last 100 candles
                    'count': len(candles)
                }))
            
        except Exception as e:
            logger.error(f"Error sending historical data: {e}")
    
    async def send_latest_closed_candles(self, influx_manager):
        """Send only properly closed candles with correct timing logic"""
        try:
            import pandas as pd
            from concurrent.futures import ThreadPoolExecutor
            
            # Calculate proper closed-candle threshold
            current_time_ms = int(time.time() * 1000)
            timeframe_ms = self._timeframe_to_ms(self.timeframe)
            
            # A candle is closed if: current_time >= candle_open_time + timeframe_duration
            # Query recent candles and filter client-side for proper closed-candle logic
            def get_recent_candles():
                # Get candles from last 2 timeframe periods to ensure we catch newly closed ones
                lookback_hours = max(2 * (timeframe_ms / (1000 * 60 * 60)), 1)  # At least 1 hour
                
                query = f'''
                    from(bucket: "{influx_manager.bucket}")
                    |> range(start: -{int(lookback_hours)}h)
                    |> filter(fn: (r) => r._measurement == "ohlcv")
                    |> filter(fn: (r) => r.exchange == "{self.exchange}")
                    |> filter(fn: (r) => r.symbol == "{self.symbol}")
                    |> filter(fn: (r) => r.timeframe == "{self.timeframe}")
                    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                    |> sort(columns: ["_time"], desc: true)
                    |> limit(n: 10)
                '''
                
                result = influx_manager.query_api.query_data_frame(query, org=influx_manager.org)
                return result
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                df = await loop.run_in_executor(executor, get_recent_candles)
            
            if not df.empty and all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                closed_candles = []
                
                for _, row in df.iterrows():
                    candle_open_ms = int(pd.to_datetime(row['_time']).timestamp() * 1000)
                    candle_close_time_ms = candle_open_ms + timeframe_ms
                    
                    # Only include candles that are actually closed
                    if current_time_ms >= candle_close_time_ms:
                        candle = {
                            'timestamp': candle_open_ms,
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']),
                            'is_closed': True,
                            'close_time': candle_close_time_ms,
                        }
                        closed_candles.append(candle)
                    else:
                        # This is a forming candle - we can send it with a flag but mark it as forming
                        remaining_ms = candle_close_time_ms - current_time_ms
                        forming_candle = {
                            'timestamp': candle_open_ms,
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']),
                            'is_closed': False,
                            'is_forming': True,
                            'remaining_ms': remaining_ms,
                            'close_time': candle_close_time_ms,
                        }
                        # Only send the most recent forming candle
                        if not closed_candles:
                            closed_candles.append(forming_candle)
                        break
                
                if closed_candles:
                    # Sort by timestamp (most recent first for this display)
                    closed_candles.sort(key=lambda x: x['timestamp'], reverse=True)
                    
                    await self.send(text_data=json.dumps({
                        'type': 'price_update',
                        'exchange': self.exchange,
                        'symbol': self.symbol,
                        'timeframe': self.timeframe,
                        'data': closed_candles[0],  # Send most recent
                        'all_recent': closed_candles[:5],  # Include last 5 for context
                        'series_status': {
                            'backfill_completed': self.series_config.backfill_completed,
                            'last_update': current_time_ms,
                        }
                    }))
            
        except Exception as e:
            logger.error(f"Error sending latest closed candles: {e}")

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000,
        }
        return timeframe_map.get(timeframe, 60 * 60 * 1000)  # Default to 1h
    
    def get_update_interval(self) -> int:
        """Get update interval based on timeframe with more intelligent scaling"""
        intervals = {
            '1m': 5,    # 5 seconds for 1-minute candles
            '5m': 15,   # 15 seconds for 5-minute candles
            '15m': 30,  # 30 seconds for 15-minute candles
            '30m': 60,  # 1 minute for 30-minute candles
            '1h': 120,  # 2 minutes for 1-hour candles
            '2h': 240,  # 4 minutes for 2-hour candles
            '4h': 300,  # 5 minutes for 4-hour candles
            '6h': 600,  # 10 minutes for 6-hour candles
            '12h': 900, # 15 minutes for 12-hour candles
            '1d': 1800, # 30 minutes for daily candles
            '3d': 3600, # 1 hour for 3-day candles
            '1w': 7200, # 2 hours for weekly candles
            '1M': 14400,# 4 hours for monthly candles
        }
        return intervals.get(self.timeframe, 60)  # Default 1 minute
    
    async def receive(self, text_data):
        """Handle client messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'ping':
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': data.get('timestamp'),
                    'server_time': int(time.time() * 1000)
                }))
            
            elif message_type == 'get_series_status':
                await self.send_series_status()
            
            elif message_type == 'request_historical':
                timeframe = data.get('timeframe', self.timeframe)
                limit = min(data.get('limit', 100), 1000)
                await self.send_requested_historical(timeframe, limit)
                
            elif message_type == 'set_update_frequency':
                # Allow client to request faster/slower updates within limits
                requested_interval = data.get('interval_seconds', self.get_update_interval())
                min_interval = max(1, self.get_update_interval() // 4)  # No faster than 1/4 default
                max_interval = self.get_update_interval() * 4  # No slower than 4x default
                
                self.custom_update_interval = max(min_interval, min(requested_interval, max_interval))
                
                await self.send(text_data=json.dumps({
                    'type': 'update_frequency_set',
                    'interval_seconds': self.custom_update_interval
                }))
                
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
    
    async def send_requested_historical(self, timeframe: str, limit: int):
        """Send requested historical data with specific parameters"""
        try:
            # Use global instance instead of creating new one
            influx_manager = get_influx_manager()
            if not influx_manager.client:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'InfluxDB not available for historical data'
                }))
                return
            
            import pandas as pd
            from concurrent.futures import ThreadPoolExecutor
            
            def get_requested_historical():
                # Calculate appropriate time range based on limit and timeframe
                timeframe_ms = self._timeframe_to_ms(timeframe)
                lookback_ms = limit * timeframe_ms
                lookback_hours = max(lookback_ms / (1000 * 60 * 60), 1)
                
                query = f'''
                    from(bucket: "{influx_manager.bucket}")
                    |> range(start: -{int(lookback_hours)}h)
                    |> filter(fn: (r) => r._measurement == "ohlcv")
                    |> filter(fn: (r) => r.exchange == "{self.exchange}")
                    |> filter(fn: (r) => r.symbol == "{self.symbol}")
                    |> filter(fn: (r) => r.timeframe == "{timeframe}")
                    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                    |> sort(columns: ["_time"])
                    |> limit(n: {limit})
                '''
                
                result = influx_manager.query_api.query_data_frame(query, org=influx_manager.org)
                return result
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                df = await loop.run_in_executor(executor, get_requested_historical)
            
            if not df.empty:
                candles = []
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                
                if all(col in df.columns for col in required_columns):
                    for _, row in df.iterrows():
                        candle = {
                            'timestamp': int(pd.to_datetime(row['_time']).timestamp() * 1000),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']),
                        }
                        candles.append(candle)
                
                await self.send(text_data=json.dumps({
                    'type': 'requested_historical_data',
                    'symbol': self.symbol,
                    'timeframe': timeframe,
                    'requested_limit': limit,
                    'actual_count': len(candles),
                    'data': candles
                }))
            else:
                await self.send(text_data=json.dumps({
                    'type': 'requested_historical_data',
                    'symbol': self.symbol,
                    'timeframe': timeframe,
                    'requested_limit': limit,
                    'actual_count': 0,
                    'data': [],
                    'message': 'No historical data available for the requested parameters'
                }))
            
        except Exception as e:
            logger.error(f"Error sending requested historical data: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to retrieve historical data: {str(e)}'
            }))
    
    async def send_series_status(self):
        """Send current series status"""
        @database_sync_to_async
        def refresh_config():
            self.series_config.refresh_from_db()
            return {
                'exchange': self.series_config.exchange.name,
                'symbol': self.series_config.symbol,
                'timeframe': self.series_config.timeframe,
                'backfill_completed': self.series_config.backfill_completed,
                'probing_completed': self.series_config.probing_completed,
                'last_ts_ms': self.series_config.last_ts_ms,
                'last_backfill_ms': self.series_config.last_backfill_ms,
                'earliest_start_ms': self.series_config.earliest_start_ms,
                'updated_at': self.series_config.updated_at.isoformat(),
            }
        
        try:
            status = await refresh_config()
            
            await self.send(text_data=json.dumps({
                'type': 'series_status',
                'status': status
            }))
            
        except Exception as e:
            logger.error(f"Error sending series status: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to get series status: {str(e)}'
            }))



class SeriesStatusConsumer(AsyncWebsocketConsumer):
    """series status consumer with better monitoring"""
    async def connect(self):
        url_kwargs = self.scope['url_route']['kwargs']
        self.exchange = url_kwargs.get('exchange', '').lower()
        self.symbol = url_kwargs.get('symbol', '').upper()
        self.timeframe = url_kwargs.get('timeframe', '1h')
        
        # Sanitize group name to avoid invalid characters
        group_name_raw = f'series_status_{self.exchange}_{self.symbol}_{self.timeframe}'
        self.room_group_name = sanitize_group_name(group_name_raw)
        
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Send initial status
        await self.send_series_status()
        
        # Start status monitoring
        self.status_task = asyncio.create_task(self.monitor_series_status())
    
    async def disconnect(self, close_code):
        if hasattr(self, 'status_task'):
            self.status_task.cancel()
        
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
    
    async def monitor_series_status(self):
        """series status monitoring with adaptive intervals"""
        try:
            base_interval = 30  # Base 30 seconds
            error_count = 0
            max_errors = 5
            
            while error_count < max_errors:
                try:
                    await asyncio.sleep(base_interval)
                    await self.send_series_status()
                    error_count = 0  # Reset on success
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error in status monitoring: {e}")
                    
                    # Exponential backoff on errors
                    backoff_interval = min(base_interval * (2 ** error_count), 300)
                    await asyncio.sleep(backoff_interval)
                    
        except asyncio.CancelledError:
            logger.info(f"series status monitoring cancelled for {self.symbol}")
    
    async def send_series_status(self):
        """series status with proper async database access and relationship handling"""
        @database_sync_to_async
        def get_series_config_data():
            try:
                from myapp.models import Exchange
                exchange = Exchange.objects.filter(name__iexact=self.exchange, isactive=1, isdeleted=0).first()
                if not exchange:
                    return None
                    
                config = SeriesConfig.objects.select_related('exchange').filter(
                    exchange=exchange,
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    isactive=1,
                    isdeleted=0
                ).first()
                
                if not config:
                    return None
                
                # Extract all needed data within the sync context
                return {
                    'exchange_name': config.exchange.name,
                    'symbol': config.symbol,
                    'timeframe': config.timeframe,
                    'series_key': config.series_key,
                    'backfill_completed': config.backfill_completed,
                    'probing_completed': config.probing_completed,
                    'last_ts_ms': config.last_ts_ms,
                    'last_backfill_ms': config.last_backfill_ms,
                    'earliest_start_ms': config.earliest_start_ms,
                    'listing_date_ms': getattr(config, 'listing_date_ms', None),
                    'probe_attempts': config.probe_attempts,
                    'last_error': config.last_error,
                    'updated_at': config.updated_at.isoformat(),
                }
            except Exception as e:
                logger.error(f"Error getting series config: {e}")
                return None
        
        try:
            config_data = await get_series_config_data()
            
            if config_data:
                # Calculate metrics using the extracted data
                current_time_ms = int(time.time() * 1000)
                timeframe_ms = self._timeframe_to_ms(config_data['timeframe'])
                
                # Data age calculation
                last_ts_ms = config_data['last_ts_ms'] or 0
                data_age_ms = current_time_ms - last_ts_ms
                data_age_hours = data_age_ms / (1000 * 60 * 60)
                
                # Staleness detection with timeframe-aware thresholds
                stale_threshold_multiplier = 3  # 3x timeframe is considered stale
                stale_threshold_ms = timeframe_ms * stale_threshold_multiplier
                is_stale = data_age_ms > stale_threshold_ms
                
                # Health status calculation
                if not config_data['backfill_completed']:
                    health_status = 'backfilling'
                elif is_stale:
                    health_status = 'stale'
                elif config_data['last_error']:
                    health_status = 'error'
                else:
                    health_status = 'healthy'
                
                # Expected next candle time
                if last_ts_ms:
                    next_candle_time_ms = last_ts_ms + timeframe_ms
                    next_candle_in_ms = next_candle_time_ms - current_time_ms
                else:
                    next_candle_time_ms = None
                    next_candle_in_ms = None
                
                status_data = {
                    'exchange': config_data['exchange_name'],
                    'symbol': config_data['symbol'],
                    'timeframe': config_data['timeframe'],
                    'series_key': config_data['series_key'],
                    'backfill_completed': config_data['backfill_completed'],
                    'probing_completed': config_data['probing_completed'],
                    'last_ts_ms': config_data['last_ts_ms'],
                    'last_backfill_ms': config_data['last_backfill_ms'],
                    'earliest_start_ms': config_data['earliest_start_ms'],
                    'listing_date_ms': config_data['listing_date_ms'],
                    'probe_attempts': config_data['probe_attempts'],
                    'last_error': config_data['last_error'],
                    'updated_at': config_data['updated_at'],
                    'data_age_ms': data_age_ms,
                    'data_age_hours': data_age_hours,
                    'is_stale': is_stale,
                    'health_status': health_status,
                    'timeframe_ms': timeframe_ms,
                    'stale_threshold_ms': stale_threshold_ms,
                    'next_candle_time_ms': next_candle_time_ms,
                    'next_candle_in_ms': next_candle_in_ms,
                }
                
                await self.send(text_data=json.dumps({
                    'type': 'series_status',
                    'status': status_data,
                    'timestamp': current_time_ms
                }))
            else:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Series not found: {self.exchange}/{self.symbol}/{self.timeframe}'
                }))
                
        except Exception as e:
            logger.error(f"Error sending series status: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to get series status: {str(e)}'
            }))
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe to milliseconds"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000,
        }
        return timeframe_map.get(timeframe, 60 * 60 * 1000)
    
    async def receive(self, text_data):
        """Handle client messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'refresh_status':
                await self.send_series_status()
            
            elif message_type == 'trigger_backfill':
                await self.trigger_series_backfill()
            
            elif message_type == 'reset_series':
                await self.reset_series_config()
                
            elif message_type == 'force_probe':
                await self.force_series_probe()
                
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
    
    async def trigger_series_backfill(self):
        """Trigger backfill for this series with proper async database access"""
        @database_sync_to_async
        def reset_backfill():
            from myapp.models import Exchange
            exchange = Exchange.objects.filter(name__iexact=self.exchange, isactive=1, isdeleted=0).first()
            if not exchange:
                raise Exception(f"Exchange {self.exchange} not found")
                
            config = SeriesConfig.objects.filter(
                exchange=exchange,
                symbol=self.symbol,
                timeframe=self.timeframe
            ).first()
            
            if not config:
                raise Exception(f"Series config not found for {self.exchange}/{self.symbol}/{self.timeframe}")
            
            config.backfill_completed = False
            config.last_error = None
            config.save()
            return config.series_key
        
        try:
            from myapp.tasks.series_tasks import run_series_orchestrator
            
            series_key = await reset_backfill()
            
            # Trigger async task
            run_series_orchestrator.delay(
                worker_id=f"manual_{int(time.time())}",
                shard_name="manual_backfill"
            )
            
            await self.send(text_data=json.dumps({
                'type': 'backfill_triggered',
                'series_key': series_key,
                'message': 'Backfill task queued'
            }))
            
        except Exception as e:
            logger.error(f"Error triggering backfill: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to trigger backfill: {str(e)}'
            }))
    
    async def reset_series_config(self):
        """Reset series configuration with proper async database access"""
        @database_sync_to_async
        def reset_config():
            from myapp.models import Exchange
            exchange = Exchange.objects.filter(name__iexact=self.exchange, isactive=1, isdeleted=0).first()
            if not exchange:
                raise Exception(f"Exchange {self.exchange} not found")
                
            config = SeriesConfig.objects.filter(
                exchange=exchange,
                symbol=self.symbol,
                timeframe=self.timeframe
            ).first()
            
            if not config:
                raise Exception(f"Series config not found for {self.exchange}/{self.symbol}/{self.timeframe}")
            
            config.backfill_completed = False
            config.probing_completed = False
            config.last_ts_ms = None
            config.last_backfill_ms = None
            config.earliest_start_ms = None
            config.probe_attempts = 0
            config.last_error = None
            config.save()
            return config.series_key
        
        try:
            series_key = await reset_config()
            
            await self.send(text_data=json.dumps({
                'type': 'series_reset',
                'message': 'Series configuration reset successfully',
                'series_key': series_key
            }))
            
        except Exception as e:
            logger.error(f"Error resetting series: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error', 
                'message': f'Failed to reset series: {str(e)}'
            }))
    
    async def force_series_probe(self):
        """Force re-probe series earliest start with proper async database access"""
        @database_sync_to_async
        def reset_probe():
            from myapp.models import Exchange
            exchange = Exchange.objects.filter(name__iexact=self.exchange, isactive=1, isdeleted=0).first()
            if not exchange:
                raise Exception(f"Exchange {self.exchange} not found")
                
            config = SeriesConfig.objects.filter(
                exchange=exchange,
                symbol=self.symbol,
                timeframe=self.timeframe
            ).first()
            
            if not config:
                raise Exception(f"Series config not found for {self.exchange}/{self.symbol}/{self.timeframe}")
            
            config.probing_completed = False
            config.earliest_start_ms = None
            config.probe_attempts = 0
            config.save()
            return config.series_key
        
        try:
            series_key = await reset_probe()
            
            await self.send(text_data=json.dumps({
                'type': 'probe_reset',
                'series_key': series_key,
                'message': 'Series probing reset - will re-probe on next backfill'
            }))
            
        except Exception as e:
            logger.error(f"Error forcing probe reset: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to reset probe: {str(e)}'
            }))
 

class OrchestratorMonitorConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for orchestrator monitoring"""
    
    async def connect(self):
        url_kwargs = self.scope['url_route']['kwargs']
        self.worker_id = url_kwargs.get('worker_id', 'all')
        
        # Sanitize group name to avoid invalid characters
        group_name_raw = f'orchestrator_monitor_{self.worker_id}'
        self.room_group_name = sanitize_group_name(group_name_raw)
        
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Send initial status
        await self.send_orchestrator_status()
        
        # Start monitoring
        self.monitor_task = asyncio.create_task(self.monitor_orchestrator())
    
    async def disconnect(self, close_code):
        if hasattr(self, 'monitor_task'):
            self.monitor_task.cancel()
        
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
    
    async def monitor_orchestrator(self):
        """Monitor orchestrator status and metrics"""
        try:
            while True:
                await asyncio.sleep(15)  # Check every 15 seconds
                await self.send_orchestrator_status()
                await self.send_system_metrics()
        except asyncio.CancelledError:
            logger.info(f"Orchestrator monitoring cancelled for {self.worker_id}")
    
    async def send_orchestrator_status(self):
        """Send orchestrator and worker status"""
        @database_sync_to_async
        def get_worker_stats():
            from django.utils import timezone
            
            workers = WorkerShard.objects.filter(isactive=1, isdeleted=0)
            if self.worker_id != 'all':
                workers = workers.filter(worker_id=self.worker_id)
            
            worker_stats = []
            for worker in workers:
                # Calculate worker health
                now = timezone.now()
                heartbeat_age = (now - worker.last_heartbeat).total_seconds() if worker.last_heartbeat else float('inf')
                is_healthy = heartbeat_age < 300  # 5 minutes
                
                # Get series stats for this worker
                assigned_series_keys = worker.assigned_series
                if assigned_series_keys:
                    # Parse series keys to get actual configs
                    series_stats = {
                        'total': len(assigned_series_keys),
                        'backfill_completed': 0,
                        'streaming': 0,
                        'errors': 0
                    }
                    
                    # Get relevant configs
                    all_configs = SeriesConfig.objects.filter(isactive=1, isdeleted=0)
                    for config in all_configs:
                        if config.series_key in assigned_series_keys:
                            if config.backfill_completed:
                                series_stats['backfill_completed'] += 1
                                series_stats['streaming'] += 1
                            if config.last_error:
                                series_stats['errors'] += 1
                else:
                    series_stats = {'total': 0, 'backfill_completed': 0, 'streaming': 0, 'errors': 0}
                
                worker_stats.append({
                    'worker_id': worker.worker_id,
                    'shard_name': worker.shard_name,
                    'status': worker.status,
                    'is_healthy': is_healthy,
                    'heartbeat_age_seconds': int(heartbeat_age),
                    'series_count': worker.series_count,
                    'backfill_progress': worker.backfill_progress,
                    'series_stats': series_stats,
                    'last_heartbeat': worker.last_heartbeat.isoformat() if worker.last_heartbeat else None
                })
            
            return worker_stats
        
        try:
            worker_stats = await get_worker_stats()
            
            # Calculate overall system stats
            total_workers = len(worker_stats)
            healthy_workers = len([w for w in worker_stats if w['is_healthy']])
            total_series = sum(w['series_count'] for w in worker_stats)
            completed_series = sum(w['series_stats']['backfill_completed'] for w in worker_stats)
            error_series = sum(w['series_stats']['errors'] for w in worker_stats)
            
            system_health = 'healthy' if healthy_workers == total_workers and total_workers > 0 else 'degraded'
            
            await self.send(text_data=json.dumps({
                'type': 'orchestrator_status',
                'system_stats': {
                    'total_workers': total_workers,
                    'healthy_workers': healthy_workers,
                    'total_series': total_series,
                    'completed_series': completed_series,
                    'error_series': error_series,
                    'completion_rate': (completed_series / total_series * 100) if total_series > 0 else 0,
                    'system_health': system_health
                },
                'workers': worker_stats,
                'timestamp': int(time.time() * 1000)
            }))
            
        except Exception as e:
            logger.error(f"Error sending orchestrator status: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to get orchestrator status: {str(e)}'
            }))
    
    async def send_system_metrics(self):
        """Send system performance metrics"""
        try:
            import psutil
            from django.db import connection
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Database metrics
            db_connections = len(connection.queries)
            
            # InfluxDB metrics (if available)
            influx_status = 'unknown'
            try:
                influx_manager = get_influx_manager()
                if influx_manager.client:
                    # Test connection
                    await asyncio.get_event_loop().run_in_executor(
                        None, influx_manager.client.ping
                    )
                    influx_status = 'connected'
                else:
                    influx_status = 'disconnected'
            except Exception:
                influx_status = 'error'
            
            metrics = {
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_total_gb': memory.total / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_used_gb': disk.used / (1024**3),
                    'disk_total_gb': disk.total / (1024**3)
                },
                'database': {
                    'connection_count': db_connections
                },
                'influxdb': {
                    'status': influx_status
                }
            }
            
            await self.send(text_data=json.dumps({
                'type': 'system_metrics',
                'metrics': metrics,
                'timestamp': int(time.time() * 1000)
            }))
            
        except Exception as e:
            logger.error(f"Error sending system metrics: {e}")
            # Send minimal metrics on error
            await self.send(text_data=json.dumps({
                'type': 'system_metrics',
                'metrics': {'system': {'cpu_percent': 0, 'memory_percent': 0}},
                'error': str(e),
                'timestamp': int(time.time() * 1000)
            }))
    
    async def receive(self, text_data):
        """Handle client messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'refresh_status':
                await self.send_orchestrator_status()
            
            elif message_type == 'get_worker_details':
                worker_id = data.get('worker_id')
                await self.send_worker_details(worker_id)
            
            elif message_type == 'trigger_health_check':
                await self.trigger_system_health_check()
                
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
    
    async def send_worker_details(self, worker_id: str):
        """Send detailed information about a specific worker"""
        @database_sync_to_async
        def get_worker_details():
            try:
                worker = WorkerShard.objects.get(worker_id=worker_id)
                
                # Get detailed series information
                assigned_keys = worker.assigned_series
                series_details = []
                
                all_configs = SeriesConfig.objects.filter(isactive=1, isdeleted=0)
                for config in all_configs:
                    if config.series_key in assigned_keys:
                        current_time_ms = int(time.time() * 1000)
                        data_age_ms = current_time_ms - (config.last_ts_ms or 0)
                        
                        series_details.append({
                            'series_key': config.series_key,
                            'exchange': config.exchange.name,
                            'symbol': config.symbol,
                            'timeframe': config.timeframe,
                            'backfill_completed': config.backfill_completed,
                            'probing_completed': config.probing_completed,
                            'last_ts_ms': config.last_ts_ms,
                            'data_age_hours': data_age_ms / (1000 * 60 * 60),
                            'last_error': config.last_error,
                            'updated_at': config.updated_at.isoformat()
                        })
                
                return {
                    'worker_id': worker.worker_id,
                    'shard_name': worker.shard_name,
                    'status': worker.status,
                    'series_count': worker.series_count,
                    'backfill_progress': worker.backfill_progress,
                    'last_heartbeat': worker.last_heartbeat.isoformat() if worker.last_heartbeat else None,
                    'series_details': series_details
                }
                
            except WorkerShard.DoesNotExist:
                return None
        
        try:
            details = await get_worker_details()
            
            if details:
                await self.send(text_data=json.dumps({
                    'type': 'worker_details',
                    'details': details
                }))
            else:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Worker {worker_id} not found'
                }))
                
        except Exception as e:
            logger.error(f"Error getting worker details: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to get worker details: {str(e)}'
            }))
    
    async def trigger_system_health_check(self):
        """Trigger a comprehensive system health check"""
        try:
            from myapp.tasks.series_tasks import monitor_series_health
            
            # Trigger health check task
            result = monitor_series_health.delay()
            
            await self.send(text_data=json.dumps({
                'type': 'health_check_triggered',
                'task_id': result.id,
                'message': 'System health check initiated'
            }))
            
        except Exception as e:
            logger.error(f"Error triggering health check: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to trigger health check: {str(e)}'
            }))

