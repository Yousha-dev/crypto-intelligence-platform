# series_orchestrator.py
import asyncio
from logging import config
import ccxt
import ccxt.pro as ccxtpro
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from asgiref.sync import sync_to_async
from django.db import transaction
import time
from datetime import timezone as timezoneDt
import logging
import hashlib
import random
from django.conf import settings
from django.utils import timezone
from myapp.models import Exchange, TradingPair, SeriesConfig, WorkerShard
from myapp.services.influx_manager import InfluxManager
import json

logger = logging.getLogger(__name__)

class SeriesOrchestrator:
    
    def __init__(self, worker_id: str = None, shard_name: str = None, max_series: int = None):
        self.worker_id = worker_id or f"worker_{int(time.time())}"
        self.shard_name = shard_name or "default"
        self.max_series = max_series 
        self.earliest_date_override_ms: Optional[int] = None
        
        # Core components
        self.influx_manager = InfluxManager()
        self.sync_exchanges: Dict[str, ccxt.Exchange] = {}
        
        # ONE WS CLIENT PER (exchange, market_type) - Maintained from original
        self.ws_exchanges: Dict[str, ccxtpro.Exchange] = {}
        self.ws_subscriptions: Dict[str, Set[str]] = {}
        
        # Configuration
        self.supported_timeframes = getattr(settings, 'TRADING_CONFIG', {}).get('SUPPORTED_TIMEFRAMES', ['5m', '1h', '1d'])
        
        # Runtime state
        self.last_ts_cache: Dict[str, int] = {}
        self.series_configs: Dict[str, SeriesConfig] = {}
        self.symbol_mapping_cache: Dict[str, str] = {}
        self.is_running = False
        
        # rate limiting with exchange-specific config
        self.rate_limits: Dict[str, float] = {}
        self.last_request_time: Dict[str, float] = {}
        self.rate_limit_backoff: Dict[str, float] = {}  # For 429 handling
        
        # WebSocket management
        self.ws_tasks: Dict[str, asyncio.Task] = {}
        
        # Phase separation tracking
        self.backfill_phase_complete = False
        self.streaming_phase_active = False
        
        # Gap detection configuration
        self.gap_detection_enabled = getattr(settings, 'TRADING_CONFIG', {}).get('GAP_DETECTION_ENABLED', True)
        self.gap_scan_interval = getattr(settings, 'TRADING_CONFIG', {}).get('GAP_SCAN_INTERVAL', 3600)  # 1 hour default
        self.gap_detection_task: Optional[asyncio.Task] = None
        self.last_gap_scan: Dict[str, int] = {}  # series_key -> last_scan_timestamp
        self.detected_gaps: Dict[str, List[Tuple[int, int]]] = {}  # series_key -> [(start_ms, end_ms), ...]
        
        # Add enhanced listing date detection constants
        self.GLOBAL_LISTING_KEYS = [
            'listingDate', 'onboardDate', 'launchDate', 'listedAt', 'listDate',
            'listingTime', 'launchTime', 'onlineTime', 'onlineDate', 'openTime', 'startTime',
            'startAt', 'start_at', 'goLiveTime', 'go_live_time', 'tradingStartTime',
            'trading_start_time', 'enableTime', 'enable_time',
            'createTime', 'createdTime', 'ctime', 'utime', 'firstOpenTime', 'first_open_time',
            'firstOpenTimestamp', 'first_open_timestamp',
            'statusChangedAt', 'status_changed_at',
        ]
        
        self.EXCHANGE_KEY_HINTS = {
            'binance':        ['onboardDate', 'listingDate', 'launchDate'],
            'binanceusdm':    ['onboardDate', 'listingDate', 'launchDate'],
            'binancecoinm':   ['onboardDate', 'listingDate', 'launchDate'],
            'bybit':          ['launchTime', 'onlineTime', 'listingTime', 'listDate'],
            'bitget':         ['openTime', 'listingTime', 'launchTime', 'onlineTime'],
            'okx':            ['listTime', 'listingTime', 'onlineTime', 'launchTime'],
            'okcoin':         ['listTime', 'listingTime', 'onlineTime', 'launchTime'],
            'kraken':         ['statusChangedAt', 'listingTime', 'onlineTime'],
            'krakenfutures':  ['listingTime', 'onlineTime', 'launchTime'],
            'coinbase':       ['status_changed_at', 'statusChangedAt'],
            'coinbaseexchange':['status_changed_at', 'statusChangedAt'],
            'mexc':           ['createTime', 'listingTime', 'onlineTime', 'launchTime'],
            'kucoin':         ['tradingStartTime', 'listingTime', 'onlineTime', 'launchTime', 'listDate'],
            'kucoinfutures':  ['listingTime', 'onlineTime', 'launchTime'],
            'bitfinex':       ['launchTime', 'listingTime', 'onlineTime'],
            'bitstamp':       ['listingTime', 'onlineTime', 'launchTime'],
        }

    async def initialize(self):
        """Initialize the orchestrator - Step 1: Configuration and setup"""
        logger.info(f"Initializing SeriesOrchestrator {self.worker_id}")
        
        try:
            # Step 1: Configuration and setup
            logger.info("Loading series configuration...")
            await self._load_series_configuration()
            logger.info(f"Loaded {len(self.series_configs)} series configurations")
            
            # Step 2: Resume point discovery
            logger.info("Discovering resume points...")
            await self._discover_resume_points()
            logger.info("Resume points discovered")
            
            # Initialize exchanges with proper symbol normalization
            logger.info("Initializing exchanges...")
            await self._initialize_exchanges_with_normalization()
            logger.info("Exchanges initialized")
            
            logger.info(f"SeriesOrchestrator initialized with {len(self.series_configs)} series")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator at step: {e}", exc_info=True)
            return False

    async def _discover_resume_points(self):
        """Step 2: Resume point discovery"""
        logger.info("Discovering resume points from InfluxDB")
        
        # Get latest timestamps from InfluxDB
        influx_timestamps = await self._run_in_executor(
            self.influx_manager.get_latest_timestamps_per_series
        )
        
        # Update last_ts_cache and series configs
        for series_key, config in self.series_configs.items():
            if series_key in influx_timestamps:
                timestamp_ms = influx_timestamps[series_key]
                self.last_ts_cache[series_key] = timestamp_ms
                
                # Update database if needed
                if config.last_ts_ms != timestamp_ms:
                    config.last_ts_ms = timestamp_ms
                    await self._async_db_save(config)
                    
            else:
                # No data in InfluxDB for this series
                self.last_ts_cache[series_key] = 0
        
        logger.info(f"Resume points discovered for {len(influx_timestamps)} series")


    def _get_listing_keys_for_exchange(self, exchange_name: str) -> List[str]:
        """Get prioritized listing keys for specific exchange"""
        exchange_name_lower = exchange_name.lower()
        preferred = self.EXCHANGE_KEY_HINTS.get(exchange_name_lower, [])
        seen, keys = set(), []
        
        # Add preferred keys first, then global keys
        for k in preferred + self.GLOBAL_LISTING_KEYS:
            if k not in seen:
                seen.add(k)
                keys.append(k)
        
        return keys
    
    def _normalize_timestamp_value(self, value) -> Tuple[Optional[str], Optional[int]]:
        """
        Enhanced timestamp normalization from your test script.
        Return (date_str 'YYYY-MM-DD', epoch_seconds) or (None, None).
        Accepts seconds/ms ints, numeric strings, or ISO-8601 strings.
        """
        if value is None:
            return None, None
        
        # Try integer/float conversion first
        try:
            v = int(value)
            if v > 10**12:  # ms -> s
                v //= 1000
            dt = datetime.fromtimestamp(v, tz=timezoneDt.utc)
            return dt.strftime('%Y-%m-%d'), v
        except Exception:
            pass
        
        # Try string conversion
        s = str(value).strip()
        if s.isdigit():
            try:
                v = int(s)
                if v > 10**12:
                    v //= 1000
                dt = datetime.fromtimestamp(v, tz=timezoneDt.utc)
                return dt.strftime('%Y-%m-%d'), v
            except Exception:
                return None, None
        
        # Try ISO format parsing
        try:
            iso = s.replace('Z', '+00:00') if s.endswith('Z') else s
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezoneDt.utc)
            else:
                dt = dt.astimezone(timezoneDt.utc)
            return dt.strftime('%Y-%m-%d'), int(dt.timestamp())
        except Exception:
            return None, None
    
    async def _initialize_exchanges_with_normalization(self):
        """Initialize CCXT exchanges with proper initialization flow"""
        
        # Step 1: Collect unique exchange combinations
        unique_exchanges = set()
        for config in self.series_configs.values():
            exchange_data = await sync_to_async(
                lambda c=config: (c.exchange.name, c.market_type)
            )()
            unique_exchanges.add(exchange_data)
        
        logger.info(f"Found {len(unique_exchanges)} unique exchange combinations to initialize")
        
        # Step 2: Initialize exchanges
        for exchange_name, market_type in unique_exchanges:
            try:
                exchange_key = f"{exchange_name}_{market_type}"
                
                # Create sync exchange
                if hasattr(ccxt, exchange_name.lower()):
                    exchange_class = getattr(ccxt, exchange_name.lower())
                    exchange_config = {
                        'enableRateLimit': True,
                        'timeout': 30000,
                        'sandbox': False,
                    }
                    
                    # Set market type
                    if market_type != 'spot':
                        exchange_config['options'] = {'defaultType': market_type}
                    
                    sync_exchange = exchange_class(exchange_config)
                    
                    # Load markets
                    logger.info(f"Loading markets for {exchange_key}...")
                    await self._run_in_executor(sync_exchange.load_markets)
                    logger.info(f"Loaded {len(sync_exchange.markets)} markets for {exchange_key}")
                    
                    self.sync_exchanges[exchange_key] = sync_exchange
                    
                    # Rate limiting setup
                    self._setup_exchange_rate_limiting(exchange_key, sync_exchange)
                    
                    # Create WebSocket exchange for streaming
                    if hasattr(ccxtpro, exchange_name.lower()):
                        ws_exchange_class = getattr(ccxtpro, exchange_name.lower())
                        ws_exchange = ws_exchange_class(exchange_config)
                        await ws_exchange.load_markets()
                        
                        self.ws_exchanges[exchange_key] = ws_exchange
                        self.ws_subscriptions[exchange_key] = set()
                    
                    logger.info(f"Initialized exchange: {exchange_key}")
                else:
                    logger.warning(f"Exchange class not found in ccxt: {exchange_name}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize exchange {exchange_name}_{market_type}: {e}", exc_info=True)
        
        # Step 3: Build symbol mapping cache
        logger.info("Building symbol mapping cache...")
        for config in self.series_configs.values():
            exchange_data = await sync_to_async(
                lambda c=config: (c.exchange.name, c.market_type, c.symbol)
            )()
            exchange_key = f"{exchange_data[0]}_{exchange_data[1]}"
            db_symbol = exchange_data[2]
            
            if exchange_key not in self.sync_exchanges:
                logger.warning(f"Exchange {exchange_key} not initialized, skipping symbol mapping for {db_symbol}")
                continue
            
            exchange = self.sync_exchanges[exchange_key]
            
            # Find matching market symbol
            if db_symbol in exchange.markets:
                mapping_key = f"{exchange_key}_{db_symbol}"
                self.symbol_mapping_cache[mapping_key] = db_symbol
                logger.debug(f"Direct symbol match: {db_symbol}")
            else:
                # Try to find a match by normalizing
                for market_symbol in exchange.markets.keys():
                    if market_symbol.replace('/', '').replace(':', '') == \
                    db_symbol.replace('/', '').replace(':', ''):
                        mapping_key = f"{exchange_key}_{db_symbol}"
                        self.symbol_mapping_cache[mapping_key] = market_symbol
                        logger.debug(f"Normalized match: {db_symbol} -> {market_symbol}")
                        break
                else:
                    logger.warning(f"No market match found for {db_symbol} on {exchange_key}")
        
        logger.info(f"Symbol mapping cache built with {len(self.symbol_mapping_cache)} entries")

    def _setup_exchange_rate_limiting(self, exchange_key: str, exchange: ccxt.Exchange):
        """Setup rate limiting per exchange"""
        # Use exchange-specific rate limits from config or exchange settings
        exchange_name = exchange_key.split('_')[0]
        trading_config = getattr(settings, 'TRADING_CONFIG', {})
        exchange_rate_limits = trading_config.get('EXCHANGE_RATE_LIMITS', {})
        
        # Get rate limit in milliseconds
        rate_limit_ms = exchange_rate_limits.get(
            exchange_name, 
            getattr(exchange, 'rateLimit', 1000)
        )
        
        self.rate_limits[exchange_key] = rate_limit_ms / 1000.0  # Convert to seconds
        self.last_request_time[exchange_key] = 0
        self.rate_limit_backoff[exchange_key] = 0
        
        logger.info(f"Rate limit for {exchange_key}: {rate_limit_ms}ms")

    async def run_backfill_phase(self):
        """Phase 1 with complete backfill before streaming"""
        logger.info("Starting backfill phase")
        
        # Filter series that need backfilling
        backfill_series = [
            config for config in self.series_configs.values()
            if not config.backfill_completed and config.symbol  # CHANGED: use symbol directly
        ]
        
        logger.info(f"Found {len(backfill_series)} series requiring backfill")
        
        # Process series in batches to manage resources
        batch_size = 10
        completed_count = 0
        
        for i in range(0, len(backfill_series), batch_size):
            batch = backfill_series[i:i + batch_size]
            
            tasks = []
            for config in batch:
                task = asyncio.create_task(self._backfill_single_series(config))
                tasks.append(task)
            
            # Wait for batch completion
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful completions
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Backfill failed for {batch[j].series_key}: {result}")
                else:
                    completed_count += 1
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        # Mark phase as complete only when ALL series are done
        self.backfill_phase_complete = True
        logger.info(f"Backfill phase completed: {completed_count}/{len(backfill_series)} series successful")

    async def _backfill_single_series(self, config: SeriesConfig):
        """Backfill with proper graceful degradation and error handling"""
        series_key = config.series_key
        exchange_data = await sync_to_async(lambda: (config.exchange.name, config.market_type))()
        exchange_key = f"{exchange_data[0]}_{exchange_data[1]}"
        
        try:
            # Step 3: Determine earliest start with binary search
            if not config.earliest_start_ms:
                earliest_start_ms = await self._determine_earliest_start(config)
                if earliest_start_ms:
                    config.earliest_start_ms = earliest_start_ms
                    await self._async_db_save(config)
                else:
                    logger.warning(f"Could not determine earliest start for {series_key}")
                    return False

            # Step 4: Historical backfill with dynamic graceful degradation
            start_ms = max(
                config.earliest_start_ms,
                self.last_ts_cache.get(series_key, 0) + self._timeframe_to_ms(config.timeframe)
            )
            
            exchange = self.sync_exchanges.get(exchange_key)
            if not exchange:
                logger.error(f"Exchange not available: {exchange_key}")
                return False
            
            # Use the symbol directly (already normalized in TradingPair)
            if not config.symbol:
                logger.error(f"No symbol configured for {series_key}")
                return False
            
            # Backfill with proper graceful degradation
            success = await self._backfill_with_graceful_degradation(config, exchange, start_ms)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to backfill series {series_key}: {e}")
            config.last_error = str(e)
            await self._async_db_save(config)
            return False

    async def _determine_earliest_start(self, config: SeriesConfig) -> Optional[int]:
        """Enhanced earliest start determination with improved listing date detection"""
        series_key = config.series_key
        
        # Check for earliest date override first
        if self.earliest_date_override_ms:
            aligned_ms = self._floor_to_timeframe(self.earliest_date_override_ms, config.timeframe)
            logger.info(f"Using earliest date override for {series_key}: {datetime.fromtimestamp(aligned_ms/1000)}")
            config.earliest_start_ms = aligned_ms
            config.probing_completed = True
            await self._async_db_save(config)
            return aligned_ms
        
        # FIX: Use sync_to_async for Django model relationship access
        exchange_data = await sync_to_async(lambda: (config.exchange.name, config.market_type))()
        
        # Check if data exists in InfluxDB (resume case)
        data_exists = await self._run_in_executor(
            self.influx_manager.check_data_exists,
            exchange_data[0],
            exchange_data[1],
            config.symbol,
            config.timeframe
        )
        
        if data_exists:
            # Resume case: start from last timestamp + timeframe
            last_ts_ms = self.last_ts_cache.get(series_key, 0)
            if last_ts_ms:
                logger.info(f"Resuming {series_key} from existing data: {datetime.fromtimestamp(last_ts_ms/1000)}")
                return last_ts_ms + self._timeframe_to_ms(config.timeframe)
        
        # Check if we've already probed and persisted the result
        if config.earliest_start_ms and config.probing_completed:
            logger.info(f"Using cached earliest start for {series_key}: {datetime.fromtimestamp(config.earliest_start_ms/1000)}")
            return config.earliest_start_ms
        
        # First backfill case: try enhanced listing date detection first
        exchange_key = f"{exchange_data[0]}_{exchange_data[1]}"
        exchange = self.sync_exchanges.get(exchange_key)
        
        if not exchange:
            logger.error(f"Exchange not available: {exchange_key}")
            return None
        
        # ENHANCED: Try to get listing date from exchange metadata
        logger.info(f"Attempting to find listing date for {series_key}")
        listing_start_ms = await self._get_listing_date(config, exchange)
        
        if listing_start_ms:
            # Validate that the listing date makes sense (not too old or too recent)
            current_ms = int(time.time() * 1000)
            ten_years_ago_ms = current_ms - (10 * 365 * 24 * 60 * 60 * 1000)
            one_minute_ago_ms = current_ms - (60 * 1000)
            
            if ten_years_ago_ms <= listing_start_ms <= one_minute_ago_ms:
                logger.info(f"Using listing date as start for {series_key}: {datetime.fromtimestamp(listing_start_ms/1000)}")
                config.earliest_start_ms = listing_start_ms
                config.probing_completed = True
                await self._async_db_save(config)
                return listing_start_ms
            else:
                logger.warning(f"Listing date for {series_key} seems invalid: {datetime.fromtimestamp(listing_start_ms/1000)}, will use probing")
        
        # Dynamic backward probing only if listing date not found or invalid
        if config.probe_attempts < 3:  # Limit probe attempts
            logger.info(f"No valid listing date found for {series_key}, starting dynamic probing")
            earliest_ms = await self._dynamic_backward_probe(config, exchange)
            if earliest_ms:
                config.earliest_start_ms = earliest_ms
                config.probe_attempts += 1
                config.probing_completed = True
                await self._async_db_save(config)
                logger.info(f"Probing completed for {series_key}: {datetime.fromtimestamp(earliest_ms/1000)}")
                return earliest_ms
        else:
            logger.warning(f"Max probe attempts reached for {series_key}")
        
        # Fallback to default start (2 years ago)
        fallback_ms = int((time.time() - 2 * 365 * 24 * 3600) * 1000)
        aligned_fallback = self._floor_to_timeframe(fallback_ms, config.timeframe)
        logger.info(f"Using fallback start for {series_key}: {datetime.fromtimestamp(aligned_fallback/1000)}")
        return aligned_fallback

    async def _dynamic_backward_probe(self, config: SeriesConfig, exchange: ccxt.Exchange) -> Optional[int]:
        """Dynamic backward probing with correct binary search"""
        logger.info(f"Starting dynamic probe with binary search for {config.series_key}")
        
        # Use the symbol directly (already normalized)
        symbol = config.symbol
        if not symbol:
            return None
        
        # Initial parameters
        now_ms = int(time.time() * 1000)
        initial_guess_ms = now_ms - (2 * 365 * 24 * 3600 * 1000)  # 2 years back
        
        # Phase 1: Coarse backward stepping to find bounds
        current_guess_ms = initial_guess_ms
        step_size_days = 30
        last_success_ms = None  # Earliest successful timestamp
        first_fail_ms = None    # Latest failed timestamp
        max_steps = 20
        steps_taken = 0
        
        exchange_data = await sync_to_async(lambda: (config.exchange.name, config.market_type))()
        exchange_key = f"{exchange_data[0]}_{exchange_data[1]}"
        
        try:
            # Backward stepping phase - find approximate bounds
            while steps_taken < max_steps:
                try:
                    # Apply rate limiting
                    await self._apply_rate_limit(exchange_key)
                    
                    # Fetch small sample with integer limit
                    ohlcv_data = await self._run_in_executor(
                        exchange.fetch_ohlcv,
                        symbol,  # Use symbol directly
                        config.timeframe,
                        current_guess_ms,
                        int(5)
                    )
                    
                    if ohlcv_data and len(ohlcv_data) > 0:
                        # Data found - this timestamp has data
                        first_candle_ms = int(ohlcv_data[0][0])
                        last_success_ms = first_candle_ms
                        
                        # Move further back to find the actual start
                        step_size_days = min(step_size_days * 2, 180)  # Cap at 6 months
                        current_guess_ms -= (step_size_days * 24 * 3600 * 1000)
                        
                        logger.debug(f"Probe success at {datetime.fromtimestamp(first_candle_ms/1000)}, stepping back {step_size_days} days")
                    else:
                        # No data - this timestamp is too early
                        first_fail_ms = current_guess_ms
                        logger.debug(f"Probe failed at {datetime.fromtimestamp(current_guess_ms/1000)}")
                        break
                    
                    steps_taken += 1
                    await asyncio.sleep(0.1)  # Small delay between probes
                    
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        await self._handle_rate_limit_error(exchange_key)
                        continue
                    else:
                        logger.warning(f"Probe step failed for {config.series_key}: {e}")
                        first_fail_ms = current_guess_ms
                        break
            
            # Phase 2: Binary search refinement between bounds
            if last_success_ms and first_fail_ms:
                refined_start_ms = await self._binary_search_earliest_data(
                    config, exchange, first_fail_ms, last_success_ms
                )
                if refined_start_ms:
                    aligned_ms = self._floor_to_timeframe(refined_start_ms, config.timeframe)
                    logger.info(f"Binary search completed for {config.series_key}: earliest ~{datetime.fromtimestamp(aligned_ms/1000)}")
                    return aligned_ms
            
            # Return best result if we have one
            if last_success_ms:
                aligned_ms = self._floor_to_timeframe(last_success_ms, config.timeframe)
                logger.info(f"Probe completed for {config.series_key}: earliest ~{datetime.fromtimestamp(aligned_ms/1000)}")
                return aligned_ms
            
        except Exception as e:
            logger.error(f"Dynamic probe failed for {config.series_key}: {e}")
        
        return None

    async def _binary_search_earliest_data(self, config: SeriesConfig, exchange: ccxt.Exchange, fail_ms: int, success_ms: int) -> Optional[int]:
        """Proper binary search implementation with correct bounds handling"""
        logger.debug(f"Starting binary search for {config.series_key} between {datetime.fromtimestamp(fail_ms/1000)} (FAIL) and {datetime.fromtimestamp(success_ms/1000)} (SUCCESS)")
        
        # Use the symbol directly (already normalized)
        symbol = config.symbol
        
        # Proper binary search bounds
        left_bound = fail_ms      # No data here
        right_bound = success_ms  # Data exists here
        best_result = success_ms  # Best known result so far
        
        max_iterations = 10
        iteration = 0
        
        exchange_data = await sync_to_async(lambda: (config.exchange.name, config.market_type))()
        exchange_key = f"{exchange_data[0]}_{exchange_data[1]}"
        
        while (right_bound - left_bound) > (24 * 60 * 60 * 1000) and iteration < max_iterations:
            mid = (left_bound + right_bound) // 2
            
            try:
                await self._apply_rate_limit(exchange_key)
                
                ohlcv_data = await self._run_in_executor(
                    exchange.fetch_ohlcv,
                    symbol,  # Use symbol directly
                    config.timeframe,
                    mid,
                    int(5)
                )
                
                if ohlcv_data and len(ohlcv_data) > 0:
                    # Data found at mid - this becomes our new SUCCESS bound
                    first_candle_ms = int(ohlcv_data[0][0])
                    right_bound = mid  # Can go earlier
                    best_result = first_candle_ms  # Update best result
                    logger.debug(f"Binary search: data found at {datetime.fromtimestamp(first_candle_ms/1000)}, searching earlier")
                else:
                    # No data at mid - this becomes our new FAIL bound
                    left_bound = mid + (60 * 60 * 1000)  # Move forward by 1 hour
                    logger.debug(f"Binary search: no data at {datetime.fromtimestamp(mid/1000)}, searching later")
                
                await asyncio.sleep(0.1)
                iteration += 1
                
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    await self._handle_rate_limit_error(exchange_key)
                    continue
                else:
                    logger.warning(f"Binary search step failed: {e}")
                    break
        
        logger.debug(f"Binary search completed after {iteration} iterations. Best result: {datetime.fromtimestamp(best_result/1000)}")
        return best_result

    async def _backfill_with_graceful_degradation(self, config: SeriesConfig, exchange: ccxt.Exchange, start_ms: int):
        """Backfill with proper graceful degradation and attempt capping"""
        series_key = config.series_key
        exchange_data = await sync_to_async(lambda: (config.exchange.name, config.market_type))()
        exchange_key = f"{exchange_data[0]}_{exchange_data[1]}"
        
        # Use the symbol directly (already normalized)
        symbol = config.symbol
        
        current_start_ms = start_ms
        near_now_ms = int(time.time() * 1000) - (5 * 60 * 1000)  # 5 minutes ago
        consecutive_failures = 0
        max_failures = 5
        
        # Graceful degradation with attempt capping
        degradation_steps = ['1d', '3d', '7d']
        current_degradation_step = 0
        max_degradation_attempts_per_step = 3
        degradation_attempts_this_step = 0
        
        while current_start_ms < near_now_ms and consecutive_failures < max_failures:
            try:
                # Apply rate limiting
                await self._apply_rate_limit(exchange_key)
                
                # Ensure limit is always an integer and handle exchange-specific limits
                exchange_name = exchange_data[0].lower()
                if exchange_name == 'binance':
                    limit = 1000  # Use fixed limit for Binance
                else:
                    exchange_limit = getattr(exchange, 'has', {}).get('fetchOHLCV', 1000)
                    if isinstance(exchange_limit, bool):
                        exchange_limit = 1000
                    limit = int(min(exchange_limit, 1000))
                 
                ohlcv_data = await self._run_in_executor(
                    exchange.fetch_ohlcv,
                    symbol,  # Use symbol directly
                    config.timeframe,
                    current_start_ms,
                    limit
                )
                
                if not ohlcv_data or len(ohlcv_data) == 0:
                    # Graceful degradation with proper capping
                    if current_degradation_step < len(degradation_steps) and degradation_attempts_this_step < max_degradation_attempts_per_step:
                        degradation_period = degradation_steps[current_degradation_step]
                        nudge_ms = self._parse_degradation_period(degradation_period)
                        current_start_ms += nudge_ms
                        degradation_attempts_this_step += 1
                        consecutive_failures += 1
                        
                        logger.warning(f"No data for {series_key}, applying graceful degradation: nudging forward by {degradation_period} (attempt {degradation_attempts_this_step}/{max_degradation_attempts_per_step})")
                        
                        # If max attempts for this step reached, escalate to next step
                        if degradation_attempts_this_step >= max_degradation_attempts_per_step:
                            current_degradation_step += 1
                            degradation_attempts_this_step = 0
                            logger.warning(f"Escalating degradation for {series_key} to step {current_degradation_step}")
                        
                        # Persist adjusted earliest_start_ms
                        config.earliest_start_ms = current_start_ms
                        await self._async_db_save(config)
                        continue
                    else:
                        # Max degradation reached
                        logger.error(f"Max degradation reached for {series_key}")
                        break
                
                # Process and validate data (rest of the method remains the same)
                formatted_data = []
                for candle in ohlcv_data:
                    if len(candle) >= 6:
                        formatted_candle = {
                            'timestamp': int(candle[0]),
                            'open': float(candle[1]) if candle[1] is not None else 0.0,
                            'high': float(candle[2]) if candle[2] is not None else 0.0,
                            'low': float(candle[3]) if candle[3] is not None else 0.0,
                            'close': float(candle[4]) if candle[4] is not None else 0.0,
                            'volume': float(candle[5]) if candle[5] is not None else 0.0,
                        }
                        
                        if self._validate_candle(formatted_candle):
                            formatted_data.append(formatted_candle)
                
                if formatted_data:
                    # Write to InfluxDB with smaller batches
                    batch_size = min(100, len(formatted_data))
                    for i in range(0, len(formatted_data), batch_size):
                        batch = formatted_data[i:i + batch_size]
                        
                        series_batch = [{
                            'exchange': exchange_data[0],
                            'market_type': exchange_data[1],
                            'symbol': config.symbol,
                            'timeframe': config.timeframe,
                            'ohlcv_data': batch
                        }]
                        
                        # Add backpressure handling
                        retry_count = 0
                        max_retries = 3
                        success = False
                        while retry_count < max_retries:
                            success = await self._run_in_executor(
                                self.influx_manager.write_ohlcv_batch,
                                series_batch,
                                self.last_ts_cache
                            )
                            
                            if success:
                                break
                            else:
                                retry_count += 1
                                if retry_count < max_retries:
                                    wait_time = min(2 ** retry_count, 10)
                                    logger.warning(f"Write failed for {series_key}, retrying in {wait_time}s (attempt {retry_count})")
                                    await asyncio.sleep(wait_time)
                        
                        if not success:
                            logger.error(f"Failed to write batch for {series_key} after {max_retries} attempts")
                            consecutive_failures += 1
                            break
                        
                        await asyncio.sleep(0.1)
                    
                    if success:
                        # Update progress
                        last_candle_ms = formatted_data[-1]['timestamp']
                        current_start_ms = last_candle_ms + self._timeframe_to_ms(config.timeframe)
                        
                        config.last_backfill_ms = last_candle_ms
                        await self._async_db_save(config)
                        
                        # Reset degradation on success
                        consecutive_failures = 0
                        current_degradation_step = 0
                        degradation_attempts_this_step = 0
                        
                        logger.debug(f"Backfilled {len(formatted_data)} candles for {series_key}")
                    else:
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1
                
                # Stop condition: received less than limit
                if len(ohlcv_data) < limit:
                    break
                    
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    await self._handle_rate_limit_error(exchange_key)
                    continue
                elif "timeout" in str(e).lower() or "time" in str(e).lower():
                    consecutive_failures += 1
                    logger.warning(f"Timeout error in backfill loop for {series_key}: {e}")
                    
                    backoff_seconds = min(5 * consecutive_failures, 60)
                    await asyncio.sleep(backoff_seconds)
                else:
                    consecutive_failures += 1
                    logger.error(f"Error in backfill loop for {series_key}: {e}")
                    
                    backoff_seconds = min(2 ** consecutive_failures, 300)
                    jitter = random.uniform(0.1, 0.5) * backoff_seconds
                    await asyncio.sleep(backoff_seconds + jitter)
        
        # Mark backfill as completed
        config.backfill_completed = True
        await self._async_db_save(config)
        logger.info(f"Backfill completed for {series_key}")
        return True


    async def _apply_rate_limit(self, exchange_key: str):
        """rate limiting with 429 handling and exponential backoff"""
        current_time = time.time()
        
        # Check if we have an active backoff
        if self.rate_limit_backoff[exchange_key] > 0:
            if current_time < self.rate_limit_backoff[exchange_key]:
                wait_time = self.rate_limit_backoff[exchange_key] - current_time
                logger.debug(f"Rate limit backoff active for {exchange_key}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            else:
                # Backoff expired, reset
                self.rate_limit_backoff[exchange_key] = 0
        
        # Apply normal rate limiting
        last_request = self.last_request_time.get(exchange_key, 0)
        time_since_last = current_time - last_request
        min_interval = self.rate_limits.get(exchange_key, 1.0)
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            # Add small jitter to prevent thundering herd
            jitter = random.uniform(0, 0.1) * wait_time
            await asyncio.sleep(wait_time + jitter)
        
        self.last_request_time[exchange_key] = time.time()

    async def _handle_rate_limit_error(self, exchange_key: str):
        """Handle 429 errors with exponential backoff and jitter"""
        current_backoff = self.rate_limit_backoff.get(exchange_key, 0)
        
        if current_backoff == 0:
            # First 429 error
            backoff_time = 1.0  # Start with 1 second
        else:
            # Exponential backoff with cap at 300 seconds (5 minutes)
            backoff_time = min(current_backoff * 2, 300)
        
        # Add jitter (10-50% of backoff time)
        jitter = random.uniform(0.1, 0.5) * backoff_time
        total_backoff = backoff_time + jitter
        
        # Set backoff expiry time
        self.rate_limit_backoff[exchange_key] = time.time() + total_backoff
        
        logger.warning(f"Rate limit hit for {exchange_key}, backing off for {total_backoff:.2f}s")
        await asyncio.sleep(total_backoff)

    def _parse_degradation_period(self, period: str) -> int:
        """Parse degradation period to milliseconds"""
        if period == '1d':
            return 24 * 60 * 60 * 1000
        elif period == '3d':
            return 3 * 24 * 60 * 60 * 1000
        elif period == '7d':
            return 7 * 24 * 60 * 60 * 1000
        else:
            return 24 * 60 * 60 * 1000  # Default to 1 day

    def _validate_candle(self, candle: dict) -> bool:
        """candle validation"""
        try:
            # Basic sanity checks
            if candle['high'] < candle['low']:
                return False
            if candle['open'] <= 0 or candle['close'] <= 0:
                return False
            if candle['high'] < max(candle['open'], candle['close']):
                return False
            if candle['low'] > min(candle['open'], candle['close']):
                return False
            
            # Check for reasonable values (not too extreme)
            ohlc_values = [candle['open'], candle['high'], candle['low'], candle['close']]
            if any(v > 1e15 or v < 1e-15 for v in ohlc_values):
                return False
            
            return True
        except Exception:
            return False

    async def _get_listing_date(self, config: SeriesConfig, exchange: ccxt.Exchange) -> Optional[int]:
        """Enhanced listing date detection using exchange metadata"""
        try:
            # Get exchange data properly
            exchange_data = await sync_to_async(lambda: (config.exchange.name, config.market_type))()
            exchange_name = exchange_data[0].lower()
            
            # Use the symbol directly (already normalized)
            symbol = config.symbol
            
            # Add debug logging to see what's happening
            logger.debug(f"Looking for listing date for {config.series_key}: symbol='{symbol}', exchange={exchange_name}")
            logger.debug(f"Available markets count: {len(exchange.markets)}")
            
            # Check if markets are loaded
            if not exchange.markets:
                logger.warning(f"No markets loaded for exchange {exchange_name}")
                return None
            
            # Better symbol matching - try exact match first, then case variations
            market_info = None
            symbol_variations = [
                symbol,                    # Exact match
                symbol.upper(),           # Uppercase
                symbol.lower(),           # Lowercase
                symbol.replace('/', ''),  # Remove slash (some exchanges)
                symbol.replace(':', ''),  # Remove colon (some exchanges)
            ]
            
            matched_symbol = None
            for sym_variant in symbol_variations:
                if sym_variant in exchange.markets:
                    market_info = exchange.markets[sym_variant]
                    matched_symbol = sym_variant
                    logger.debug(f"Found market using symbol variant: '{sym_variant}' for {config.series_key}")
                    break
            
            if not market_info:
                # Show available symbols for debugging
                available_symbols = list(exchange.markets.keys())[:10]  # First 10 for debugging
                logger.warning(f"Symbol {symbol} not found in markets for {config.series_key}")
                logger.debug(f"First 10 available symbols: {available_symbols}")
                return None
            
            info = market_info.get('info', {}) or {}
            
            # Add debug logging for market info
            logger.debug(f"Market info keys for {matched_symbol}: {list(info.keys())[:20]}")  # First 20 keys
            
            # Get prioritized listing keys for this exchange
            listing_keys = self._get_listing_keys_for_exchange(exchange_name)
            
            matched_key = None
            date_str = None
            epoch = None
            
            # Add debug logging for each key attempt
            logger.debug(f"Trying {len(listing_keys)} listing keys for {config.series_key}")
            
            # Try info-based keys first (prioritized by exchange)
            for key in listing_keys:
                if key in info and info[key] not in (None, '', 'null', 0):
                    logger.debug(f"Found key '{key}' with value: {info[key]} for {config.series_key}")
                    date_str, epoch = self._normalize_timestamp_value(info[key])
                    if date_str and epoch:
                        matched_key = key
                        logger.info(f"Found listing date for {config.series_key}: {date_str} (key={matched_key})")
                        break
                    else:
                        logger.debug(f"Key '{key}' failed normalization: {info[key]}")
                else:
                    # Only log missing keys at debug level to avoid spam
                    if key in info:
                        logger.debug(f"Key '{key}' has null/empty value: {info[key]}")
            
            # Fallback to CCXT 'created' field
            if not date_str and market_info.get('created'):
                logger.debug(f"Trying CCXT 'created' field: {market_info['created']} for {config.series_key}")
                date_str, epoch = self._normalize_timestamp_value(market_info['created'])
                if date_str and epoch:
                    matched_key = 'created'
                    logger.info(f"Found listing date for {config.series_key}: {date_str} (key={matched_key})")
            
            if epoch:
                # Convert to milliseconds and align to timeframe
                listing_ms = int(epoch * 1000)
                aligned_ms = self._floor_to_timeframe(listing_ms, config.timeframe)
                
                # Store the listing date info in config with proper async save
                config.listing_date_ms = aligned_ms
                config.listing_date_key = matched_key
                await self._async_db_save(config)
                
                logger.info(f"Stored listing date for {config.series_key}: {datetime.fromtimestamp(aligned_ms/1000)} (key={matched_key})")
                return aligned_ms
            else:
                logger.debug(f"No listing date found for {config.series_key} in market info")
                # Show some useful debug info about what was available
                info_keys_with_values = {k: v for k, v in info.items() if v not in (None, '', 'null', 0)}
                logger.debug(f"Non-empty info keys for {config.series_key}: {list(info_keys_with_values.keys())[:10]}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting listing date for {config.series_key}: {e}", exc_info=True)
            return None

    def _parse_listing_date(self, date_value) -> Optional[float]:
        """Parse various listing date formats"""
        if isinstance(date_value, (int, float)):
            # Assume it's a timestamp
            if date_value > 1e10:  # Milliseconds
                return date_value / 1000
            else:  # Seconds
                return date_value
        
        if isinstance(date_value, str):
            try:
                # Try ISO format
                dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                return dt.timestamp()
            except:
                try:
                    # Try timestamp string
                    return float(date_value)
                except:
                    pass
        
        return None

    async def initialize_with_filters(self, timeframes_filter=None, exchanges_filter=None, max_series=None):
        """Initialize with filters applied early to avoid loading unnecessary exchanges"""
        logger.info(f"Initializing SeriesOrchestrator {self.worker_id} with filters")
        
        try:
            # Step 1: Load series configuration with early filtering
            await self._load_series_configuration_filtered(timeframes_filter, exchanges_filter, max_series)
            logger.info(f"Loaded {len(self.series_configs)} filtered series configurations")
            
            # Step 2: Resume point discovery (only for filtered series)
            logger.info("Discovering resume points...")
            await self._discover_resume_points()
            logger.info("Resume points discovered")
            
            # Step 3: Initialize only the needed exchanges
            logger.info("Initializing exchanges (filtered)...")
            await self._initialize_exchanges_with_normalization()
            logger.info("Exchanges initialized")
            
            logger.info(f"SeriesOrchestrator initialized with {len(self.series_configs)} series")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
            return False
    
    async def _load_series_configuration(self):
        """Load series configuration from database with shard filtering"""
        try:
            # Apply shard filtering if shard_name is specified
            if self.shard_name and self.shard_name != "default":
                # Load series from the specific worker shard assignment
                worker_shard = await sync_to_async(WorkerShard.objects.get)(
                    shard_name=self.shard_name,
                    isactive=1,
                    isdeleted=0
                )
                
                assigned_series_keys = worker_shard.assigned_series or []
                if not assigned_series_keys:
                    logger.warning(f"No series assigned to shard {self.shard_name}")
                    return
                
                logger.info(f"Loading {len(assigned_series_keys)} series for shard {self.shard_name}")
                
                # Load only the assigned series configurations
                created_count = 0
                for series_key in assigned_series_keys:
                    # Parse series key: "exchangeid_market_type_symbol_timeframe"
                    parts = series_key.split('_')
                    if len(parts) < 4:
                        logger.warning(f"Invalid series key format: {series_key}")
                        continue
                    
                    exchange_id = int(parts[0])
                    market_type = parts[1]
                    symbol = '_'.join(parts[2:-1])  # Handle symbols with underscores
                    timeframe = parts[-1]
                    
                    try:
                        # Get the specific series config
                        series_config = await sync_to_async(SeriesConfig.objects.get)(
                            exchange__exchangeid=exchange_id,
                            market_type=market_type,
                            symbol=symbol,
                            timeframe=timeframe,
                            isactive=1,
                            isdeleted=0
                        )
                        
                        self.series_configs[series_key] = series_config
                        logger.debug(f"Loaded series config: {series_key}")
                        
                    except SeriesConfig.DoesNotExist:
                        logger.warning(f"Series config not found: {series_key}")
                        continue
                    except Exception as e:
                        logger.error(f"Error loading series config {series_key}: {e}")
                        continue
                
                logger.info(f"Shard series configuration loaded: {len(self.series_configs)} series for shard {self.shard_name}")
                return
            
            # Original logic for non-shard mode (load all series)
            # Get active exchanges and trading pairs asynchronously
            exchanges = await sync_to_async(list)(
                Exchange.objects.filter(isactive=1, isdeleted=0)
            )
            
            trading_pairs = await sync_to_async(list)(
                TradingPair.objects.filter(isactive=1, isdeleted=0).select_related('exchange')
            )
            
            logger.info(f"Loading configuration for {len(exchanges)} exchanges and {len(trading_pairs)} trading pairs")
            
            # Series expansion - create series for each (exchange, market_type, symbol, timeframe)
            created_count = 0
            for pair in trading_pairs:
                for timeframe in self.supported_timeframes:
                    series_key = f"{pair.exchange.exchangeid}_{pair.exchange.market_type}_{pair.symbol}_{timeframe}"
                    
                    # Get or create SeriesConfig asynchronously
                    series_config, created = await sync_to_async(SeriesConfig.objects.get_or_create)(
                        exchange=pair.exchange,
                        market_type=pair.exchange.market_type,
                        symbol=pair.symbol,
                        timeframe=timeframe,
                        defaults={
                            'earliest_start_ms': None,
                            'last_backfill_ms': None,
                            'last_ts_ms': None,
                            'probing_completed': False,
                            'backfill_completed': False,
                            'probe_attempts': 0,
                            'last_error': None,
                            'isactive': 1,
                            'isdeleted': 0
                        }
                    )
                    
                    self.series_configs[series_key] = series_config
                    
                    if created:
                        created_count += 1
                        logger.debug(f"Created new series config: {series_key}")
            
            logger.info(f"Series configuration loaded: {len(self.series_configs)} total, {created_count} created")
            
        except WorkerShard.DoesNotExist:
            logger.error(f"Worker shard {self.shard_name} not found")
            raise
        except Exception as e:
            logger.error(f"Error loading series configuration: {e}")
            raise
        
    async def _load_series_configuration_filtered(self, timeframes_filter=None, exchanges_filter=None, max_series=None):
        """Load series configuration with early filtering to avoid unnecessary work"""
        try:
            # Apply shard filtering first if shard_name is specified
            if self.shard_name and self.shard_name != "default":
                # Load series from the specific worker shard assignment
                worker_shard = await sync_to_async(WorkerShard.objects.get)(
                    shard_name=self.shard_name,
                    isactive=1,
                    isdeleted=0
                )
                
                assigned_series_keys = worker_shard.assigned_series or []
                if not assigned_series_keys:
                    logger.warning(f"No series assigned to shard {self.shard_name}")
                    return
                
                logger.info(f"Loading {len(assigned_series_keys)} series for shard {self.shard_name} with filters")
                
                # Apply additional filters to the assigned series
                filtered_series_keys = []
                for series_key in assigned_series_keys:
                    # Parse series key to apply filters
                    parts = series_key.split('_')
                    if len(parts) < 4:
                        continue
                    
                    exchange_id = int(parts[0])
                    market_type = parts[1]
                    symbol = '_'.join(parts[2:-1])
                    timeframe = parts[-1]
                    
                    # Apply timeframe filter
                    if timeframes_filter and timeframe not in timeframes_filter:
                        continue
                    
                    # Apply max_series filter
                    if max_series and len(filtered_series_keys) >= max_series:
                        break
                    
                    # Get exchange name for exchange filter
                    try:
                        exchange = await sync_to_async(Exchange.objects.get)(exchangeid=exchange_id)
                        if exchanges_filter and exchange.name.lower() not in exchanges_filter:
                            continue
                        
                        filtered_series_keys.append(series_key)
                    except Exchange.DoesNotExist:
                        continue
                
                # Load the filtered series configurations
                created_count = 0
                for series_key in filtered_series_keys:
                    parts = series_key.split('_')
                    exchange_id = int(parts[0])
                    market_type = parts[1]
                    symbol = '_'.join(parts[2:-1])
                    timeframe = parts[-1]
                    
                    try:
                        series_config = await sync_to_async(SeriesConfig.objects.get)(
                            exchange__exchangeid=exchange_id,
                            market_type=market_type,
                            symbol=symbol,
                            timeframe=timeframe,
                            isactive=1,
                            isdeleted=0
                        )
                        
                        self.series_configs[series_key] = series_config
                        
                    except SeriesConfig.DoesNotExist:
                        continue
                
                logger.info(f"Shard filtered series configuration loaded: {len(self.series_configs)} series for shard {self.shard_name}")
                return
            
            # Original filtered logic for non-shard mode
            # Build the filter for exchanges early
            exchange_filter = {}
            if exchanges_filter:
                exchange_filter['name__in'] = [ex.lower() for ex in exchanges_filter]
            
            # Get filtered exchanges
            exchanges = await sync_to_async(list)(
                Exchange.objects.filter(isactive=1, isdeleted=0, **exchange_filter)
            )
            
            if not exchanges:
                logger.warning(f"No exchanges found matching filter: {exchanges_filter}")
                return
            
            # Get trading pairs for filtered exchanges only
            trading_pairs = await sync_to_async(list)(
                TradingPair.objects.filter(
                    isactive=1, 
                    isdeleted=0, 
                    exchange__in=exchanges
                ).select_related('exchange')
            )
            
            logger.info(f"Loading configuration for {len(exchanges)} exchanges and {len(trading_pairs)} trading pairs (filtered)")
            
            # Apply timeframe filtering early
            target_timeframes = timeframes_filter if timeframes_filter else self.supported_timeframes
            
            # Series expansion with early filtering
            created_count = 0
            series_created = 0
            for pair in trading_pairs:
                for timeframe in target_timeframes:
                    series_key = f"{pair.exchange.exchangeid}_{pair.exchange.market_type}_{pair.symbol}_{timeframe}"
                    
                    # Apply max_series early
                    if max_series and series_created >= max_series:
                        break
                    
                    series_config, created = await sync_to_async(SeriesConfig.objects.get_or_create)(
                        exchange=pair.exchange,
                        market_type=pair.exchange.market_type,
                        symbol=pair.symbol,
                        timeframe=timeframe,
                        defaults={
                            'earliest_start_ms': None,
                            'last_backfill_ms': None,
                            'last_ts_ms': None,
                            'probing_completed': False,
                            'backfill_completed': False,
                            'probe_attempts': 0,
                            'last_error': None,
                            'isactive': 1,
                            'isdeleted': 0
                        }
                    )
                    
                    self.series_configs[series_key] = series_config
                    series_created += 1
                    
                    if created:
                        created_count += 1
                
                # Early break for max_series
                if max_series and series_created >= max_series:
                    break
            
            logger.info(f"Series configuration loaded: {len(self.series_configs)} total, {created_count} created (filtered)")
            
        except WorkerShard.DoesNotExist:
            logger.error(f"Worker shard {self.shard_name} not found")
            raise
        except Exception as e:
            logger.error(f"Error loading filtered series configuration: {e}")
            raise

    async def run_streaming_phase(self):
        """Phase 2 with proper phase separation, WebSocket management, and gap detection"""
        # Only start streaming if backfill phase is complete
        if not self.backfill_phase_complete:
            logger.error("Cannot start streaming phase - backfill phase not complete")
            return
        
        logger.info("Starting streaming phase with WebSocket connections")
         
        # Group ONLY completed series by exchange_key for shared WS clients
        completed_series = [
            config for config in self.series_configs.values()
            if config.backfill_completed and config.symbol  # CHANGED: use symbol directly
        ]
        
        if not completed_series:
            logger.warning("No completed series available for streaming")
            return
        
        series_by_exchange = {}
        for config in completed_series:
            exchange_data = await sync_to_async(lambda c=config: (c.exchange.name, c.market_type))()
            exchange_key = f"{exchange_data[0]}_{exchange_data[1]}"
            if exchange_key not in series_by_exchange:
                series_by_exchange[exchange_key] = []
            series_by_exchange[exchange_key].append(config)
        
        logger.info(f"Starting streaming for {len(series_by_exchange)} exchange groups with {len(completed_series)} total series")
        
        # Mark streaming phase as active
        self.streaming_phase_active = True
        
        # Start gap detection task if enabled
        if self.gap_detection_enabled:
            logger.info("Starting gap detection task")
            self.gap_detection_task = asyncio.create_task(self._gap_detection_loop())
        
        # Start one WebSocket task per (exchange, market_type)
        for exchange_key, configs in series_by_exchange.items():
            if exchange_key in self.ws_exchanges:
                task = asyncio.create_task(self._stream_exchange_group(exchange_key, configs))
                self.ws_tasks[exchange_key] = task
        
        # Wait for all streaming tasks (gap detection runs concurrently)
        try:
            await asyncio.gather(*self.ws_tasks.values(), return_exceptions=True)
        except Exception as e:
            logger.error(f"Streaming phase error: {e}")
        finally:
            self.streaming_phase_active = False
            
            # Stop gap detection
            if self.gap_detection_task and not self.gap_detection_task.done():
                self.gap_detection_task.cancel()

    async def _gap_detection_loop(self):
        """Periodic gap detection and targeted backfill scheduling"""
        logger.info(f"Gap detection started - scanning every {self.gap_scan_interval} seconds")
        
        while self.is_running and self.streaming_phase_active:
            try:
                # Get all active series that have completed initial backfill
                active_series = [
                    config for config in self.series_configs.values()
                    if config.backfill_completed and config.symbol and config.isactive  # CHANGED: use symbol directly
                ]
                
                if not active_series:
                    logger.debug("No active series for gap detection")
                    await asyncio.sleep(self.gap_scan_interval)
                    continue
                
                logger.info(f"Starting gap detection scan for {len(active_series)} series")
                gaps_found = 0
                gaps_filled = 0
                
                # Process series in small batches to avoid overwhelming the system
                batch_size = 5
                for i in range(0, len(active_series), batch_size):
                    batch = active_series[i:i + batch_size]
                    
                    # Scan each series in the batch
                    tasks = []
                    for config in batch:
                        task = asyncio.create_task(self._scan_series_for_gaps(config))
                        tasks.append(task)
                    
                    # Wait for batch completion
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results and schedule backfills
                    for j, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"Gap scan failed for {batch[j].series_key}: {result}")
                        elif result:
                            series_key = batch[j].series_key
                            detected_gaps = result
                            gaps_found += len(detected_gaps)
                            
                            # Store detected gaps
                            self.detected_gaps[series_key] = detected_gaps
                            
                            # Schedule targeted backfills for gaps
                            filled_count = await self._schedule_gap_backfills(batch[j], detected_gaps)
                            gaps_filled += filled_count
                    
                    # Small delay between batches
                    await asyncio.sleep(1)
                
                logger.info(f"Gap detection scan completed: {gaps_found} gaps found, {gaps_filled} gaps filled")
                
                # Wait for next scan interval
                await asyncio.sleep(self.gap_scan_interval)
                
            except asyncio.CancelledError:
                logger.info("Gap detection cancelled")
                break
            except Exception as e:
                logger.error(f"Error in gap detection loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _scan_series_for_gaps(self, config: SeriesConfig) -> Optional[List[Tuple[int, int]]]:
        """Scan a single series for missing candles and return detected gaps"""
        series_key = config.series_key
        
        try:
            # Skip if recently scanned (avoid over-scanning)
            current_time = int(time.time())
            last_scan = self.last_gap_scan.get(series_key, 0)
            min_scan_interval = max(self.gap_scan_interval // 2, 300)  # At least 5 minutes
            
            if current_time - last_scan < min_scan_interval:
                return None
            
            self.last_gap_scan[series_key] = current_time
            
            # Get the time range to scan for gaps
            scan_end_ms = int(time.time() * 1000) - (5 * 60 * 1000)  # 5 minutes ago
            timeframe_ms = self._timeframe_to_ms(config.timeframe)
            
            # Determine scan start time (last 24 hours or from earliest_start_ms)
            scan_duration_ms = 24 * 60 * 60 * 1000  # 24 hours
            scan_start_ms = max(
                scan_end_ms - scan_duration_ms,
                config.earliest_start_ms or 0
            )
            
            # Align to timeframe boundaries
            scan_start_ms = self._floor_to_timeframe(scan_start_ms, config.timeframe)
            scan_end_ms = self._floor_to_timeframe(scan_end_ms, config.timeframe)
            
            if scan_start_ms >= scan_end_ms:
                return None
            
            # Query InfluxDB for existing candles in the time range
            existing_timestamps = await self._run_in_executor(
                self.influx_manager.get_existing_timestamps,
                config.exchange.name,
                config.market_type,
                config.symbol,
                config.timeframe,
                scan_start_ms,
                scan_end_ms
            )
            
            # Generate expected timestamps
            expected_timestamps = []
            current_ts = scan_start_ms
            while current_ts <= scan_end_ms:
                expected_timestamps.append(current_ts)
                current_ts += timeframe_ms
            
            # Find missing timestamps
            existing_set = set(existing_timestamps)
            missing_timestamps = [ts for ts in expected_timestamps if ts not in existing_set]
            
            if not missing_timestamps:
                logger.debug(f"No gaps found for {series_key}")
                return None
            
            # Group consecutive missing timestamps into gap ranges
            gaps = self._group_consecutive_gaps(missing_timestamps, timeframe_ms)
            
            if gaps:
                logger.info(f"Found {len(gaps)} gaps for {series_key}: {[(datetime.fromtimestamp(start/1000), datetime.fromtimestamp(end/1000)) for start, end in gaps]}")
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error scanning for gaps in {series_key}: {e}")
            return None

    def _group_consecutive_gaps(self, missing_timestamps: List[int], timeframe_ms: int) -> List[Tuple[int, int]]:
        """Group consecutive missing timestamps into gap ranges"""
        if not missing_timestamps:
            return []
        
        missing_timestamps.sort()
        gaps = []
        current_gap_start = missing_timestamps[0]
        current_gap_end = missing_timestamps[0]
        
        for i in range(1, len(missing_timestamps)):
            timestamp = missing_timestamps[i]
            
            # Check if this timestamp is consecutive to the current gap
            if timestamp == current_gap_end + timeframe_ms:
                # Extend current gap
                current_gap_end = timestamp
            else:
                # End current gap and start a new one
                gaps.append((current_gap_start, current_gap_end))
                current_gap_start = timestamp
                current_gap_end = timestamp
        
        # Add the final gap
        gaps.append((current_gap_start, current_gap_end))
        
        return gaps

    async def _schedule_gap_backfills(self, config: SeriesConfig, gaps: List[Tuple[int, int]]) -> int:
        """Schedule targeted backfills for detected gaps"""
        series_key = config.series_key
        exchange_key = f"{config.exchange.name}_{config.market_type}"
        filled_count = 0
        
        if not gaps:
            return filled_count
        
        exchange = self.sync_exchanges.get(exchange_key)
        if not exchange:
            logger.error(f"Exchange not available for gap backfill: {exchange_key}")
            return filled_count
        
        logger.info(f"Scheduling backfill for {len(gaps)} gaps in {series_key}")
        
        # Use the symbol directly (already normalized)
        symbol = config.symbol
        
        for gap_start_ms, gap_end_ms in gaps:
            try:
                # Apply rate limiting before gap backfill
                await self._apply_rate_limit(exchange_key)
                
                # Calculate limit for this gap
                timeframe_ms = self._timeframe_to_ms(config.timeframe)
                gap_duration_ms = gap_end_ms - gap_start_ms + timeframe_ms
                expected_candles = max(1, gap_duration_ms // timeframe_ms)
                
                # Limit the number of candles to fetch to avoid overwhelming
                limit = min(expected_candles, 1000)
                
                logger.debug(f"Filling gap in {series_key}: {datetime.fromtimestamp(gap_start_ms/1000)} to {datetime.fromtimestamp(gap_end_ms/1000)} ({limit} candles expected)")
                
                # Fetch OHLCV data for the gap
                ohlcv_data = await self._run_in_executor(
                    exchange.fetch_ohlcv,
                    symbol,  # Use symbol directly
                    config.timeframe,
                    gap_start_ms,
                    int(limit)
                )
                
                if ohlcv_data and len(ohlcv_data) > 0:
                    # Process and validate data
                    formatted_data = []
                    for candle in ohlcv_data:
                        if len(candle) >= 6:
                            candle_timestamp = int(candle[0])
                            
                            # Only include candles within the gap range
                            if gap_start_ms <= candle_timestamp <= gap_end_ms:
                                formatted_candle = {
                                    'timestamp': candle_timestamp,
                                    'open': float(candle[1]) if candle[1] is not None else 0.0,
                                    'high': float(candle[2]) if candle[2] is not None else 0.0,
                                    'low': float(candle[3]) if candle[3] is not None else 0.0,
                                    'close': float(candle[4]) if candle[4] is not None else 0.0,
                                    'volume': float(candle[5]) if candle[5] is not None else 0.0,
                                }
                                
                                if self._validate_candle(formatted_candle):
                                    formatted_data.append(formatted_candle)
                    
                    if formatted_data:
                        # Write gap data to InfluxDB
                        series_batch = [{
                            'exchange': config.exchange.name,
                            'market_type': config.market_type,
                            'symbol': config.symbol,
                            'timeframe': config.timeframe,
                            'ohlcv_data': formatted_data
                        }]
                        
                        success = await self._run_in_executor(
                            self.influx_manager.write_ohlcv_batch,
                            series_batch,
                            self.last_ts_cache
                        )
                        
                        if success:
                            filled_count += 1
                            logger.info(f"Gap filled for {series_key}: {len(formatted_data)} candles written for gap {datetime.fromtimestamp(gap_start_ms/1000)} to {datetime.fromtimestamp(gap_end_ms/1000)}")
                            
                            # Update cache if this gap affects the latest timestamp
                            if formatted_data:
                                latest_gap_timestamp = max(candle['timestamp'] for candle in formatted_data)
                                current_latest = self.last_ts_cache.get(series_key, 0)
                                if latest_gap_timestamp > current_latest:
                                    self.last_ts_cache[series_key] = latest_gap_timestamp
                        else:
                            logger.warning(f"Failed to write gap data for {series_key}")
                    else:
                        logger.warning(f"No valid data received for gap in {series_key}")
                else:
                    logger.warning(f"No data returned for gap in {series_key}")
                
                # Small delay between gap fills
                await asyncio.sleep(0.1)
                
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    await self._handle_rate_limit_error(exchange_key)
                    continue
                else:
                    logger.error(f"Error filling gap in {series_key}: {e}")
                    break
        
        return filled_count
    

    async def _stream_exchange_group(self, exchange_key: str, configs: List[SeriesConfig]):
        """streaming with improved reconnection and error handling"""
        logger.info(f"Starting WebSocket streaming for {exchange_key} with {len(configs)} series")
        
        ws_exchange = self.ws_exchanges.get(exchange_key)
        if not ws_exchange:
            logger.error(f"WebSocket exchange not available: {exchange_key}")
            return
        
        # Track subscriptions for this WebSocket client
        subscribed_series = set()
        reconnect_attempts = 0
        max_reconnects = 10
        base_delay = 1
        
        # Set is_running to True at the start of streaming
        self.is_running = True
        
        while self.is_running and self.streaming_phase_active and reconnect_attempts < max_reconnects:
            try:
                # Subscribe to all series for this exchange
                subscription_success_count = 0
                for config in configs:
                    try:
                        # use config.symbol directly instead of normalized_symbol
                        subscription_key = f"{config.symbol}_{config.timeframe}"
                        
                        if subscription_key not in subscribed_series:
                            # Use CCXT Pro watch_ohlcv with direct symbol
                            await ws_exchange.watch_ohlcv(config.symbol, config.timeframe)
                            subscribed_series.add(subscription_key)
                            self.ws_subscriptions[exchange_key].add(subscription_key)
                            subscription_success_count += 1
                            logger.debug(f"Subscribed to {config.symbol} {config.timeframe} on {exchange_key}")
                            
                            # Small delay between subscriptions to avoid overwhelming
                            await asyncio.sleep(0.1)
                            
                    except Exception as e:
                        logger.error(f"Failed to subscribe to {config.series_key}: {e}")
                
                logger.info(f"Successfully subscribed to {subscription_success_count}/{len(configs)} series on {exchange_key}")
                
                # Reset reconnect attempts on successful subscription
                reconnect_attempts = 0
                
                # Main streaming loop with continuous processing
                logger.info(f"Entering main streaming loop for {exchange_key}")
                while self.is_running and self.streaming_phase_active:
                    try:
                        # Process updates for all subscribed series with timeout
                        update_tasks = []
                        for config in configs:
                            # use config.symbol directly
                            if f"{config.symbol}_{config.timeframe}" in subscribed_series:
                                task = asyncio.create_task(self._process_ws_updates(ws_exchange, config))
                                update_tasks.append(task)
                        
                        if update_tasks:
                            # Wait for all update tasks with timeout
                            await asyncio.wait_for(
                                asyncio.gather(*update_tasks, return_exceptions=True),
                                timeout=30.0
                            )
                        else:
                            # If no tasks, wait a bit to avoid busy loop
                            await asyncio.sleep(1.0)
                        
                    except asyncio.TimeoutError:
                        logger.debug(f"Timeout in streaming loop for {exchange_key} (this is normal)")
                        continue
                    except Exception as e:
                        logger.error(f"Error in streaming loop for {exchange_key}: {e}")
                        break
                
                logger.info(f"Streaming loop ended for {exchange_key} (is_running={self.is_running}, streaming_active={self.streaming_phase_active})")
                
            except Exception as e:
                reconnect_attempts += 1
                
                # Calculate delay with exponential backoff and jitter
                base_backoff = min(base_delay * (2 ** (reconnect_attempts - 1)), 300)  # Cap at 5 minutes
                jitter = random.uniform(0.1, 0.5) * base_backoff
                total_delay = base_backoff + jitter
                
                logger.error(f"WebSocket error for {exchange_key} (attempt {reconnect_attempts}/{max_reconnects}): {e}")
                logger.info(f"Reconnecting in {total_delay:.2f} seconds...")
                
                # Close current connection
                try:
                    await ws_exchange.close()
                except Exception as close_error:
                    logger.warning(f"Error closing WebSocket connection: {close_error}")
                
                await asyncio.sleep(total_delay)
                
                # Clear subscription tracking
                subscribed_series.clear()
                if exchange_key in self.ws_subscriptions:
                    self.ws_subscriptions[exchange_key].clear()
                
                # Recreate WebSocket exchange
                try:
                    exchange_name = exchange_key.split('_')[0]
                    market_type = '_'.join(exchange_key.split('_')[1:])
                    
                    if hasattr(ccxtpro, exchange_name.lower()):
                        ws_exchange_class = getattr(ccxtpro, exchange_name.lower())
                        exchange_config = {
                            'enableRateLimit': True,
                            'timeout': 30000,
                            'sandbox': False,
                        }
                        
                        if market_type != 'spot':
                            exchange_config['options'] = {'defaultType': market_type}
                        
                        ws_exchange = ws_exchange_class(exchange_config)
                        await ws_exchange.load_markets()
                        self.ws_exchanges[exchange_key] = ws_exchange
                        
                except Exception as recreate_error:
                    logger.error(f"Failed to recreate WebSocket for {exchange_key}: {recreate_error}")
                    break
        
        logger.info(f"WebSocket streaming ended for {exchange_key}")

    async def _process_ws_updates(self, ws_exchange: ccxtpro.Exchange, config: SeriesConfig):
        """WebSocket updates with correct closed-candle filtering"""
        try:
            # Watch for OHLCV updates using the symbol directly (already normalized)
            ohlcv_data = await ws_exchange.watch_ohlcv(config.symbol, config.timeframe)
            
            if not ohlcv_data:
                return
            
            # Correct closed-candle filtering logic
            current_time_ms = int(time.time() * 1000)
            timeframe_ms = self._timeframe_to_ms(config.timeframe)
            
            closed_candles = []
            for candle in ohlcv_data:
                if len(candle) >= 6:
                    candle_open_ms = int(candle[0])
                    
                    # A candle is closed when: current_time >= candle_open_time + timeframe_duration
                    candle_close_time_ms = candle_open_ms + timeframe_ms
                    
                    if current_time_ms >= candle_close_time_ms:
                        formatted_candle = {
                            'timestamp': candle_open_ms,
                            'open': float(candle[1]) if candle[1] is not None else 0.0,
                            'high': float(candle[2]) if candle[2] is not None else 0.0,
                            'low': float(candle[3]) if candle[3] is not None else 0.0,
                            'close': float(candle[4]) if candle[4] is not None else 0.0,
                            'volume': float(candle[5]) if candle[5] is not None else 0.0,
                        }
                        
                        if self._validate_candle(formatted_candle):
                            closed_candles.append(formatted_candle)
                    else:
                        # Log forming candles for debugging (but don't process them)
                        remaining_ms = candle_close_time_ms - current_time_ms
                        logger.debug(f"Skipping forming candle for {config.series_key}: {remaining_ms/1000:.1f}s remaining")
            
            if closed_candles:
                # Deduplication with cache consistency
                await self._write_candles_with_dedup(config, closed_candles)
                
        except Exception as e:
            logger.error(f"Error processing WebSocket update for {config.series_key}: {e}")

    async def _write_candles_with_dedup(self, config: SeriesConfig, candles: List[dict]):
        """candle writing with proper cache synchronization"""
        series_key = config.series_key
        
        # Sort candles by timestamp ascending (maintain monotonicity)
        candles.sort(key=lambda x: x['timestamp'])
        
        # deduplication using cache
        last_ts_ms = self.last_ts_cache.get(series_key, 0)
        new_candles = [c for c in candles if c['timestamp'] > last_ts_ms]
        
        if new_candles:
            # Write with batch processing
            series_batch = [{
                'exchange': config.exchange.name,
                'market_type': config.market_type,
                'symbol': config.symbol,
                'timeframe': config.timeframe,
                'ohlcv_data': new_candles
            }]
            
            success = await self._run_in_executor(
                self.influx_manager.write_ohlcv_batch,
                series_batch,
                self.last_ts_cache
            )
            
            if success:
                # Update local cache and database
                latest_timestamp = new_candles[-1]['timestamp']
                self.last_ts_cache[series_key] = latest_timestamp
                
                # Periodically sync cache to database (every 10th update to reduce DB load)
                if random.random() < 0.1:  # 10% chance
                    config.last_ts_ms = latest_timestamp
                    await self._async_db_save(config)
                
                logger.debug(f"Streamed {len(new_candles)} closed candles for {config.series_key}")
            else:
                logger.warning(f"Failed to write candles for {config.series_key}")

    # Utility methods
    async def _run_in_executor(self, func, *args):
        """Run synchronous function in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)

    async def _async_db_query(self, queryset):
        """Execute database query asynchronously"""
        from asgiref.sync import sync_to_async
        return await sync_to_async(list)(queryset)

    async def _async_db_get_or_create(self, model, **kwargs):
        """Execute get_or_create asynchronously"""
        from asgiref.sync import sync_to_async
        return await sync_to_async(model.objects.get_or_create)(**kwargs)

    async def _async_db_save(self, instance):
        """Save model instance asynchronously"""
        from asgiref.sync import sync_to_async
        return await sync_to_async(instance.save)()

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

    def _floor_to_timeframe(self, timestamp_ms: int, timeframe: str) -> int:
        """Floor timestamp to proper calendar boundaries"""
        import datetime
        
        if timeframe in ['1d', '1w', '1M']:
            dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000, tz=datetime.timezoneDt.utc)
            
            if timeframe == '1d':
                # Align to UTC midnight
                dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            elif timeframe == '1w':
                # Align to Monday UTC
                days_since_monday = dt.weekday()
                dt = dt - datetime.timedelta(days=days_since_monday)
                dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            elif timeframe == '1M':
                # Align to first day of month
                dt = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            return int(dt.timestamp() * 1000)
        else:
            # Regular timeframe alignment
            timeframe_ms = self._timeframe_to_ms(timeframe)
            return (timestamp_ms // timeframe_ms) * timeframe_ms

    async def run(self):
        """Main execution method with proper phase separation"""
        self.is_running = True
        
        try:
            # Initialize
            if not await self.initialize():
                logger.error("Initialization failed")
                return
            
            # Phase 1: Backfill (MUST complete fully before streaming)
            logger.info("=== PHASE 1: BACKFILL ===")
            await self.run_backfill_phase()
            
            if not self.backfill_phase_complete:
                logger.error("Backfill phase did not complete successfully")
                return
            
            # Phase 2: Streaming (only starts after Phase 1 completion)
            logger.info("=== PHASE 2: STREAMING ===")
            await self.run_streaming_phase()
            
        except Exception as e:
            logger.error(f"Orchestrator run failed: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources including gap detection task"""
        self.is_running = False
        self.streaming_phase_active = False
        
        # Cancel gap detection task
        if self.gap_detection_task and not self.gap_detection_task.done():
            logger.info("Stopping gap detection task")
            self.gap_detection_task.cancel()
            try:
                await self.gap_detection_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all WebSocket tasks
        for task in self.ws_tasks.values():
            if not task.done():
                task.cancel()
        
        # Close WebSocket exchanges
        for ws_exchange in self.ws_exchanges.values():
            try:
                await ws_exchange.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket exchange: {e}")
        
        # Close sync exchanges
        for exchange in self.sync_exchanges.values():
            try:
                if hasattr(exchange, 'close'):
                    await self._run_in_executor(exchange.close)
            except Exception as e:
                logger.warning(f"Error closing sync exchange: {e}")
        
        # Close InfluxDB
        if self.influx_manager:
            self.influx_manager.close()
        
        logger.info("SeriesOrchestrator cleanup completed")
