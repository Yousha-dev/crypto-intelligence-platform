from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.write_api import WriteOptions
import pandas as pd
from typing import List, Dict, Optional, Tuple
from django.conf import settings
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import random

logger = logging.getLogger(__name__)

class InfluxManager:
    """InfluxDB manager with better performance, error handling, and batching"""

    def __init__(self):
        self.client = None
        self.write_api = None
        self.query_api = None
        self.bucket = None
        self.org = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # FIX: Improved batching and performance features
        self.batch_size = 500  # Reduced batch size
        self.write_queue = queue.Queue(maxsize=5000)  # Reduced queue size
        self.batch_thread = None
        self.batch_thread_running = False
        self.write_lock = threading.Lock()
        
        # Performance metrics
        self.write_stats = {
            'total_points': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'dropped_points': 0,  # Track dropped points
            'last_write_time': 0,
            'avg_batch_time': 0,
        }
        
        # Error handling
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.backoff_time = 1.0
        
        if hasattr(settings, 'INFLUXDB_CONFIG'):
            try:
                config = settings.INFLUXDB_CONFIG
                self.client = InfluxDBClient(
                    url=config['url'],
                    token=config['token'],
                    org=config['org'],
                    timeout=60_000,  # FIX: Increased timeout to 60 seconds
                    retries=2,  # Reduced retries to prevent buildup
                )
                
                # FIX: More conservative write options
                write_options = WriteOptions(
                    batch_size=self.batch_size,
                    flush_interval=5_000,   # FIX: Reduced flush interval to 5 seconds
                    jitter_interval=1_000,  # FIX: Reduced jitter
                    retry_interval=3_000,   # FIX: Reduced retry interval
                    max_retries=2,          # FIX: Reduced max retries
                    exponential_base=2,
                )
                
                self.write_api = self.client.write_api(write_options=write_options)
                self.query_api = self.client.query_api()
                self.bucket = config['bucket']
                self.org = config['org']
                
                # Ensure bucket exists
                self._ensure_bucket()
                
                # Start batch processing thread
                self._start_batch_thread()
                
                logger.info("InfluxDB client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize InfluxDB client: {e}")
                self.client = None
        else:
            logger.warning("InfluxDB not configured in settings")

    def _ensure_bucket(self):
        """Ensure bucket exists with configurable retention"""
        try:
            config = settings.INFLUXDB_CONFIG
            retention_days = config.get('retention_days')
            
            buckets_api = self.client.buckets_api()
            bucket_list = buckets_api.find_buckets(name=self.bucket)
            
            retention_rules = []
            if retention_days is not None:
                from influxdb_client.domain.retention_rule import RetentionRule
                retention_rules = [RetentionRule(type="expire", every_seconds=retention_days * 24 * 3600)]
            
            if not bucket_list.buckets: 
                bucket = buckets_api.create_bucket(
                    bucket_name=self.bucket,
                    org=self.org,
                    retention_rules=retention_rules  # Empty if infinite
                )
                retention_msg = f"{retention_days} days" if retention_days else "infinite"
                logger.info(f"Created InfluxDB bucket with {retention_msg} retention: {self.bucket}")
            else:
                logger.info(f"InfluxDB bucket exists: {self.bucket}")
                
        except Exception as e:
            logger.warning(f"Could not ensure bucket: {e}")

    def _start_batch_thread(self):
        """Start background thread for batch processing"""
        if self.batch_thread is None or not self.batch_thread.is_alive():
            self.batch_thread_running = True
            self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
            self.batch_thread.start()
            logger.info("Started InfluxDB batch processing thread")

    def _batch_processor(self):
        """FIX: Improved batch processor with better error handling and backpressure"""
        batch_points = []
        last_flush_time = time.time()
        flush_interval = 5.0  # FIX: Reduced flush interval to 5 seconds
        consecutive_timeouts = 0
        max_consecutive_timeouts = 3
        
        while self.batch_thread_running:
            try:
                # Try to get points from queue (with timeout)
                try:
                    point = self.write_queue.get(timeout=0.5)  # Shorter timeout
                    if point is None:  # Shutdown signal
                        break
                    batch_points.append(point)
                except queue.Empty:
                    # No new points, check if we should flush existing batch
                    pass
                
                current_time = time.time()
                should_flush = (
                    len(batch_points) >= self.batch_size or
                    (batch_points and (current_time - last_flush_time) >= flush_interval)
                )
                
                if should_flush and batch_points:
                    success = self._flush_batch(batch_points)
                    if success:
                        consecutive_timeouts = 0
                    else:
                        consecutive_timeouts += 1
                        
                        # If too many consecutive timeouts, increase flush interval
                        if consecutive_timeouts >= max_consecutive_timeouts:
                            flush_interval = min(flush_interval * 1.5, 30.0)  # Max 30 seconds
                            logger.warning(f"Increased flush interval to {flush_interval}s due to consecutive failures")
                    
                    batch_points = []
                    last_flush_time = current_time
                    
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                # Clear problematic batch and continue
                batch_points = []
                time.sleep(self.backoff_time)
        
        # Flush remaining points on shutdown
        if batch_points:
            self._flush_batch(batch_points)
        
        logger.info("InfluxDB batch processor thread stopped")

    def _flush_batch(self, points: List[Point]) -> bool:
        """FIX: Improved batch flushing with better timeout handling"""
        if not points or not self.client:
            return False
        
        start_time = time.time()
        retry_count = 0
        max_retries = 2  # FIX: Reduced max retries
        
        while retry_count < max_retries:
            try:
                with self.write_lock:
                    self.write_api.write(bucket=self.bucket, org=self.org, record=points)
                
                # Success - update stats and reset error tracking
                write_time = time.time() - start_time
                self.write_stats['total_points'] += len(points)
                self.write_stats['successful_batches'] += 1
                self.write_stats['last_write_time'] = time.time()
                
                # Update average batch time (exponential moving average)
                if self.write_stats['avg_batch_time'] == 0:
                    self.write_stats['avg_batch_time'] = write_time
                else:
                    alpha = 0.1  # Smoothing factor
                    self.write_stats['avg_batch_time'] = (
                        alpha * write_time + (1 - alpha) * self.write_stats['avg_batch_time']
                    )
                
                self.consecutive_failures = 0
                self.backoff_time = 1.0  # Reset backoff
                
                logger.debug(f"Flushed batch of {len(points)} points in {write_time:.3f}s")
                return True
                
            except InfluxDBError as e:
                retry_count += 1
                self.consecutive_failures += 1
                
                error_code = getattr(e, 'status', 0)
                error_message = str(e).lower()
                
                if "timeout" in error_message:
                    # Handle timeout errors specifically
                    backoff = min(5 * retry_count, 30)  # 5s, 10s, max 30s
                    logger.warning(f"InfluxDB timeout (attempt {retry_count}), backing off for {backoff}s")
                    time.sleep(backoff)
                    
                elif error_code == 429:  # Rate limit
                    backoff = min(self.backoff_time * (2 ** retry_count), 60)  # Max 60s
                    jitter = random.uniform(0.1, 0.5) * backoff
                    sleep_time = backoff + jitter
                    
                    logger.warning(f"Rate limit hit, backing off for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    
                elif 400 <= error_code < 500:
                    # Client error - don't retry
                    logger.error(f"Client error writing to InfluxDB: {e}")
                    break
                    
                else:
                    # Server error or network issue - retry with backoff
                    backoff = min(self.backoff_time * retry_count, 30)
                    logger.warning(f"InfluxDB write error (attempt {retry_count}): {e}, retrying in {backoff}s")
                    time.sleep(backoff)
                    
            except Exception as e:
                retry_count += 1
                error_message = str(e).lower()
                
                if "timeout" in error_message:
                    backoff = min(10 * retry_count, 60)  # 10s, 20s, max 60s
                    logger.warning(f"Timeout error (attempt {retry_count}): {e}, retrying in {backoff}s")
                    time.sleep(backoff)
                else:
                    logger.error(f"Unexpected error writing to InfluxDB (attempt {retry_count}): {e}")
                    time.sleep(min(retry_count * 2, 10))  # Simple backoff
        
        # All retries failed
        self.write_stats['failed_batches'] += 1
        self.write_stats['dropped_points'] += len(points)
        logger.error(f"Failed to write batch of {len(points)} points after {max_retries} attempts")
        
        # Exponential backoff for consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.backoff_time = min(self.backoff_time * 2, 60)
            logger.warning(f"Multiple consecutive failures, increasing backoff to {self.backoff_time}s")
        
        return False

    def get_latest_timestamps_per_series(self) -> Dict[str, int]:
        """Query with better error handling and caching"""
        if not self.client:
            return {}
        
        try:
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r._field == "open")
                |> group(columns: ["exchange", "market_type", "symbol", "timeframe"])
                |> max(column: "_time")
                |> map(fn: (r) => ({{
                    series_key: r.exchange + "_" + r.market_type + "_" + r.symbol + "_" + r.timeframe,
                    timestamp: r._time
                }}))
                |> yield(name: "latest_timestamps")
            '''
            
            # Use query() instead of query_data_frame() to avoid pandas in async context
            tables = self.query_api.query(query, org=self.org)
            
            last_ts_cache = {}
            for table in tables:
                for record in table.records:
                    series_key = record.values.get('series_key')
                    timestamp = record.values.get('timestamp')
                    
                    if series_key and timestamp:
                        # Convert nanoseconds to milliseconds
                        timestamp_ms = int(timestamp.timestamp() * 1000)
                        last_ts_cache[series_key] = timestamp_ms
            
            logger.info(f"Retrieved {len(last_ts_cache)} series timestamps from InfluxDB")
            return last_ts_cache
            
        except Exception as e:
            logger.error(f"Error querying latest timestamps: {e}")
            return {}

    def get_existing_timestamps(self, exchange: str, market_type: str, symbol: str, 
                            timeframe: str, start_ms: int, end_ms: int) -> List[int]:
        """Get existing timestamps for a series within a time range"""
        if not self.client:
            return []
            
        try: 
            query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: {start_ms}ms, stop: {end_ms}ms)
            |> filter(fn: (r) => r._measurement == "ohlcv")
            |> filter(fn: (r) => r.exchange == "{exchange.lower()}")
            |> filter(fn: (r) => r.market_type == "{market_type.lower()}")
            |> filter(fn: (r) => r.symbol == "{symbol.upper()}")
            |> filter(fn: (r) => r.timeframe == "{timeframe}")
            |> filter(fn: (r) => r._field == "open")
            |> keep(columns: ["_time"])
            |> yield(name: "existing_timestamps")
            '''
            
            # Use query() instead of query_data_frame() to avoid pandas in async context
            tables = self.query_api.query(query)
            timestamps = []
            
            for table in tables:
                for record in table.records:
                    timestamp = record.get_time()
                    if timestamp:
                        timestamp_ms = int(timestamp.timestamp() * 1000)
                        timestamps.append(timestamp_ms)
            
            return sorted(timestamps)
            
        except Exception as e:
            logger.error(f"Error getting existing timestamps: {e}")
            return []

    def write_ohlcv_batch(self, series_data: List[Dict], dedup_cache: Dict[str, int]) -> bool:
        """write with improved batching and cache consistency"""
        if not self.client or not series_data:
            return False
        
        try:
            points = []
            updates_made = False
            cache_updates = {}  # Track cache updates for atomic operation
            
            for series_entry in series_data:
                exchange = series_entry['exchange'].lower()
                market_type = series_entry['market_type'].lower()
                symbol = series_entry['symbol'].upper()
                timeframe = series_entry['timeframe']
                ohlcv_data = series_entry['ohlcv_data']
                
                series_key = f"{exchange}_{market_type}_{symbol}_{timeframe}"
                last_ts_ms = dedup_cache.get(series_key, 0)
                
                # Filter and sort data
                valid_candles = []
                for candle in ohlcv_data:
                    candle_ts_ms = int(candle.get('timestamp', 0))
                    
                    # deduplication with strict ordering
                    if candle_ts_ms > last_ts_ms:
                        valid_candles.append((candle_ts_ms, candle))
                
                # Sort ascending by timestamp (maintain monotonicity)
                valid_candles.sort(key=lambda x: x[0])
                
                # Create points with validation
                for candle_ts_ms, candle in valid_candles:
                    try:
                        # Convert to nanoseconds for InfluxDB
                        timestamp_ns = candle_ts_ms * 1_000_000
                        
                        point = (
                            Point("ohlcv")
                            .tag("exchange", exchange)
                            .tag("market_type", market_type)
                            .tag("symbol", symbol)
                            .tag("timeframe", timeframe)
                            .time(timestamp_ns)
                        )
                        
                        # Add OHLCV fields with validation
                        valid_point = True
                        for field in ['open', 'high', 'low', 'close', 'volume']:
                            value = candle.get(field, 0)
                            try:
                                if value is not None and str(value) != 'nan':
                                    float_value = float(value)
                                    if not pd.isna(float_value) and abs(float_value) < 1e15 and float_value >= 0:
                                        point = point.field(field, float_value)
                                    else:
                                        # Invalid value - skip this point
                                        valid_point = False
                                        break
                                else:
                                    point = point.field(field, 0.0)
                            except (ValueError, TypeError, OverflowError):
                                valid_point = False
                                break
                        
                        if valid_point:
                            points.append(point)
                            
                            # Track cache update (but don't apply yet)
                            cache_updates[series_key] = max(
                                cache_updates.get(series_key, last_ts_ms), 
                                candle_ts_ms
                            )
                            updates_made = True
                        
                    except Exception as candle_error:
                        logger.warning(f"Error processing candle for {series_key}: {candle_error}")
                        continue
            
            if points:
                # Use batching for better performance
                if len(points) > self.batch_size:
                    # Split large batches
                    for i in range(0, len(points), self.batch_size):
                        batch = points[i:i + self.batch_size]
                        self._queue_points_for_write(batch)
                else:
                    self._queue_points_for_write(points)
                
                # Atomic cache update only after successful queueing
                for series_key, timestamp in cache_updates.items():
                    dedup_cache[series_key] = timestamp
                
                logger.debug(f"Queued {len(points)} OHLCV points for InfluxDB write")
                
            return updates_made
            
        except Exception as e:
            logger.error(f"Error in OHLCV batch write: {e}")
            return False

    def _queue_points_for_write(self, points: List[Point]):
        """FIX: Improved queueing with backpressure handling"""
        queued_count = 0
        dropped_count = 0
        
        for point in points:
            try:
                if not self.write_queue.full():
                    self.write_queue.put(point, block=False)
                    queued_count += 1
                else:
                    # Queue is full - implement backpressure
                    # Try to wait a short time for queue to drain
                    try:
                        self.write_queue.put(point, block=True, timeout=1.0)
                        queued_count += 1
                    except queue.Full:
                        dropped_count += 1
                        self.write_stats['dropped_points'] += 1
                        
                        # Log every 100th drop to avoid spam
                        if self.write_stats['dropped_points'] % 100 == 0:
                            logger.warning(f"Write queue backpressure: dropped {self.write_stats['dropped_points']} points total")
                        
            except Exception as e:
                logger.error(f"Error queueing point for write: {e}")
                dropped_count += 1
        
        if dropped_count > 0:
            logger.warning(f"Queued {queued_count} points, dropped {dropped_count} due to queue pressure")

    def check_data_exists(self, exchange: str, market_type: str, symbol: str, timeframe: str) -> bool:
        """Data existence check with better performance"""
        if not self.client:
            return False
        
        try:
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: -90d)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.exchange == "{exchange.lower()}")
                |> filter(fn: (r) => r.market_type == "{market_type.lower()}")
                |> filter(fn: (r) => r.symbol == "{symbol.upper()}")
                |> filter(fn: (r) => r.timeframe == "{timeframe}")
                |> limit(n: 1)
                |> count()
                |> yield(name: "existence_check")
            '''
            
            # Use query() instead of query_data_frame() to avoid pandas in async context
            tables = self.query_api.query(query, org=self.org)
            
            for table in tables:
                for record in table.records:
                    count = record.get_value()
                    if count and count > 0:
                        logger.debug(f"Data existence check for {exchange}/{symbol}/{timeframe}: True")
                        return True
            
            logger.debug(f"Data existence check for {exchange}/{symbol}/{timeframe}: False")
            return False
            
        except Exception as e:
            logger.error(f"Error checking data existence: {e}")
            return False

    def get_series_range(self, exchange: str, market_type: str, symbol: str, timeframe: str) -> Tuple[Optional[int], Optional[int]]:
        """Series range query with better error handling"""
        if not self.client:
            return None, None
        
        try:
            query = f'''
                data = from(bucket: "{self.bucket}")
                |> range(start: -90d)
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.exchange == "{exchange.lower()}")
                |> filter(fn: (r) => r.market_type == "{market_type.lower()}")
                |> filter(fn: (r) => r.symbol == "{symbol.upper()}")
                |> filter(fn: (r) => r.timeframe == "{timeframe}")
                |> filter(fn: (r) => r._field == "close")

                earliest = data |> first() |> set(key: "bound", value: "earliest")
                latest = data |> last() |> set(key: "bound", value: "latest")
                
                union(tables: [earliest, latest])
                |> yield(name: "series_range")
            '''
            
            # Use query() instead of query_data_frame() to avoid pandas in async context
            tables = self.query_api.query(query, org=self.org)
            
            earliest_ms = None
            latest_ms = None
            
            for table in tables:
                for record in table.records:
                    timestamp = record.get_time()
                    bound = record.values.get('bound')
                    
                    if timestamp and bound:
                        timestamp_ms = int(timestamp.timestamp() * 1000)
                        if bound == 'earliest':
                            earliest_ms = timestamp_ms
                        elif bound == 'latest':
                            latest_ms = timestamp_ms
            
            logger.debug(f"Series range for {exchange}/{symbol}/{timeframe}: {earliest_ms} to {latest_ms}")
            return earliest_ms, latest_ms
            
        except Exception as e:
            logger.error(f"Error getting series range: {e}")
            return None, None

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        uptime = time.time() - self.write_stats.get('start_time', time.time())
        
        stats = {
            **self.write_stats,
            'queue_size': self.write_queue.qsize(),
            'consecutive_failures': self.consecutive_failures,
            'current_backoff_time': self.backoff_time,
            'batch_thread_alive': self.batch_thread.is_alive() if self.batch_thread else False,
            'uptime_seconds': uptime,
        }
        
        # Calculate rates
        if uptime > 0:
            stats['points_per_second'] = self.write_stats['total_points'] / uptime
            stats['batches_per_minute'] = (self.write_stats['successful_batches'] * 60) / uptime
        
        return stats

    def flush_pending_writes(self, timeout: float = 30.0):
        """Force flush all pending writes with timeout"""
        if not self.write_queue.empty():
            logger.info(f"Flushing {self.write_queue.qsize()} pending writes...")
            
            start_time = time.time()
            initial_queue_size = self.write_queue.qsize()
            
            while not self.write_queue.empty() and (time.time() - start_time) < timeout:
                time.sleep(0.1)  # Small sleep to allow batch processor to work
            
            remaining = self.write_queue.qsize()
            flushed = initial_queue_size - remaining
            
            logger.info(f"Flushed {flushed} writes, {remaining} remaining")
            
            if remaining > 0:
                logger.warning(f"Timeout reached, {remaining} writes still pending")

    def close(self):
        """cleanup with proper resource management"""
        try:
            logger.info("Shutting down InfluxDB Manager...")
            
            # Stop batch thread
            self.batch_thread_running = False
            if self.write_queue and not self.write_queue.full():
                self.write_queue.put(None, block=False)  # Shutdown signal
            
            # Wait for batch thread to finish
            if self.batch_thread and self.batch_thread.is_alive():
                self.batch_thread.join(timeout=10.0)
                if self.batch_thread.is_alive():
                    logger.warning("Batch thread did not shut down gracefully")
            
            # Flush any remaining writes
            self.flush_pending_writes(timeout=15.0)
            
            # Close write API
            if self.write_api:
                try:
                    self.write_api.close()
                except Exception as e:
                    logger.error(f"Error closing write API: {e}")
            
            # Close client
            if self.client:
                try:
                    self.client.close()
                except Exception as e:
                    logger.error(f"Error closing InfluxDB client: {e}")
            
            # Shutdown executor
            if self._executor:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except Exception as e:
                    logger.error(f"Error shutting down executor: {e}")
            
            # Log final stats
            stats = self.get_performance_stats()
            logger.info(f"Final InfluxDB stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error during InfluxDB manager cleanup: {e}")
        
        logger.info("InfluxDB Manager shutdown complete")