from celery import shared_task
from django.utils import timezone
import logging
logger = logging.getLogger(__name__)


@shared_task
def initialize_series_configs():
    """Initialize series configurations and assign to workers"""
    from myapp.models import Exchange, TradingPair, SeriesConfig, WorkerShard
    from django.conf import settings
    import hashlib
     
    try: 
        # Get all active exchanges and pairs
        exchanges = Exchange.objects.filter(isactive=1, isdeleted=0)
        trading_pairs = TradingPair.objects.filter(isactive=1, isdeleted=0).select_related('exchange')
        
        timeframes = getattr(settings, 'TRADING_CONFIG', {}).get('SUPPORTED_TIMEFRAMES', ['5m', '1h', '1d'])
        created_count = 0
        updated_count = 0
         
        # Create/update series configurations
        for pair in trading_pairs:
            for timeframe in timeframes:
                series_config, created = SeriesConfig.objects.get_or_create(
                    exchange=pair.exchange,
                    market_type=pair.exchange.market_type or 'spot',
                    symbol=pair.symbol,
                    timeframe=timeframe,
                    defaults={
                        'earliest_start_ms': None,
                        'last_backfill_ms': None,
                        'last_ts_ms': None,
                        'probing_completed': False,
                        'backfill_completed': False,
                        'isactive': 1,
                        'isdeleted': 0
                    }
                )
                
                if created:
                    created_count += 1
                else:
                    # Update active status if needed
                    if not series_config.isactive:
                        series_config.isactive = 1
                        series_config.save()
                        updated_count += 1
        
        # Assign series to worker shards using stable hashing
        series_configs = SeriesConfig.objects.filter(isactive=1, isdeleted=0)
        
        # Create default worker shards if none exist
        default_shards = 3
        for i in range(default_shards):
            shard_name = f"shard_{i}"
            WorkerShard.objects.get_or_create(
                shard_name=shard_name,
                defaults={
                    'worker_id': f"worker_{shard_name}",
                    'status': 'idle',
                    'assigned_series': [],
                    'series_count': 0,
                    'isactive': 1,
                    'isdeleted': 0
                }
            )
        
        # Distribute series across shards
        shards = list(WorkerShard.objects.filter(isactive=1, isdeleted=0))
        shard_assignments = {shard.shardid: [] for shard in shards}
        
        for config in series_configs:
            # Use stable hash to assign series to shard
            series_key = f"{config.exchange.exchangeid}_{config.market_type}_{config.symbol}_{config.timeframe}"
            hash_value = int(hashlib.md5(series_key.encode()).hexdigest(), 16)
            shard_index = hash_value % len(shards)
            shard_id = shards[shard_index].shardid
            shard_assignments[shard_id].append(series_key)
        
        # Update shard assignments
        for shard in shards:
            assigned_series = shard_assignments[shard.shardid]
            shard.assigned_series = assigned_series
            shard.series_count = len(assigned_series)
            shard.save()
        
        logger.info(f"Series initialization completed: {created_count} created, {updated_count} updated")
        return f"Initialized {created_count} new series, updated {updated_count}, distributed across {len(shards)} shards"
        
    except Exception as e:
        logger.error(f"Error initializing series configs: {e}")
        return f"Error: {e}"

@shared_task
def monitor_series_health():
    """Monitor series health and performance"""
    from myapp.models import SeriesConfig, WorkerShard
    from myapp.services.influx_manager import InfluxManager
    import time
    
    try:
        influx_manager = InfluxManager()
        if not influx_manager.client:
            return "InfluxDB not available for monitoring"
        
        current_time_ms = int(time.time() * 1000)
        alerts = []
        
        # Check for stale series (no data in last 3x timeframe)
        active_series = SeriesConfig.objects.filter(
            isactive=1,
            isdeleted=0,
            backfill_completed=True
        )
        
        stale_threshold = {
            '1m': 3 * 60 * 1000,      # 3 minutes
            '5m': 15 * 60 * 1000,     # 15 minutes
            '15m': 45 * 60 * 1000,    # 45 minutes
            '1h': 3 * 60 * 60 * 1000, # 3 hours
            '1d': 3 * 24 * 60 * 60 * 1000, # 3 days
        }
        
        stale_series = []
        for config in active_series:
            if config.last_ts_ms:
                age_ms = current_time_ms - config.last_ts_ms
                threshold = stale_threshold.get(config.timeframe, 60 * 60 * 1000)  # Default 1 hour
                
                if age_ms > threshold:
                    stale_series.append({
                        'series_key': config.series_key,
                        'age_minutes': age_ms // (60 * 1000),
                        'exchange': config.exchange.name,
                        'symbol': config.symbol,
                        'timeframe': config.timeframe
                    })
        
        if stale_series:
            alerts.append(f"Found {len(stale_series)} stale series")
        
        # Check worker shard health
        stale_workers = []
        workers = WorkerShard.objects.filter(isactive=1, isdeleted=0)
        
        for worker in workers:
            if worker.last_heartbeat:
                age = timezone.now() - worker.last_heartbeat
                if age.total_seconds() > 300:  # 5 minutes
                    stale_workers.append(worker.worker_id)
        
        if stale_workers:
            alerts.append(f"Found {len(stale_workers)} stale workers: {stale_workers}")
        
        # Check incomplete backfills
        incomplete_backfills = SeriesConfig.objects.filter(
            isactive=1,
            isdeleted=0,
            backfill_completed=False
        ).count()
        
        if incomplete_backfills > 0:
            alerts.append(f"Found {incomplete_backfills} incomplete backfills")
        
        result = {
            'timestamp': timezone.now().isoformat(),
            'alerts': alerts,
            'stale_series_count': len(stale_series),
            'stale_workers_count': len(stale_workers),
            'incomplete_backfills': incomplete_backfills,
            'status': 'healthy' if not alerts else 'degraded'
        }
        
        logger.info(f"Series health check: {result['status']} - {len(alerts)} alerts")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in series health monitoring: {e}")
        return {'status': 'error', 'message': str(e)}

    
@shared_task(queue='maintenance')
def detect_and_report_gaps():
    """Periodic gap detection and reporting - MAINTENANCE ONLY"""
    from myapp.models import SeriesConfig
    from myapp.services.influx_manager import InfluxManager
    
    try:
        influx_manager = InfluxManager()
        active_series = SeriesConfig.objects.filter(
            isactive=1,
            isdeleted=0,
            backfill_completed=True
        )
        
        gaps_detected = 0
        for config in active_series[:10]:  # Limit to avoid overload
            # Check for gaps in last 24 hours
            gaps = influx_manager.detect_gaps(
                config.exchange.name,
                config.market_type,
                config.symbol,
                config.timeframe,
                hours_back=24
            )
            
            if gaps:
                gaps_detected += len(gaps)
                logger.warning(f"Gaps detected in {config.series_key}: {len(gaps)} gaps")
        
        return f"Gap detection completed: {gaps_detected} gaps found"
        
    except Exception as e:
        logger.error(f"Error in gap detection: {e}")
        return f"Error: {e}"
    
