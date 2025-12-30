from datetime import datetime, timedelta
from django.utils import timezone
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from myapp.services.series_orchestrator import SeriesOrchestrator
from myapp.models import WorkerShard, SeriesConfig
import asyncio
import signal
import sys
import logging
from datetime import timezone as timezoneDt
import time
 
logger = logging.getLogger(__name__) 
 
class Command(BaseCommand):
    help = 'Run the trading data orchestrator following the specification flow'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--worker-id',
            type=str,
            help='Specific worker ID to run (if not provided, will be auto-generated)'
        )
        parser.add_argument(
            '--shard-name',
            type=str,
            help='Specific shard name to process (e.g., shard_0, shard_1)'
        )
        parser.add_argument(
            '--phase',
            choices=['backfill', 'stream', 'both'],
            default='both',
            help='Which phase to run: backfill, stream, or both (default: both)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Dry run mode - show what would be processed without actual execution'
        )
        parser.add_argument(
            '--max-series',
            type=int,
            help='Maximum number of series to process (for testing)'
        )
        parser.add_argument(
            '--timeframes',
            type=str,
            help='Comma-separated list of timeframes to process (e.g., 1h,1d)'
        )
        parser.add_argument(
            '--exchanges',
            type=str,  
            help='Comma-separated list of exchanges to process (e.g., binance,okx)'
        )
        parser.add_argument(
            '--status',
            action='store_true',
            help='Show current system status and exit'
        )
        parser.add_argument(
            '--list-shards',
            action='store_true',
            help='List all available worker shards and exit'
        )
        parser.add_argument(
            '--reset-backfill',
            action='store_true',
            help='Reset backfill_completed flags for testing'
        )
        parser.add_argument(
            '--debug-series',
            action='store_true',
            help='Show detailed series status and exit'
        )
        parser.add_argument(
        '--earliest-date',
        type=str,
        help='Override earliest fetch date for all symbols (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)'
        )
        parser.add_argument(
            '--resume-backfill',
            action='store_true',
            help='Automatically backfill from resume points before streaming'
        )
    
    def handle(self, *args, **options):
        """Main command handler"""
        
        # Handle status and utility commands
        if options['status']:
            self.show_system_status()
            return
            
        if options['list_shards']:
            self.list_worker_shards()
            return
            
        if options['debug_series']:
            self.show_series_status()
            return
            
        if options['reset_backfill']:
            self.reset_backfill_flags()
            return
        
        # Validate shard exists if specified
        if options['shard_name']:
            if not self.validate_shard(options['shard_name']):
                raise CommandError(f"Shard '{options['shard_name']}' not found")
        
        try:
            # Run the orchestrator
            asyncio.run(self.run_orchestrator(options))
            
        except KeyboardInterrupt:
            self.stdout.write('\nInterrupted by user')
        except Exception as e:
            self.stderr.write(f'Orchestrator failed: {e}')
            raise CommandError(f'Orchestrator failed: {e}')
    
    def show_system_status(self):
        """Show current system status"""
        self.stdout.write(self.style.HTTP_INFO('=== TRADING ORCHESTRATOR STATUS ==='))
        
        # Worker status
        workers = WorkerShard.objects.filter(isactive=1, isdeleted=0)
        self.stdout.write(f'\nWorkers: {workers.count()}')
        
        for worker in workers:
            heartbeat_age = 'Never'
            if worker.last_heartbeat:
                age_seconds = (timezone.now() - worker.last_heartbeat).total_seconds()
                heartbeat_age = f'{int(age_seconds)}s ago'
            
            self.stdout.write(f'  {worker.shard_name}: {worker.status} ({worker.series_count} series, heartbeat: {heartbeat_age})')
        
        # Series status
        series_total = SeriesConfig.objects.filter(isactive=1, isdeleted=0).count()
        series_backfilled = SeriesConfig.objects.filter(isactive=1, isdeleted=0, backfill_completed=True).count()
        series_errors = SeriesConfig.objects.filter(isactive=1, isdeleted=0).exclude(last_error__isnull=True).count()
        
        self.stdout.write(f'\nSeries Configurations:')
        self.stdout.write(f'  Total: {series_total}')
        if series_total > 0:
            self.stdout.write(f'  Backfill Completed: {series_backfilled} ({series_backfilled/series_total*100:.1f}%)')
        self.stdout.write(f'  With Errors: {series_errors}')
        
        # Exchange breakdown
        from django.db import models
        exchange_stats = SeriesConfig.objects.filter(
            isactive=1, isdeleted=0
        ).values('exchange__name').annotate(
            total=models.Count('seriesconfigid'),
            completed=models.Count('seriesconfigid', filter=models.Q(backfill_completed=True))
        )
        
        self.stdout.write(f'\nBy Exchange:')
        for stat in exchange_stats:
            completion = stat['completed'] / stat['total'] * 100 if stat['total'] > 0 else 0
            self.stdout.write(f'  {stat["exchange__name"]}: {stat["completed"]}/{stat["total"]} ({completion:.1f}%)')
    
    def show_series_status(self):
        """Show detailed series status for debugging"""
        self.stdout.write(self.style.HTTP_INFO('=== SERIES STATUS DEBUG ==='))
        
        total_series = SeriesConfig.objects.filter(isactive=1, isdeleted=0).count()
        completed_series = SeriesConfig.objects.filter(
            isactive=1, 
            isdeleted=0, 
            backfill_completed=True
        ).count()
        pending_series = total_series - completed_series
        
        self.stdout.write(f"Total active series: {total_series}")
        self.stdout.write(f"Backfill completed: {completed_series}")
        self.stdout.write(f"Backfill pending: {pending_series}")
        
        # Show some examples of pending series
        if pending_series > 0:
            self.stdout.write("\nPending series examples:")
            pending_examples = SeriesConfig.objects.filter(
                isactive=1, 
                isdeleted=0, 
                backfill_completed=False
            )[:10]
            
            for config in pending_examples:
                self.stdout.write(f"  {config.series_key} (Exchange: {config.exchange.name}, Symbol: {config.symbol})")
        else:
            self.stdout.write("\nCompleted series examples:")
            completed_examples = SeriesConfig.objects.filter(
                isactive=1, 
                isdeleted=0, 
                backfill_completed=True
            )[:10]
            
            for config in completed_examples:
                last_ts = "Never"
                if config.last_ts_ms:
                    from datetime import datetime
                    last_ts = datetime.fromtimestamp(config.last_ts_ms/1000).strftime('%Y-%m-%d %H:%M')
                self.stdout.write(f"  {config.series_key} (Last: {last_ts})")
        
        # Show series with errors
        error_series = SeriesConfig.objects.filter(
            isactive=1, 
            isdeleted=0
        ).exclude(last_error__isnull=True).exclude(last_error='')
        
        if error_series.exists():
            self.stdout.write(f"\nSeries with errors: {error_series.count()}")
            for config in error_series[:5]:
                self.stdout.write(f"  {config.series_key}: {config.last_error[:100]}")
    
    def reset_backfill_flags(self):
        """Reset backfill_completed flags for testing"""
        self.stdout.write("Resetting backfill flags for testing...")
        
        # Get the series to reset first, then update them
        series_to_reset = SeriesConfig.objects.filter(
            isactive=1, 
            isdeleted=0,
        )[:5]  # Only first 5 matching series
        
        # Get the IDs to update
        series_ids = [config.seriesconfigid for config in series_to_reset]
        
        if not series_ids:
            self.stdout.write("No matching series found to reset")
            return
        
        # Now update using the IDs
        count = SeriesConfig.objects.filter(
            seriesconfigid__in=series_ids
        ).update(
            backfill_completed=False,
            probing_completed=False,
            last_error=None,
            earliest_start_ms=None,  # Also reset this for fresh probing
            last_ts_ms=None,
            last_backfill_ms=None
        )
        
        self.stdout.write(f"Reset backfill flags for {count} series:")
        for config in series_to_reset:
            self.stdout.write(f"  - {config.series_key} ({config.exchange.name} {config.symbol})")
        
        self.stdout.write("\nNow run: python manage.py run_trading_orchestrator --phase=backfill --max-series=1")
    
    def list_worker_shards(self):
        """List all available worker shards"""
        workers = WorkerShard.objects.filter(isactive=1, isdeleted=0).order_by('shard_name')
        
        if not workers.exists():
            self.stdout.write(self.style.WARNING('No worker shards found. Run "python manage.py init_trading" first.'))
            return
        
        self.stdout.write(self.style.HTTP_INFO('=== AVAILABLE WORKER SHARDS ==='))
        
        for worker in workers:
            series_count = len(worker.assigned_series) if worker.assigned_series else 0
            self.stdout.write(f'Shard: {worker.shard_name}')
            self.stdout.write(f'  Worker ID: {worker.worker_id}') 
            self.stdout.write(f'  Status: {worker.status}')
            self.stdout.write(f'  Series Count: {series_count}')
            self.stdout.write(f'  Progress: {worker.backfill_progress:.1f}%')
            if worker.last_heartbeat:
                self.stdout.write(f'  Last Heartbeat: {worker.last_heartbeat}')
            self.stdout.write('')
    
    def validate_shard(self, shard_name: str) -> bool:
        """Validate that shard exists"""
        return WorkerShard.objects.filter(
            shard_name=shard_name,
            isactive=1,
            isdeleted=0
        ).exists()
    
    async def run_orchestrator(self, options):
        """Run the orchestrator asynchronously"""
        worker_id = options.get('worker_id') or f"orchestrator_{int(time.time())}"
        shard_name = options.get('shard_name')
        phase = options.get('phase')
        dry_run = options.get('dry_run')
        max_series = options.get('max_series')
        earliest_date = options.get('earliest_date')
        
        # Parse filters
        timeframes_filter = None
        if options.get('timeframes'):
            timeframes_filter = [tf.strip() for tf in options['timeframes'].split(',')]
        
        exchanges_filter = None
        if options.get('exchanges'):
            exchanges_filter = [ex.strip().lower() for ex in options['exchanges'].split(',')]
        
        # Parse earliest date if provided
        earliest_date_ms = None
        if earliest_date:
            try:
                earliest_date_ms = self.parse_earliest_date(earliest_date)
                self.stdout.write(f'Earliest Date Override: {earliest_date} ({datetime.fromtimestamp(earliest_date_ms/1000)} UTC)')
            except ValueError as e:
                raise CommandError(f'Invalid earliest-date format: {e}')
        
        self.stdout.write(f'Starting Trading Orchestrator')
        self.stdout.write(f'Worker ID: {worker_id}')
        self.stdout.write(f'Shard: {shard_name or "auto-select"}')
        self.stdout.write(f'Phase: {phase}')
        self.stdout.write(f'Max Series: {max_series or "unlimited"}')
        if timeframes_filter:
            self.stdout.write(f'Timeframes: {", ".join(timeframes_filter)}')
        if exchanges_filter:
            self.stdout.write(f'Exchanges: {", ".join(exchanges_filter)}')
        
        # Create orchestrator with earliest date override
        orchestrator = SeriesOrchestrator(worker_id, shard_name, max_series)
        if earliest_date_ms:
            orchestrator.earliest_date_override_ms = earliest_date_ms
        
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            self.stdout.write(f'\nReceived signal {sig}, shutting down gracefully...')
            orchestrator.is_running = False
            sys.exit(0)
        
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Use the optimized initialization with filters applied EARLY
            self.stdout.write('Initializing orchestrator with filters...')
            
            if timeframes_filter or exchanges_filter or max_series:
                # Use the optimized filtered initialization
                success = await orchestrator.initialize_with_filters(
                    timeframes_filter=timeframes_filter,
                    exchanges_filter=exchanges_filter,
                    max_series=max_series
                )
            else:
                # Use standard initialization if no filters
                success = await orchestrator.initialize()
            
            if not success:
                raise CommandError('Failed to initialize orchestrator')
            
            series_count = len(orchestrator.series_configs)
            self.stdout.write(f'Orchestrator initialized with {series_count} series configurations')
            
            if series_count == 0:
                self.stdout.write(self.style.WARNING('No series match the specified filters'))
                return
            
            if dry_run:
                self.stdout.write(self.style.WARNING('DRY RUN MODE - No actual data collection will occur'))
                await self.show_orchestrator_plan(orchestrator, phase)
                return
            
            # Update worker heartbeat
            await self.update_worker_status(worker_id, shard_name, 'initializing', series_count)
            
            # Handle stream-only mode by checking if backfill is actually needed
            if phase == 'stream':
                # For stream-only mode, check if all series have completed backfill
                incomplete_series = [
                    config for config in orchestrator.series_configs.values()
                    if not config.backfill_completed
                ]
                
                if incomplete_series:
                    self.stdout.write(self.style.WARNING(f'Found {len(incomplete_series)} series that need backfill:'))
                    for config in incomplete_series[:5]:  # Show first 5
                        self.stdout.write(f'  - {config.series_key}')
                    if len(incomplete_series) > 5:
                        self.stdout.write(f'  ... and {len(incomplete_series) - 5} more')
                    
                    user_input = input('\nRun backfill for these series first? [y/N]: ')
                    if user_input.lower() in ['y', 'yes']:
                        phase = 'both'  # Switch to both phases
                        self.stdout.write('Switching to both phases...')
                    else:
                        self.stdout.write('Continuing with stream-only mode (may not receive data for incomplete series)')
                        # Force backfill_phase_complete for stream-only mode
                        orchestrator.backfill_phase_complete = True
                else:
                    # All series are backfilled, safe to stream
                    orchestrator.backfill_phase_complete = True
                    self.stdout.write(f'All {series_count} series have completed backfill - ready for streaming')
            
            # Run phases based on selection
            if phase in ['backfill', 'both']:
                self.stdout.write(self.style.SUCCESS('=== Starting Backfill Phase ==='))
                await self.update_worker_status(worker_id, shard_name, 'backfilling', series_count)
                
                start_time = time.time()
                await orchestrator.run_backfill_phase()
                duration = time.time() - start_time
                
                self.stdout.write(self.style.SUCCESS(f'Backfill phase completed in {duration:.1f} seconds'))
            
            if phase in ['stream', 'both']:
                self.stdout.write(self.style.SUCCESS('=== Starting Streaming Phase ==='))
                await self.update_worker_status(worker_id, shard_name, 'streaming', series_count)
                
                # Streaming runs indefinitely until interrupted
                await orchestrator.run_streaming_phase()
            
        except Exception as e:
            await self.update_worker_status(worker_id, shard_name, 'error', 0)
            self.stderr.write(f'Orchestrator error: {e}')
            logger.exception("Orchestrator error details:")
            raise
        finally:
            await self.update_worker_status(worker_id, shard_name, 'idle', 0)
            await orchestrator.cleanup()
            self.stdout.write('Orchestrator cleanup completed')
            
    def parse_earliest_date(self, date_str: str) -> int:
        """Parse earliest date string to milliseconds"""
        from datetime import datetime
        import re
        
        # Clean the input
        date_str = date_str.strip()
        
        # Try different formats
        formats = [
            '%Y-%m-%d',                    # 2023-01-01
            '%Y-%m-%d %H:%M:%S',          # 2023-01-01 12:00:00
            '%Y-%m-%d %H:%M',             # 2023-01-01 12:00
            '%Y/%m/%d',                   # 2023/01/01
            '%Y/%m/%d %H:%M:%S',          # 2023/01/01 12:00:00
            '%m/%d/%Y',                   # 01/01/2023
            '%m/%d/%Y %H:%M:%S',          # 01/01/2023 12:00:00
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                # Assume UTC if no timezone specified
                dt = dt.replace(tzinfo=timezoneDt.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
        
        # Try relative dates
        if date_str.lower().endswith('d'):  # e.g., "30d" = 30 days ago
            try:
                days = int(date_str[:-1])
                dt = timezone.now() - timedelta(days=days)
                return int(dt.timestamp() * 1000)
            except ValueError:
                pass
        
        raise ValueError(f"Unable to parse date: {date_str}. Use format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
    
    async def show_orchestrator_plan(self, orchestrator, phase):
        """Show what the orchestrator would do in dry run mode"""
        self.stdout.write('=== ORCHESTRATOR EXECUTION PLAN ===')
        
        series_by_exchange = {}
        backfill_needed = 0
        streaming_ready = 0
        
        for config in orchestrator.series_configs.values():
            exchange_key = f"{config.exchange.name}_{config.market_type}"
            if exchange_key not in series_by_exchange:
                series_by_exchange[exchange_key] = []
            series_by_exchange[exchange_key].append(config)
            
            if not config.backfill_completed:
                backfill_needed += 1
            else:
                streaming_ready += 1
        
        for exchange_key, configs in series_by_exchange.items():
            self.stdout.write(f'\n{exchange_key.upper()}:')
            
            # Group by timeframe
            by_timeframe = {}
            for config in configs:
                if config.timeframe not in by_timeframe:
                    by_timeframe[config.timeframe] = []
                by_timeframe[config.timeframe].append(config.symbol)
            
            for timeframe, symbols in by_timeframe.items():
                self.stdout.write(f'  {timeframe}: {len(symbols)} symbols ({", ".join(symbols[:5])}{"..." if len(symbols) > 5 else ""})')
        
        self.stdout.write(f'\n=== PHASE SUMMARY ===')
        if phase in ['backfill', 'both']:
            self.stdout.write(f'Backfill Phase: {backfill_needed} series need backfilling')
        
        if phase in ['stream', 'both']:
            self.stdout.write(f'Streaming Phase: {streaming_ready} series ready for streaming')
    
    async def update_worker_status(self, worker_id: str, shard_name: str, status: str, series_count: int):
        """Update worker status in database"""
        from asgiref.sync import sync_to_async
        from django.utils import timezone
        
        @sync_to_async
        def update_status():
            if shard_name:
                try:
                    worker = WorkerShard.objects.get(shard_name=shard_name)
                    worker.status = status
                    worker.worker_id = worker_id
                    worker.series_count = series_count
                    worker.last_heartbeat = timezone.now()
                    worker.save()
                except WorkerShard.DoesNotExist:
                    pass
        
        await update_status()