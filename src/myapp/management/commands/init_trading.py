from django.core.management.base import BaseCommand
from django.utils import timezone
from myapp.models import (
    Exchange, TradingPair, SeriesConfig, WorkerShard
)
from django.conf import settings
import hashlib
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand): 
    help = 'Initialize trading system with exchanges, series configs, and worker shards'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--reset',
            action='store_true',
            help='Reset existing data before initialization',
        )
        parser.add_argument(
            '--workers',
            type=int,
            default=3,
            help='Number of worker shards to create (default: 3)',
        )
        parser.add_argument(
            '--exchanges-only',
            action='store_true',
            help='Only initialize exchanges and trading pairs',
        )
        parser.add_argument(
            '--series-only',
            action='store_true',
            help='Only initialize series configurations',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be created without actually creating it',
        )
     
    def handle(self, *args, **options):
        self.dry_run = options['dry_run']
        
        if self.dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No actual changes will be made'))
        
        if options['reset'] and not self.dry_run:
            self.stdout.write('Resetting existing trading data...')
            self.reset_data()
        
        if not options['series_only']:
            # Create exchanges and trading pairs
            self.create_exchanges()
            self.create_trading_pairs()
        
        if not options['exchanges_only']:
            # Create series configurations and worker shards
            self.create_series_configurations()
            self.create_worker_shards(options['workers'])
        
        if self.dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN COMPLETED - No changes were made'))
        else:
            self.stdout.write(self.style.SUCCESS('Successfully initialized trading system'))
        
    def reset_data(self):
        """Reset existing trading data"""
        models_to_reset = [
            ('SeriesConfig', SeriesConfig),
            ('WorkerShard', WorkerShard),
            ('TradingPair', TradingPair),
            ('Exchange', Exchange),
        ]
        
        for model_name, model_class in models_to_reset:
            count = model_class.objects.count()
            if count > 0:
                model_class.objects.all().delete()
                self.stdout.write(f'Deleted {count} {model_name} records')
        
    def create_exchanges(self):
        """Create supported exchanges with market types based on test data"""
        exchanges_data = [
            # Binance
            {
                'name': 'binance',
                'display_name': 'Binance Spot',
                'market_type': 'spot',
            },
            {
                'name': 'binance',
                'display_name': 'Binance USD-M Futures',
                'market_type': 'swap',
            },
            # Binance Coin-M
            {
                'name': 'binancecoinm',
                'display_name': 'Binance Coin-M Futures',
                'market_type': 'swap',
            },
            # Binance USD-M
            {
                'name': 'binanceusdm',
                'display_name': 'Binance USD-M Perpetual',
                'market_type': 'swap',
            },
            # KuCoin
            {
                'name': 'kucoin',
                'display_name': 'KuCoin Spot',
                'market_type': 'spot',
            },
            # KuCoin Futures
            {
                'name': 'kucoinfutures',
                'display_name': 'KuCoin Futures',
                'market_type': 'swap',
            },
            # Bitget
            {
                'name': 'bitget',
                'display_name': 'Bitget Spot',
                'market_type': 'spot',
            },
            {
                'name': 'bitget',
                'display_name': 'Bitget Futures',
                'market_type': 'swap',
            },
            # Bybit
            {
                'name': 'bybit',
                'display_name': 'Bybit Spot',
                'market_type': 'spot',
            },
            {
                'name': 'bybit',
                'display_name': 'Bybit Futures',
                'market_type': 'swap',
            },
            # Coinbase
            {
                'name': 'coinbase',
                'display_name': 'Coinbase Spot',
                'market_type': 'spot',
            },
            {
                'name': 'coinbase',
                'display_name': 'Coinbase Futures',
                'market_type': 'swap',
            },
            # Coinbase Exchange
            {
                'name': 'coinbaseexchange',
                'display_name': 'Coinbase Exchange',
                'market_type': 'spot',
            },
            # Kraken
            {
                'name': 'kraken',
                'display_name': 'Kraken Spot',
                'market_type': 'spot',
            },
            # Kraken Futures
            {
                'name': 'krakenfutures',
                'display_name': 'Kraken Futures',
                'market_type': 'swap',
            },
            # MEXC
            {
                'name': 'mexc',
                'display_name': 'MEXC Spot',
                'market_type': 'spot',
            },
            {
                'name': 'mexc',
                'display_name': 'MEXC Futures',
                'market_type': 'swap',
            },
            # OKX
            {
                'name': 'okx',
                'display_name': 'OKX Spot',
                'market_type': 'spot',
            },
            {
                'name': 'okx',
                'display_name': 'OKX Futures',
                'market_type': 'swap',
            },
        ]
        
        created_count = 0
        updated_count = 0
        
        for exchange_data in exchanges_data:
            exchange_data.update({
                'isactive': 1,
                'isdeleted': 0,
                'createdby': 1,
                'updatedby': 1,
            })
            
            if self.dry_run:
                self.stdout.write(f'Would create/update: {exchange_data["display_name"]} ({exchange_data["market_type"]})')
                continue
            
            exchange, created = Exchange.objects.get_or_create(
                name=exchange_data['name'],
                market_type=exchange_data['market_type'],
                defaults=exchange_data
            )
            
            if created:
                created_count += 1
                self.stdout.write(f'Created exchange: {exchange_data["display_name"]} ({exchange_data["market_type"]})')
            else:
                # Update existing exchange
                for key, value in exchange_data.items():
                    if key not in ['name', 'market_type']:
                        setattr(exchange, key, value)
                exchange.updated_at = timezone.now()
                exchange.save()
                updated_count += 1
                self.stdout.write(f'Updated exchange: {exchange_data["display_name"]} ({exchange_data["market_type"]})')
        
        if not self.dry_run:
            self.stdout.write(f'Exchanges: {created_count} created, {updated_count} updated')

    def create_trading_pairs(self):
        """Create trading pairs for each exchange based on test data"""
        if self.dry_run:
            exchanges = []
            # Mock data for dry run - estimate from our exchange data above
            for i in range(18):  # 18 exchanges from our list
                exchanges.append(type('MockExchange', (), {
                    'name': f'exchange_{i}',
                    'market_type': 'spot',
                    'display_name': f'Exchange {i}'
                })())
        else:
            exchanges = Exchange.objects.filter(isactive=1, isdeleted=0)
        
        # Define trading pairs by exchange and market type based on your test data
        exchange_pairs = {
            # Binance Spot
            ('binance', 'spot'): [
                {'symbol': 'ETH/BTC', 'base_currency': 'ETH', 'quote_currency': 'BTC'},
                {'symbol': 'LTC/BTC', 'base_currency': 'LTC', 'quote_currency': 'BTC'},
                {'symbol': 'BNB/BTC', 'base_currency': 'BNB', 'quote_currency': 'BTC'},
                {'symbol': 'NEO/BTC', 'base_currency': 'NEO', 'quote_currency': 'BTC'},
                {'symbol': 'QTUM/ETH', 'base_currency': 'QTUM', 'quote_currency': 'ETH'},
            ],
            # Binance USD-M Swap
            ('binance', 'swap'): [
                {'symbol': 'BTC/USDT:USDT', 'base_currency': 'BTC', 'quote_currency': 'USDT'},
                {'symbol': 'ETH/USDT:USDT', 'base_currency': 'ETH', 'quote_currency': 'USDT'},
                {'symbol': 'BCH/USDT:USDT', 'base_currency': 'BCH', 'quote_currency': 'USDT'},
                {'symbol': 'XRP/USDT:USDT', 'base_currency': 'XRP', 'quote_currency': 'USDT'},
                {'symbol': 'LTC/USDT:USDT', 'base_currency': 'LTC', 'quote_currency': 'USDT'},
            ],
            # Binance Coin-M
            ('binancecoinm', 'swap'): [
                {'symbol': 'BTC/USD:BTC', 'base_currency': 'BTC', 'quote_currency': 'USD'},
                {'symbol': 'ETH/USD:ETH', 'base_currency': 'ETH', 'quote_currency': 'USD'},
                {'symbol': 'LINK/USD:LINK', 'base_currency': 'LINK', 'quote_currency': 'USD'},
                {'symbol': 'BNB/USD:BNB', 'base_currency': 'BNB', 'quote_currency': 'USD'},
                {'symbol': 'TRX/USD:TRX', 'base_currency': 'TRX', 'quote_currency': 'USD'},
            ],
            # Binance USD-M
            ('binanceusdm', 'swap'): [
                {'symbol': 'BTC/USDT:USDT', 'base_currency': 'BTC', 'quote_currency': 'USDT'},
                {'symbol': 'ETH/USDT:USDT', 'base_currency': 'ETH', 'quote_currency': 'USDT'},
                {'symbol': 'BCH/USDT:USDT', 'base_currency': 'BCH', 'quote_currency': 'USDT'},
                {'symbol': 'XRP/USDT:USDT', 'base_currency': 'XRP', 'quote_currency': 'USDT'},
                {'symbol': 'LTC/USDT:USDT', 'base_currency': 'LTC', 'quote_currency': 'USDT'},
            ],
            # KuCoin Spot
            ('kucoin', 'spot'): [
                {'symbol': 'HAI/USDT', 'base_currency': 'HAI', 'quote_currency': 'USDT'},
                {'symbol': 'GHX/USDT', 'base_currency': 'GHX', 'quote_currency': 'USDT'},
                {'symbol': 'STND/USDT', 'base_currency': 'STND', 'quote_currency': 'USDT'},
                {'symbol': 'ACE/USDT', 'base_currency': 'ACE', 'quote_currency': 'USDT'},
                {'symbol': 'TOWER/USDT', 'base_currency': 'TOWER', 'quote_currency': 'USDT'},
            ],
            # KuCoin Futures
            ('kucoinfutures', 'swap'): [
                {'symbol': 'BTC/USD:BTC', 'base_currency': 'BTC', 'quote_currency': 'USD'},
                {'symbol': 'BTC/USDT:USDT', 'base_currency': 'BTC', 'quote_currency': 'USDT'},
                {'symbol': 'ETH/USDT:USDT', 'base_currency': 'ETH', 'quote_currency': 'USDT'},
                {'symbol': 'BCH/USDT:USDT', 'base_currency': 'BCH', 'quote_currency': 'USDT'},
                {'symbol': 'BSV/USDT:USDT', 'base_currency': 'BSV', 'quote_currency': 'USDT'},
            ],
            # Bitget Spot
            ('bitget', 'spot'): [
                {'symbol': 'ETH/USDT', 'base_currency': 'ETH', 'quote_currency': 'USDT'},
                {'symbol': 'BTC/USDT', 'base_currency': 'BTC', 'quote_currency': 'USDT'},
                {'symbol': 'ETH/BTC', 'base_currency': 'ETH', 'quote_currency': 'BTC'},
                {'symbol': 'LTC/USDT', 'base_currency': 'LTC', 'quote_currency': 'USDT'},
                {'symbol': 'BCH/USDT', 'base_currency': 'BCH', 'quote_currency': 'USDT'},
            ],
            # Bitget Futures
            ('bitget', 'swap'): [
                {'symbol': 'SEI/USDT:USDT', 'base_currency': 'SEI', 'quote_currency': 'USDT'},
                {'symbol': 'CYBER/USDT:USDT', 'base_currency': 'CYBER', 'quote_currency': 'USDT'},
                {'symbol': 'BAKE/USDT:USDT', 'base_currency': 'BAKE', 'quote_currency': 'USDT'},
                {'symbol': 'BIGTIME/USDT:USDT', 'base_currency': 'BIGTIME', 'quote_currency': 'USDT'},
                {'symbol': 'WAXP/USDT:USDT', 'base_currency': 'WAXP', 'quote_currency': 'USDT'},
            ],
            # Bybit Spot
            ('bybit', 'spot'): [
                {'symbol': 'BTC/USDT', 'base_currency': 'BTC', 'quote_currency': 'USDT'},
                {'symbol': 'ETH/USDT', 'base_currency': 'ETH', 'quote_currency': 'USDT'},
                {'symbol': 'XRP/USDT', 'base_currency': 'XRP', 'quote_currency': 'USDT'},
                {'symbol': 'ETH/BTC', 'base_currency': 'ETH', 'quote_currency': 'BTC'},
                {'symbol': 'XRP/BTC', 'base_currency': 'XRP', 'quote_currency': 'BTC'},
            ],
            # Bybit Futures
            ('bybit', 'swap'): [
                {'symbol': 'BCH/USDT:USDT', 'base_currency': 'BCH', 'quote_currency': 'USDT'},
                {'symbol': 'LINK/USDT:USDT', 'base_currency': 'LINK', 'quote_currency': 'USDT'},
                {'symbol': 'LTC/USDT:USDT', 'base_currency': 'LTC', 'quote_currency': 'USDT'},
                {'symbol': 'XTZ/USDT:USDT', 'base_currency': 'XTZ', 'quote_currency': 'USDT'},
                {'symbol': 'BTC/USD:BTC', 'base_currency': 'BTC', 'quote_currency': 'USD'},
            ],
            # Coinbase Spot
            ('coinbase', 'spot'): [
                {'symbol': 'AVNT/USD', 'base_currency': 'AVNT', 'quote_currency': 'USD'},
                {'symbol': 'AVNT/USDC', 'base_currency': 'AVNT', 'quote_currency': 'USDC'},
                {'symbol': 'BTC/USD', 'base_currency': 'BTC', 'quote_currency': 'USD'},
                {'symbol': 'BTC/USDC', 'base_currency': 'BTC', 'quote_currency': 'USDC'},
                {'symbol': 'ETH/USD', 'base_currency': 'ETH', 'quote_currency': 'USD'},
            ],
            # Coinbase Futures
            ('coinbase', 'swap'): [
                {'symbol': 'BTC/USDC:USDC', 'base_currency': 'BTC', 'quote_currency': 'USDC'},
                {'symbol': 'ETH/USDC:USDC', 'base_currency': 'ETH', 'quote_currency': 'USDC'},
                {'symbol': 'SOL/USDC:USDC', 'base_currency': 'SOL', 'quote_currency': 'USDC'},
                {'symbol': 'XRP/USDC:USDC', 'base_currency': 'XRP', 'quote_currency': 'USDC'},
                {'symbol': 'DOGE/USDC:USDC', 'base_currency': 'DOGE', 'quote_currency': 'USDC'},
            ],
            # Coinbase Exchange
            ('coinbaseexchange', 'spot'): [
                {'symbol': 'QSP/USD', 'base_currency': 'QSP', 'quote_currency': 'USD'},
                {'symbol': 'RENDER/USD', 'base_currency': 'RENDER', 'quote_currency': 'USD'},
                {'symbol': 'KMNO/USD', 'base_currency': 'KMNO', 'quote_currency': 'USD'},
                {'symbol': 'NEON/USD', 'base_currency': 'NEON', 'quote_currency': 'USD'},
                {'symbol': 'MASK/GBP', 'base_currency': 'MASK', 'quote_currency': 'GBP'},
            ],
            # Kraken Spot
            ('kraken', 'spot'): [
                {'symbol': '0G/EUR', 'base_currency': '0G', 'quote_currency': 'EUR'},
                {'symbol': '0G/USD', 'base_currency': '0G', 'quote_currency': 'USD'},
                {'symbol': '1INCH/EUR', 'base_currency': '1INCH', 'quote_currency': 'EUR'},
                {'symbol': '1INCH/USD', 'base_currency': '1INCH', 'quote_currency': 'USD'},
                {'symbol': 'AAVE/ETH', 'base_currency': 'AAVE', 'quote_currency': 'ETH'},
            ],
            # Kraken Futures
            ('krakenfutures', 'swap'): [
                {'symbol': 'BTC/USD:BTC', 'base_currency': 'BTC', 'quote_currency': 'USD'},
                {'symbol': 'ETH/USD:ETH', 'base_currency': 'ETH', 'quote_currency': 'USD'},
                {'symbol': 'LTC/USD:LTC', 'base_currency': 'LTC', 'quote_currency': 'USD'},
                {'symbol': 'XRP/USD:XRP', 'base_currency': 'XRP', 'quote_currency': 'USD'},
                {'symbol': 'BTC/USD:USD', 'base_currency': 'BTC', 'quote_currency': 'USD'},
            ],
            # MEXC Spot
            ('mexc', 'spot'): [
                {'symbol': 'METAL/USDT', 'base_currency': 'METAL', 'quote_currency': 'USDT'},
                {'symbol': 'RARI/USDT', 'base_currency': 'RARI', 'quote_currency': 'USDT'},
                {'symbol': 'PYTH/USDT', 'base_currency': 'PYTH', 'quote_currency': 'USDT'},
                {'symbol': 'AO/USDT', 'base_currency': 'AO', 'quote_currency': 'USDT'},
                {'symbol': 'BROCK/USDT', 'base_currency': 'BROCK', 'quote_currency': 'USDT'},
            ],
            # MEXC Futures
            ('mexc', 'swap'): [
                {'symbol': 'BTC/USDT:USDT', 'base_currency': 'BTC', 'quote_currency': 'USDT'},
                {'symbol': 'ETH/USDT:USDT', 'base_currency': 'ETH', 'quote_currency': 'USDT'},
                {'symbol': 'BCH/USDT:USDT', 'base_currency': 'BCH', 'quote_currency': 'USDT'},
                {'symbol': 'LTC/USDT:USDT', 'base_currency': 'LTC', 'quote_currency': 'USDT'},
                {'symbol': 'ETC/USDT:USDT', 'base_currency': 'ETC', 'quote_currency': 'USDT'},
            ],
            # OKX Spot
            ('okx', 'spot'): [
                {'symbol': 'CRV/USDT', 'base_currency': 'CRV', 'quote_currency': 'USDT'},
                {'symbol': 'BTC/USDT', 'base_currency': 'BTC', 'quote_currency': 'USDT'},
                {'symbol': 'ETH/USDT', 'base_currency': 'ETH', 'quote_currency': 'USDT'},
                {'symbol': 'OKB/USDT', 'base_currency': 'OKB', 'quote_currency': 'USDT'},
                {'symbol': 'SOL/USDT', 'base_currency': 'SOL', 'quote_currency': 'USDT'},
            ],
            # OKX Futures
            ('okx', 'swap'): [
                {'symbol': 'BTC/USD:BTC', 'base_currency': 'BTC', 'quote_currency': 'USD'},
                {'symbol': 'ETH/USD:ETH', 'base_currency': 'ETH', 'quote_currency': 'USD'},
                {'symbol': 'XRP/USD:XRP', 'base_currency': 'XRP', 'quote_currency': 'USD'},
                {'symbol': 'BCH/USD:BCH', 'base_currency': 'BCH', 'quote_currency': 'USD'},
                {'symbol': 'ETC/USD:ETC', 'base_currency': 'ETC', 'quote_currency': 'USD'},
            ],
        }
        
        created_count = 0
        updated_count = 0
        
        for exchange in exchanges:
            if self.dry_run:
                # Estimate pairs for this exchange
                pairs_for_exchange = exchange_pairs.get((exchange.name, exchange.market_type), [])
                for pair_data in pairs_for_exchange:
                    self.stdout.write(f'Would create: {pair_data["symbol"]} on {exchange.display_name}')
                continue
            
            # Get pairs for this specific exchange and market type
            pairs_for_exchange = exchange_pairs.get((exchange.name, exchange.market_type), [])
            
            for pair_data in pairs_for_exchange:
                pair_data_full = {
                    **pair_data,
                    'isactive': 1,
                    'isdeleted': 0,
                    'createdby': 1,
                    'updatedby': 1,
                }
                
                trading_pair, created = TradingPair.objects.get_or_create(
                    exchange=exchange,
                    symbol=pair_data['symbol'],
                    defaults=pair_data_full
                )
                
                if created:
                    created_count += 1
                    if created_count <= 20:  # Limit output
                        self.stdout.write(f'Created: {pair_data["symbol"]} on {exchange.display_name}')
                    elif created_count == 21:
                        self.stdout.write('... (additional trading pairs created)')
                else:
                    updated_count += 1
        
        if not self.dry_run:
            self.stdout.write(f'Trading pairs: {created_count} created, {updated_count} updated')
        else:
            # Calculate estimated total for dry run
            total_estimated = sum(len(pairs) for pairs in exchange_pairs.values())
            self.stdout.write(f'Would create approximately {total_estimated} trading pairs across all exchanges')

    def create_series_configurations(self):
        """Create series configurations for each trading pair and timeframe"""
        timeframes = getattr(settings, 'TRADING_CONFIG', {}).get('SUPPORTED_TIMEFRAMES', ['5m', '1h', '1d'])
        
        if self.dry_run:
            # Estimate counts based on our exchange pairs
            total_pairs = sum([
                len([
                    ('binance', 'spot'), ('binance', 'swap'), ('binancecoinm', 'swap'), ('binanceusdm', 'swap'),
                    ('kucoin', 'spot'), ('kucoinfutures', 'swap'), ('bitget', 'spot'), ('bitget', 'swap'),
                    ('bybit', 'spot'), ('bybit', 'swap'), ('coinbase', 'spot'), ('coinbase', 'swap'),
                    ('coinbaseexchange', 'spot'), ('kraken', 'spot'), ('krakenfutures', 'swap'),
                    ('mexc', 'spot'), ('mexc', 'swap'), ('okx', 'spot'), ('okx', 'swap')
                ]) * 5  # 5 pairs per exchange/market_type combo
            ])
            total_series = total_pairs * len(timeframes)
            self.stdout.write(f'Would create approximately {total_series} series configurations')
            return
        
        trading_pairs = TradingPair.objects.filter(isactive=1, isdeleted=0).select_related('exchange')
        
        created_count = 0
        updated_count = 0
        
        self.stdout.write(f'Creating series configurations for {len(timeframes)} timeframes...')
        
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
                        'probe_attempts': 0,
                        'isactive': 1,
                        'isdeleted': 0,
                    }
                )
                
                if created:
                    created_count += 1
                    if created_count <= 20:  # Limit output
                        self.stdout.write(f'Created series: {pair.exchange.name}/{pair.symbol}/{timeframe}')
                    elif created_count == 21:
                        self.stdout.write('... (additional series configs created)')
                else:
                    updated_count += 1
        
        self.stdout.write(f'Series configurations: {created_count} created, {updated_count} updated')

    def create_worker_shards(self, worker_count: int):
        """Create worker shards and distribute series"""
        if self.dry_run:
            self.stdout.write(f'Would create {worker_count} worker shards')
            return
        
        # Clear existing worker shards
        WorkerShard.objects.filter(isactive=1, isdeleted=0).delete()
        
        # Create worker shards
        workers = []
        for i in range(worker_count):
            worker = WorkerShard.objects.create(
                shard_name=f'shard_{i}',
                worker_id=f'worker_shard_{i}',
                status='idle',
                assigned_series=[],
                series_count=0,
                backfill_progress=0.0,
                isactive=1,
                isdeleted=0,
            )
            workers.append(worker)
            self.stdout.write(f'Created worker shard: {worker.shard_name}')
        
        # Distribute series across shards using stable hashing
        series_configs = SeriesConfig.objects.filter(isactive=1, isdeleted=0)
        shard_assignments = {worker.shardid: [] for worker in workers}
        
        for config in series_configs:
            # Create series key
            series_key = f"{config.exchange.exchangeid}_{config.market_type}_{config.symbol}_{config.timeframe}"
            
            # Use stable hash to assign to shard
            hash_value = int(hashlib.md5(series_key.encode()).hexdigest(), 16)
            shard_index = hash_value % len(workers)
            worker_id = workers[shard_index].shardid
            shard_assignments[worker_id].append(series_key)
        
        # Update worker assignments
        for worker in workers:
            assigned_series = shard_assignments[worker.shardid]
            worker.assigned_series = assigned_series
            worker.series_count = len(assigned_series)
            worker.save()
            
            self.stdout.write(f'Worker {worker.shard_name}: assigned {len(assigned_series)} series')
        
        total_series = sum(len(series) for series in shard_assignments.values())
        self.stdout.write(f'Distributed {total_series} series across {worker_count} workers')