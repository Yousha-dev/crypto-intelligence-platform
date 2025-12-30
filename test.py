import ccxt
import time
from datetime import datetime, timezone

POPULAR_EXCHANGES = [
    'binance', 'binancecoinm', 'binanceusdm', 'kucoin', 'kucoinfutures', 'bitget',
    'bybit', 'coinbase', 'coinbaseexchange', 'kraken', 'krakenfutures', 'mexc', 'okx'
]

GLOBAL_LISTING_KEYS = [
    'listingDate', 'onboardDate', 'launchDate', 'listedAt', 'listDate',
    'listingTime', 'launchTime', 'onlineTime', 'onlineDate', 'openTime', 'startTime',
    'startAt', 'start_at', 'goLiveTime', 'go_live_time', 'tradingStartTime',
    'trading_start_time', 'enableTime', 'enable_time',
    'createTime', 'createdTime', 'ctime', 'utime', 'firstOpenTime', 'first_open_time',
    'firstOpenTimestamp', 'first_open_timestamp',
    'statusChangedAt', 'status_changed_at',
]

EXCHANGE_KEY_HINTS = {
    'binance':        ['onboardDate', 'listingDate', 'launchDate'],
    'binanceusdm':    ['onboardDate', 'listingDate', 'launchDate'],
    'binancecoinm':   ['onboardDate', 'listingDate', 'launchDate'],
    'bybit':          ['launchTime', 'onlineTime', 'listingTime', 'listDate'],
    'bitget':         ['listingTime', 'launchTime', 'onlineTime'],
    'okx':            ['listingTime', 'listTime', 'onlineTime', 'launchTime'],
    'okcoin':         ['listingTime', 'listTime', 'onlineTime', 'launchTime'],
    'kraken':         ['statusChangedAt', 'listingTime', 'onlineTime'],
    'krakenfutures':  ['listingTime', 'onlineTime', 'launchTime'],
    'coinbase':       ['status_changed_at', 'statusChangedAt'],
    'coinbaseexchange':['status_changed_at', 'statusChangedAt'],
    'mexc':           ['listingTime', 'onlineTime', 'launchTime'],
    'kucoin':         ['listingTime', 'onlineTime', 'launchTime', 'listDate'],
    'kucoinfutures':  ['listingTime', 'onlineTime', 'launchTime'],
    'bitfinex':       ['launchTime', 'listingTime', 'onlineTime'],
    'bitstamp':       ['listingTime', 'onlineTime', 'launchTime'],
}

def normalize_ts(value):
    """
    Return (date_str 'YYYY-MM-DD', epoch_seconds) or (None, None).
    Accepts seconds/ms ints, numeric strings, or ISO-8601 strings.
    """
    if value is None:
        return None, None
    try:
        v = int(value)
        if v > 10**12:  # ms -> s
            v //= 1000
        dt = datetime.fromtimestamp(v, tz=timezone.utc)
        return dt.strftime('%Y-%m-%d'), v
    except Exception:
        pass
    s = str(value).strip()
    if s.isdigit():
        try:
            v = int(s)
            if v > 10**12:
                v //= 1000
            dt = datetime.fromtimestamp(v, tz=timezone.utc)
            return dt.strftime('%Y-%m-%d'), v
        except Exception:
            return None, None
    try:
        iso = s.replace('Z', '+00:00') if s.endswith('Z') else s
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime('%Y-%m-%d'), int(dt.timestamp())
    except Exception:
        return None, None

def listing_keys_for(ex_id):
    preferred = EXCHANGE_KEY_HINTS.get(ex_id, [])
    seen, keys = set(), []
    for k in preferred + GLOBAL_LISTING_KEYS:
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys

def get_markets_with_listing_date_top5(delay_seconds=1.5, include_created_fallback=True, sort_order='asc'):
    for ex_id in POPULAR_EXCHANGES:
        try:
            exchange = getattr(ccxt, ex_id)({'enableRateLimit': True})
            print(f"\n=== {exchange.id.upper()} ===")

            markets = exchange.load_markets()
            keys = listing_keys_for(ex_id)

            by_type_with = {}     # mtype -> list of (epoch, date_str, symbol, matched_key)
            by_type_without = {}  # mtype -> list of symbol (no date)

            for symbol, m in markets.items():
                mtype = m.get('type', 'unknown')
                info = m.get('info', {}) or {}

                matched_key, date_str, epoch = None, None, None
                # Try info-based keys
                for key in keys:
                    if key in info and info[key] not in (None, '', 'null'):
                        date_str, epoch = normalize_ts(info[key])
                        if date_str:
                            matched_key = key
                            break
                # Optional fallback to CCXT 'created'
                if not date_str and include_created_fallback and m.get('created'):
                    date_str, epoch = normalize_ts(m['created'])
                    if date_str:
                        matched_key = 'created'

                if date_str and epoch is not None:
                    by_type_with.setdefault(mtype, []).append((epoch, date_str, symbol, matched_key))
                else:
                    by_type_without.setdefault(mtype, []).append(symbol)

            if not by_type_with and not by_type_without:
                print("  No markets found.")
                time.sleep(delay_seconds)
                continue

            for mtype in sorted(set(list(by_type_with.keys()) + list(by_type_without.keys()))):
                with_rows = by_type_with.get(mtype, [])
                without_rows = by_type_without.get(mtype, [])

                # Sort dated rows
                with_rows.sort(key=lambda r: r[0], reverse=(sort_order == 'desc'))

                # Take up to 5, filling with N/A if needed
                output = []
                for row in with_rows[:5]:
                    _, date_str, symbol, matched_key = row
                    output.append((symbol, date_str, matched_key))
                remaining = 5 - len(output)
                if remaining > 0 and without_rows:
                    for sym in without_rows[:remaining]:
                        output.append((sym, 'N/A', '-'))

                # Skip empty types entirely
                if not output:
                    continue

                total_count = len(with_rows) + len(without_rows)
                shown = len(output)
                print(f"  Market type: {mtype} (showing {shown} of {total_count})")
                for symbol, date_str, matched_key in output:
                    print(f"    {symbol:32} | {date_str} | key={matched_key}")

        except Exception as e:
            print(f"  [Error loading {ex_id}]: {e}")

        time.sleep(delay_seconds)

if __name__ == "__main__":
    # sort_order: 'asc' for oldest first, 'desc' for newest first
    get_markets_with_listing_date_top5(delay_seconds=1.5, include_created_fallback=True, sort_order='asc')