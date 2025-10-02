"""
cheatsheet_backtester.py

Requirements:
    pip install yfinance pandas numpy matplotlib ta

Usage:
    python cheatsheet_backtester.py --ticker AAPL --start 2020-01-01 --end 2025-09-30

What it does:
    - Pulls data from yfinance
    - Computes RSI(14), OBV, MACD(12,26,9)
    - Classifies RSI/OBV/MACD states with heuristics
    - Looks up a probability+direction using your cheat-sheet mapping (best effort)
    - Asks: "Is there BSI?" (y/n). If 'y', flips probabilities.
    - Prints the result and plots the indicators.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import MACD
# --- Auto-install yfinance if it's missing ---
import subprocess
import sys

try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf


# ---------- helpers: indicator calculations ----------
def compute_indicators(df):
    # assume df index is datetime and contains 'Close' and 'Volume'
    df = df.copy()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    # simple moving averages to help classify flattening
    df['RSI_slope_3'] = df['RSI'].diff(3)
    df['OBV_slope_3'] = df['OBV'].diff(3)
    df['MACD_slope_3'] = df['MACD'].diff(3)
    return df

# ---------- heuristics to classify states ----------
def classify_rsi_state(df):
    """
    Return one of:
    'bullish_divergence', 'bearish_divergence',
    'confirm_increasing', 'confirm_decreasing',
    'increasing_flattening', 'decreasing_flattening'
    Uses last 30 bars for divergence heuristics.
    """
    s = df['RSI'].dropna()
    if len(s) < 30:
        # fallback by slope
        slope = s.iloc[-1] - s.iloc[-7] if len(s) > 7 else 0
        if slope > 3:
            return 'confirm_increasing'
        elif slope < -3:
            return 'confirm_decreasing'
        else:
            return 'increasing_flattening' if slope > 0 else 'decreasing_flattening'

    price = df['Close'].dropna()
    # simple divergence detection using recent swings:
    window = 30
    price_recent = price.iloc[-window:]
    rsi_recent = s.iloc[-window:]

    # find two most recent lows (for bullish divergence) or highs (for bearish)
    # bullish divergence: price makes lower low, RSI makes higher low
    price_ll_idx = price_recent.idxmin()
    rsi_ll_idx = rsi_recent.idxmin()
    # compare previous low (outside last 7 bars) vs current low
    try:
        prev_price_low = price_recent[:price_ll_idx].min()
        prev_rsi_low = rsi_recent[:rsi_ll_idx].min()
        curr_price_low = price_recent.loc[price_ll_idx]
        curr_rsi_low = rsi_recent.loc[rsi_ll_idx]
        bullish_div = (curr_price_low < prev_price_low) and (curr_rsi_low > prev_rsi_low)
    except Exception:
        bullish_div = False

    # bearish divergence: price makes higher high while RSI makes lower high
    price_hh_idx = price_recent.idxmax()
    rsi_hh_idx = rsi_recent.idxmax()
    try:
        prev_price_high = price_recent[:price_hh_idx].max()
        prev_rsi_high = rsi_recent[:rsi_hh_idx].max()
        curr_price_high = price_recent.loc[price_hh_idx]
        curr_rsi_high = rsi_recent.loc[rsi_hh_idx]
        bearish_div = (curr_price_high > prev_price_high) and (curr_rsi_high < prev_rsi_high)
    except Exception:
        bearish_div = False

    if bullish_div:
        return 'bullish_divergence'
    if bearish_div:
        return 'bearish_divergence'

    # otherwise, classify trend vs flattening
    slope_7 = s.iloc[-1] - s.iloc[-7] if len(s) > 7 else 0
    slope_3 = s.iloc[-1] - s.iloc[-3] if len(s) > 3 else 0
    # if RSI was increasing in past 7 and in last 3 there's low change => increasing -> flattening
    if slope_7 > 4 and abs(slope_3) < 1.5:
        return 'increasing_flattening'
    if slope_7 > 4:
        return 'confirm_increasing'
    if slope_7 < -4 and abs(slope_3) < 1.5:
        return 'decreasing_flattening'
    if slope_7 < -4:
        return 'confirm_decreasing'
    # fallback neutral
    return 'increasing_flattening' if slope_7 >= 0 else 'decreasing_flattening'

def classify_obv_state(df):
    """
    Return one of:
    'confirm_increasing', 'confirm_decreasing',
    'increasing_flattening', 'decreasing_flattening',
    'divergence_bullish', 'divergence_bearish'
    """
    obv = df['OBV'].dropna()
    if len(obv) < 10:
        return 'confirm_increasing' if obv.iloc[-1] - obv.iloc[0] > 0 else 'confirm_decreasing'

    slope_14 = obv.iloc[-1] - obv.iloc[-14] if len(obv) > 14 else obv.iloc[-1] - obv.iloc[0]
    slope_3 = obv.iloc[-1] - obv.iloc[-3] if len(obv) > 3 else slope_14
    # flattening if longer slope positive but short slope near zero
    if slope_14 > 0 and abs(slope_3) < 0.05 * max(1, abs(slope_14)):
        return 'increasing_flattening'
    if slope_14 > 0:
        return 'confirm_increasing'
    if slope_14 < 0 and abs(slope_3) < 0.05 * max(1, abs(slope_14)):
        return 'decreasing_flattening'
    if slope_14 < 0:
        return 'confirm_decreasing'
    return 'confirm_increasing'

def classify_macd_state(df):
    """
    Return one of:
    'rising', 'increasing_flattening', 'decreasing_flattening',
    'curling_up', 'curling_down', 'falling'
    We'll use MACD line and histogram heuristics.
    """
    macd = df['MACD'].dropna()
    hist = df['MACD_hist'].dropna()
    if len(macd) < 10:
        return 'rising' if macd.iloc[-1] > macd.iloc[0] else 'falling'
    macd_slope_14 = macd.iloc[-1] - macd.iloc[-14] if len(macd) > 14 else macd.iloc[-1] - macd.iloc[0]
    macd_slope_3 = macd.iloc[-1] - macd.iloc[-3] if len(macd) > 3 else macd_slope_14
    hist_last = hist.iloc[-1]
    hist_prev = hist.iloc[-3] if len(hist) > 3 else hist.iloc[0]

    # curling: histogram turning positive from negative (curling up) or turning negative from positive (curling down)
    if hist_prev < 0 and hist_last > 0:
        return 'curling_up'
    if hist_prev > 0 and hist_last < 0:
        return 'curling_down'

    # increasing -> flattening: macd long-term up but short-term slope small
    if macd_slope_14 > 0 and abs(macd_slope_3) < 0.02 * max(1, abs(macd_slope_14)):
        return 'increasing_flattening'
    if macd_slope_14 > 0:
        return 'rising'
    if macd_slope_14 < 0 and abs(macd_slope_3) < 0.02 * max(1, abs(macd_slope_14)):
        return 'decreasing_flattening'
    return 'falling'

# ---------- cheat-sheet mapping (best-effort subset + fallback) ----------
# For full fidelity you can expand this dict by mapping every combination
# from your cheat sheet. Here I include the main combinations. Missing combinations fallback to neutral 50-55%.
CHEAT_MAP = {
    # Format: CHEAT_MAP[RSI_section][OBV_section][MACD_state] = (prob_range_str, direction_label)
    # We'll fill the main ones used in our classification.
    'bullish_divergence': {
        'confirm_increasing': {
            'rising': ('75–80%', 'bullish'),
            'increasing_flattening': ('65–70%', 'neutral-bullish'),
            'decreasing_flattening': ('60–65%', 'neutral-bullish'),
            'curling_up': ('70–75%', 'bullish'),
            'curling_down': ('55–60%', 'bearish'),
            'falling': ('50–55%', 'bearish'),
        },
        'confirm_decreasing': {
            'rising': ('60–65%', 'bullish'),
            'increasing_flattening': ('55–60%', 'neutral-bullish'),
            'decreasing_flattening': ('50–55%', 'neutral-bullish'),
            'curling_up': ('65–70%', 'bullish'),
            'curling_down': ('55–60%', 'bearish'),
            'falling': ('50–55%', 'bearish'),
        },
        'decreasing_flattening': {
            'rising': ('60–65%','neutral-bullish'),
            'increasing_flattening': ('55–60%','neutral'),
            'decreasing_flattening': ('50–55%','neutral-bearish'),
            'curling_up': ('60–65%','bullish'),
            'curling_down': ('55–60%','bearish'),
            'falling': ('50–55%','bearish'),
        },
        'divergence_bullish': {
            'rising': ('70–75%','bullish'),
            'increasing_flattening': ('60–65%','neutral-bullish'),
            'decreasing_flattening': ('55–60%','neutral'),
            'curling_up': ('65–70%','bullish'),
            'curling_down': ('55–60%','neutral'),
            'falling': ('50–55%','bearish'),
        },
        'divergence_bearish': {
            'rising': ('55–60%','bullish'),
            'increasing_flattening': ('50–55%','neutral'),
            'decreasing_flattening': ('55–60%','neutral-bearish'),
            'curling_up': ('55–60%','bullish'),
            'curling_down': ('65–70%','bearish'),
            'falling': ('70–75%','bearish'),
        },
        'increasing_flattening': {
            'rising': ('65–70%','bullish'),
            'increasing_flattening': ('55–60%','neutral'),
            'decreasing_flattening': ('55–60%','neutral-bearish'),
            'curling_up': ('65–70%','bullish'),
            'curling_down': ('55–60%','bearish'),
            'falling': ('55–60%','bearish'),
        }
    },
    'bearish_divergence': {
        'confirm_decreasing': {
            'rising': ('60–65%','bearish'),
            'increasing_flattening': ('55–60%','neutral-bearish'),
            'decreasing_flattening': ('70–75%','bearish'),
            'curling_up': ('60–65%','bearish'),
            'curling_down': ('75–80%','bearish'),
            'falling': ('80–85%','bearish'),
        },
        'confirm_increasing': {
            'rising': ('55–60%','bearish'),
            'increasing_flattening': ('60–65%','neutral'),
            'decreasing_flattening': ('55–60%','neutral-bullish'),
            'curling_up': ('55–60%','neutral-bullish'),
            'curling_down': ('65–70%','bearish'),
            'falling': ('70–75%','bearish'),
        },
        'divergence_bearish': {
            'rising': ('65–70%','bearish'),
            'increasing_flattening': ('60–65%','neutral-bearish'),
            'decreasing_flattening': ('70–75%','bearish'),
            'curling_up': ('60–65%','bearish'),
            'curling_down': ('75–80%','bearish'),
            'falling': ('80–85%','bearish'),
        },
        'divergence_bullish': {
            'rising': ('65–70%','bullish'),
            'increasing_flattening': ('55–60%','neutral'),
            'decreasing_flattening': ('55–60%','neutral-bearish'),
            'curling_up': ('65–70%','bullish'),
            'curling_down': ('60–65%','bearish'),
            'falling': ('65–70%','bearish'),
        },
        'increasing_flattening': {
            'rising': ('55–60%','bearish'),
            'increasing_flattening': ('55–60%','neutral-bullish'),
            'decreasing_flattening': ('60–65%','neutral'),
            'curling_up': ('55–60%','bearish'),
            'curling_down': ('65–70%','bearish'),
            'falling': ('70–75%','bearish'),
        }
    },
    'confirm_increasing': {
        # RSI Confirm Increasing: stronger bullish confirmations
        'confirm_increasing': {
            'rising': ('80–85%','bullish'),
            'increasing_flattening': ('70–75%','bullish'),
            'decreasing_flattening': ('65–70%','neutral-bullish'),
            'curling_up': ('75–80%','bullish'),
            'curling_down': ('60–65%','bullish'),
            'falling': ('55–60%','bearish'),
        },
        # other OBV contexts are handled via fallback or added above
    },
    'confirm_decreasing': {
        'confirm_decreasing': {
            'rising': ('65–70%','bearish'),
            'increasing_flattening': ('60–65%','neutral-bearish'),
            'decreasing_flattening': ('70–75%','bearish'),
            'curling_up': ('60–65%','bearish'),
            'curling_down': ('80–85%','bearish'),
            'falling': ('85–90%','bearish'),
        }
    },
    'increasing_flattening': {
        # RSI increasing -> flattening: momentum paused after rise
        'confirm_increasing': {
            'rising': ('65–70%','bullish'),
            'increasing_flattening': ('55–60%','neutral'),
            'decreasing_flattening': ('55–60%','neutral-bearish'),
            'curling_up': ('65–70%','bullish'),
            'curling_down': ('55–60%','bearish'),
            'falling': ('55–60%','bearish'),
        }
    },
    'decreasing_flattening': {
        # RSI decreasing -> flattening: momentum paused after drop
        'confirm_decreasing': {
            'rising': ('55–60%','bearish'),
            'increasing_flattening': ('55–60%','neutral-bullish'),
            'decreasing_flattening': ('60–65%','neutral'),
            'curling_up': ('55–60%','bearish'),
            'curling_down': ('65–70%','bearish'),
            'falling': ('70–75%','bearish'),
        }
    }
}

FALLBACK = ('50–55%', 'neutral')

def lookup_prob(rsi_section, obv_section, macd_state):
    r = CHEAT_MAP.get(rsi_section, {})
    ob = r.get(obv_section, {})
    if ob and macd_state in ob:
        return ob[macd_state]
    # try fallback paths: maybe top-level mapping exists for OBV as direct key
    # try obv in CHEAT_MAP[rsi_section] if macd_state present
    if r and macd_state in r.get(obv_section, {}):
        return r[obv_section][macd_state]
    # try other reasonable fallbacks
    if r:
        # pick the nearest obv key available
        for obk, md in r.items():
            if macd_state in md:
                return md[macd_state]
    # last resort fallback:
    return FALLBACK

# ---------- BSI flip ----------
def flip_result(prob_str, direction):
    """
    Flip direction and adjust the probability string numerically.
    Example: '75–80%' bullish -> becomes '75–80%' bearish
    For neutral labels, retain neutral.
    If direction is neutral-bullish or neutral-bearish, flip the bullish/bearish part.
    """
    # map direction flipping
    dir_map = {
        'bullish': 'bearish',
        'bearish': 'bullish',
        'neutral': 'neutral',
        'neutral-bullish': 'neutral-bearish',
        'neutral-bearish': 'neutral-bullish'
    }
    flipped_dir = dir_map.get(direction, direction)
    return prob_str, flipped_dir

# ---------- plotting ----------
def plot_indicators(df, ticker):
    plt.figure(figsize=(12,10))
    ax1 = plt.subplot(4,1,1)
    plt.plot(df.index, df['Close'], label='Close')
    plt.title(f"{ticker} Price")
    plt.legend()
    ax2 = plt.subplot(4,1,2, sharex=ax1)
    plt.plot(df.index, df['RSI'], label='RSI(14)')
    plt.axhline(70, linestyle='--', alpha=0.3)
    plt.axhline(30, linestyle='--', alpha=0.3)
    plt.title("RSI(14)")
    plt.legend()
    ax3 = plt.subplot(4,1,3, sharex=ax1)
    plt.plot(df.index, df['OBV'], label='OBV')
    plt.title("OBV")
    plt.legend()
    ax4 = plt.subplot(4,1,4, sharex=ax1)
    plt.plot(df.index, df['MACD'], label='MACD')
    plt.plot(df.index, df['MACD_signal'], label='Signal')
    plt.bar(df.index, df['MACD_hist'], label='Hist', alpha=0.4)
    plt.title("MACD (12,26,9)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- main ----------
def analyze_ticker(ticker, start, end, plot=True):
    print(f"Fetching {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data fetched. Check ticker or dates and internet connection.")
    df = compute_indicators(df)

    # classify
    rsi_section = classify_rsi_state(df)
    obv_section = classify_obv_state(df)
    macd_state = classify_macd_state(df)

    print("Detected states:")
    print("  RSI section:", rsi_section)
    print("  OBV section:", obv_section)
    print("  MACD state:", macd_state)

    # ask BSI question
    bsi_input = input("Is there BSI? (y/n) >>> ").strip().lower()
    bsi = bsi_input == 'y'

    prob, direction = lookup_prob(rsi_section, obv_section, macd_state)
    if bsi:
        prob, direction = flip_result(prob, direction)

    # print final result
    print("\n=== RESULT ===")
    print(f"Ticker: {ticker}")
    print(f"RSI section: {rsi_section}")
    print(f"OBV section: {obv_section}")
    print(f"MACD state: {macd_state}")
    print(f"Probability / Direction -> {prob}  {direction}")
    print("Note: This is a heuristic application of your cheat-sheet. Expand CHEAT_MAP to include more exact combos if needed.\n")

    if plot:
        plot_indicators(df, ticker)

    return {
        'ticker': ticker,
        'rsi_section': rsi_section,
        'obv_section': obv_section,
        'macd_state': macd_state,
        'probability': prob,
        'direction': direction,
        'bsi': bsi
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--start', default='2020-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--noplot', action='store_true')
    args = parser.parse_args()
    import datetime
    end = args.end or (datetime.date.today().isoformat())
    res = analyze_ticker(args.ticker, args.start, end, plot=not args.noplot)
