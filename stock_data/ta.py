import yfinance as yf
import pandas as pd
from smartmoneyconcepts import smc


def download_stock_data(ticker: str, period="1y", interval="1d"):
    """
    Download and prepare OHLCV data
    """
    df = yf.Ticker(ticker).history(period=period, interval=interval)

    if df.empty:
        raise ValueError("No data returned")

    # Standardize columns
    df = df.rename(columns=str.lower)

    required_cols = ["open", "high", "low", "close", "volume"]
    df = df[required_cols]

    df = df.dropna()

    return df


def detect_smc_patterns(df):

    results = df.copy()

    # ✅ Use a smaller swing_length appropriate for your data size
    # Rule of thumb: swing_length should be ~5-10% of total bars
    swing_length = max(5, len(df) // 20)
    print(f"Using swing_length={swing_length} for {len(df)} bars")

    # 1️⃣ Swing Highs & Lows — compute ONCE and reuse
    swing = smc.swing_highs_lows(df, swing_length=swing_length)
    swing_prefixed = swing.add_prefix("swing_")
    results = results.join(swing_prefixed)

    swing_count = swing["HighLow"].notna().sum() if "HighLow" in swing.columns else 0
    print(f"  Swings detected: {swing_count}")

    # 2️⃣ BOS & CHoCH — pass the already-computed swing
    try:
        structure = smc.bos_choch(df, swing_highs_lows=swing)
        results = results.join(structure.add_prefix("structure_"))
        bos_count = structure["BOS"].notna().sum() if "BOS" in structure.columns else 0
        choch_count = structure["CHOCH"].notna().sum() if "CHOCH" in structure.columns else 0
        print(f"  BOS: {bos_count}, CHoCH: {choch_count}")
    except Exception as e:
        print(f"  BOS/CHoCH failed: {e}")

    # 3️⃣ Order Blocks — reuse swing
    try:
        ob = smc.ob(df, swing_highs_lows=swing)
        results = results.join(ob.add_prefix("ob_"))
        ob_count = ob["OB"].notna().sum() if "OB" in ob.columns else 0
        print(f"  Order Blocks: {ob_count}")
    except Exception as e:
        print(f"  Order Blocks failed: {e}")

    # 4️⃣ Fair Value Gaps
    try:
        fvg = smc.fvg(df)
        results = results.join(fvg.add_prefix("fvg_"))
        fvg_count = fvg["FVG"].notna().sum() if "FVG" in fvg.columns else 0
        print(f"  FVGs: {fvg_count}")
    except Exception as e:
        print(f"  FVG failed: {e}")

    # 5️⃣ Liquidity — reuse swing
    try:
        liquidity = smc.liquidity(df, swing_highs_lows=swing)
        results = results.join(liquidity.add_prefix("liq_"))
        liq_count = liquidity["Liquidity"].notna().sum() if "Liquidity" in liquidity.columns else 0
        print(f"  Liquidity levels: {liq_count}")
    except Exception as e:
        print(f"  Liquidity failed: {e}")

    # 6️⃣ Retracements — reuse swing (may not exist in all versions)
    try:
        retracements = smc.retracements(df, swing_highs_lows=swing)
        results = results.join(retracements.add_prefix("ret_"))
        print(f"  Retracements: OK")
    except AttributeError:
        print(f"  Retracements: not available in this version of smartmoneyconcepts")
    except Exception as e:
        print(f"  Retracements failed: {e}")

    return results


def main():
    ticker = "RELIANCE.NS"
    period = "1y"
    interval = "1d"

    print(f"\nDownloading {ticker} data...")
    df = download_stock_data(ticker, period, interval)
    print(f"Downloaded {len(df)} bars")

    print("\nDetecting Smart Money Concept patterns...")
    smc_results = detect_smc_patterns(df)

    # Save CSVs
    df.to_csv(f"{ticker}_ohlcv.csv")
    smc_results.to_csv(f"{ticker}_smc_signals.csv")

    print("\nDone ✅")
    print(f"Saved: {ticker}_ohlcv.csv")
    print(f"Saved: {ticker}_smc_signals.csv")


if __name__ == "__main__":
    main()