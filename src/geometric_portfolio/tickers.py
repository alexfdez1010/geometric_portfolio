import yfinance as yf

TICKERS = {
    "S&P 500 (VOO)": "VOO",
    "Nasdaq (QQQ)": "QQQ",
    "SPDR Dow Jones (DIA)": "DIA",
    "SPDR Gold Trust (GLD)": "GLD",
    "iShares Silver Trust (SLV)": "SLV",
    "S&P 500 VIX Short-term Futures Index (VILX.L)": "VILX.L",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "United States Oil (USO)": "USO",
    "iShares 20+ Year Treasury (TLT)": "TLT",
    "iShares 7-10 Year Treasury (IEF)": "IEF",
    "iPath S&P 500 VIX (VXX)": "VXX",
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "Tesla Inc. (TSLA)": "TSLA",
    "Meta Platforms Inc. (META)": "META",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "Walmart Inc. (WMT)": "WMT",
    "Invesco S&P 500 Equal Weight ETF (RSP)": "RSP",
    "First Trust NASDAQ-100 Equal Weighted Index Fund (QQEW)": "QQEW",
    "Invesco S&P SmallCap 600 Equal Weight ETF (EWSC)": "EWSC",
    "Invesco S&P MidCap 400 Equal Weight ETF (EWMC)": "EWMC",
    "ProShares VIX Short-Term Futures ETF (VIXY)": "VIXY",
    "ProShares VIX Mid-Term Futures ETF (VIXM)": "VIXM",
    "ProShares Ultra VIX Short-Term Futures ETF (UVXY)": "UVXY",
    "ProShares Ultra S&P500 (SSO)": "SSO",
    "ProShares UltraPro QQQ (TQQQ)": "TQQQ",
    "ProShares Ultra Gold (UGL)": "UGL",
    "WisdomTree Gold 3x Daily Leveraged (3GOL.L)": "3GOL.L",
    "Direxion Daily Gold Miners Bull 2X Shares (NUGT)": "NUGT",
    "ProShares Ultra Silver (AGQ)": "AGQ",
}

CATEGORIES = {
    "Equity ETFs": ["S&P 500 (VOO)", "Nasdaq (QQQ)", "SPDR Dow Jones (DIA)", "Invesco S&P 500 Equal Weight ETF (RSP)", "First Trust NASDAQ-100 Equal Weighted Index Fund (QQEW)", "Invesco S&P SmallCap 600 Equal Weight ETF (EWSC)", "Invesco S&P MidCap 400 Equal Weight ETF (EWMC)"],
    "Leveraged ETFs": ["ProShares Ultra S&P500 (SSO)", "ProShares UltraPro QQQ (TQQQ)", "ProShares Ultra Gold (UGL)", "WisdomTree Gold 3x Daily Leveraged (3GOL.L)", "Direxion Daily Gold Miners Bull 2X Shares (NUGT)", "ProShares Ultra Silver (AGQ)"],
    "Crypto": ["Bitcoin (BTC-USD)", "Ethereum (ETH-USD)"],
    "Commodities": ["SPDR Gold Trust (GLD)", "iShares Silver Trust (SLV)", "United States Oil (USO)"],
    "Treasuries": ["iShares 20+ Year Treasury (TLT)", "iShares 7-10 Year Treasury (IEF)"],
    "VIX ETFs": ["S&P 500 VIX Short-term Futures Index (VILX.L)", "iPath S&P 500 VIX (VXX)", "ProShares VIX Short-Term Futures ETF (VIXY)", "ProShares VIX Mid-Term Futures ETF (VIXM)", "ProShares Ultra VIX Short-Term Futures ETF (UVXY)"],
    "Stocks": ["Apple Inc. (AAPL)", "Microsoft Corp. (MSFT)", "Alphabet Inc. (GOOGL)", "Amazon.com Inc. (AMZN)", "NVIDIA Corp. (NVDA)", "Tesla Inc. (TSLA)", "Meta Platforms Inc. (META)", "Berkshire Hathaway (BRK-B)", "Walmart Inc. (WMT)"]
}

def validate_ticker(ticker: str) -> bool:
    """
    Validate ticker via yfinance by attempting to download data.
    """
    try:
        df = yf.download(ticker, period="1d", auto_adjust=True, progress=False)
        return not df.empty
    except Exception:
        return False

def resolve_ticker(name_or_symbol: str) -> str:
    """
    Resolve a ticker symbol or friendly name to a valid ticker.
    """
    if name_or_symbol in TICKERS:
        return TICKERS[name_or_symbol]
    symbol = name_or_symbol.upper().strip()
    if validate_ticker(symbol):
        return symbol
    raise ValueError(f"Ticker {symbol} is invalid or no data found")