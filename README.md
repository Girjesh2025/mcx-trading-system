# MCX Trading System

An advanced technical analysis and trading system for Multi Commodity Exchange (MCX) instruments.

## Setup Instructions

1. **Login to Fyers API**:
   First, run the login GUI to get your access token:
   ```
   python mcx_login_gui.py
   ```
   Follow the on-screen instructions to authenticate with Fyers and save your access token.

2. **Required Python Packages**:
   - pandas
   - numpy
   - matplotlib
   - colorama
   - tabulate
   - fyers_apiv3

## Running the Application

The MCX trading system has several modes:

```
python mcx_trader.py              # Normal mode with analysis
python mcx_trader.py dashboard    # Display market dashboard only
python mcx_trader.py signals      # Show consolidated signals across timeframes
python mcx_trader.py virtual      # Run automated virtual trading session
python mcx_trader.py realtime     # Start real-time market monitoring 
python mcx_trader.py eod          # Run end-of-day analysis
python mcx_trader.py gold         # Place a simulated GOLD trade
python mcx_trader.py close SYMBOL # Close positions for a specific symbol
```

## Features

- Real-time technical analysis with multiple indicators (RSI, CCI, ATR, Bollinger Bands)
- Color-coded dashboard with market status
- Consolidated signals table across multiple timeframes
- Virtual trading with position tracking and performance metrics
- End-of-day analysis and reporting
- Automated signal generation with score-based filtering

## Note

This system operates in simulation mode by default and does not place real trades. All trades are virtual and use simulated capital.

## Access Token Management

The login process generates an access token saved in the `mcx_master` directory. The token is valid for one day and needs to be refreshed by running the login process again. 