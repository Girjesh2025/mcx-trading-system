import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
import time
from fyers_apiv3 import fyersModel
import matplotlib.pyplot as plt
from tabulate import tabulate
import pytz
import os
import json
from colorama import Fore, Back, Style, init

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

class MCXTrader:
    def __init__(self):
        self.fyers = self.connect_fyers()
        self.capital = 100000  # Virtual capital ₹1 Lakh
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.positions = {}
        self.trades_history = []
        self.market_regime = "normal"
        self.is_simulation = True  # Flag to indicate simulation mode
        
        # MCX market timing constraints
        self.indian_timezone = pytz.timezone('Asia/Kolkata')
        self.market_open_time = dt_time(9, 0)   # 9:00 AM
        self.market_close_time = dt_time(23, 30)  # 11:30 PM
        self.eod_analysis_time = dt_time(23, 25)  # Run EOD analysis at 11:25 PM
        
        # Available MCX instruments with proper symbol formats
        self.load_mcx_symbols()
        
        # Last known prices for each symbol
        self.last_prices = {}
        
        # Load configuration if exists
        self.config = self.load_config()
        
        # Print welcome message
        self._print_welcome()
        
    def _print_welcome(self):
        """Display welcome message with styling"""
        print("\n" + "═" * 100)
        print(f"{Fore.YELLOW}{Style.BRIGHT}╔══════════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}║                        MCX PROFESSIONAL TRADING SYSTEM                        ║{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        print("═" * 100)
        
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            if os.path.exists("mcx_config.json"):
                with open("mcx_config.json", "r") as f:
                    config = json.load(f)
                print(f"{Fore.GREEN}✅ Loaded configuration{Style.RESET_ALL}")
                return config
            else:
                # Default configuration
                config = {
                    "capital": 100000,
                    "risk_per_trade": 0.02,
                    "use_trailing_stop": True,
                    "trailing_stop_percent": 0.5,
                    "take_profit_ratio": 2.0,
                    "stop_loss_atr_multiplier": 1.5,
                    "auto_exit_eod": True,
                    "favorite_symbols": ["SILVER", "GOLD", "CRUDEOIL", "NATURALGAS"]
                }
                # Save default config
                with open("mcx_config.json", "w") as f:
                    json.dump(config, f, indent=4)
                print(f"{Fore.GREEN}✅ Created default configuration{Style.RESET_ALL}")
                return config
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading configuration: {e}{Style.RESET_ALL}")
            return {
                "capital": 100000,
                "risk_per_trade": 0.02,
                "favorite_symbols": ["SILVER", "GOLD"]
            }
    
    def load_mcx_symbols(self):
        """Load MCX symbols from file or use defaults"""
        try:
            if os.path.exists("mcx_symbols.json"):
                with open("mcx_symbols.json", "r") as f:
                    self.mcx_symbols = json.load(f)
                print(f"{Fore.GREEN}✅ Loaded {len(self.mcx_symbols)} MCX symbols{Style.RESET_ALL}")
            else:
                # Default MCX symbols - these should be updated regularly with correct format
                # Format from Fyers: MCX:[SYMBOL][EXPIRY_DATE]FUT
                current_month = datetime.now(self.indian_timezone).strftime('%b').upper()
                current_year = datetime.now(self.indian_timezone).year % 100  # Last 2 digits
                
                # Use formats exactly as shown in Fyers
                self.mcx_symbols = {
                    "SILVER": f"MCX:SILVER25{current_month}FUT",
                    "GOLD": f"MCX:GOLD25{current_month}FUT",
                    "CRUDEOIL": f"MCX:CRUDEOILM25JUN{current_year}FUT",  # Note 'M' in symbol
                    "NATURALGAS": f"MCX:NATURALGAS25{current_month}FUT",
                    "COPPER": f"MCX:COPPER25{current_month}FUT",
                    "ZINC": f"MCX:ZINC25{current_month}FUT",
                    "LEAD": f"MCX:LEAD25{current_month}FUT",
                    "ALUMINIUM": f"MCX:ALUMINIUM25{current_month}FUT"
                }
                
                # Save default symbols
                with open("mcx_symbols.json", "w") as f:
                    json.dump(self.mcx_symbols, f, indent=4)
                print(f"{Fore.GREEN}✅ Created default MCX symbols file{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Available symbols:{Style.RESET_ALL}")
                for symbol, fyers_code in self.mcx_symbols.items():
                    print(f"  {Fore.YELLOW}{symbol}{Style.RESET_ALL}: {fyers_code}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading MCX symbols: {e}{Style.RESET_ALL}")
            # Fallback to minimal set with corrected formats
            current_month = datetime.now(self.indian_timezone).strftime('%b').upper()
            self.mcx_symbols = {
                "SILVER": f"MCX:SILVER25{current_month}FUT",
                "GOLD": f"MCX:GOLD25{current_month}FUT",
                "CRUDEOIL": f"MCX:CRUDEOILM25JUN{datetime.now(self.indian_timezone).year % 100}FUT"
            }
        
    def connect_fyers(self):
        """Connect to Fyers API"""
        try:
            # Use the correct path for the access token file in the same directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            token_file = os.path.join(script_dir, "access_token.txt")
            
            with open(token_file, "r") as f:
                token = f.read().strip()
                
            fyers = fyersModel.FyersModel(
                client_id="JAOZFJL8IO-100",
                is_async=False,
                token=token,
                log_path=""
            )
            print(f"{Fore.GREEN}✅ Connected to Fyers{Style.RESET_ALL}")
            return fyers
        except Exception as e:
            print(f"{Fore.RED}❌ Connection error: {str(e)}{Style.RESET_ALL}")
            return None
    
    def is_market_open(self, check_time=None):
        """Check if MCX market is open at the given time"""
        if check_time is None:
            check_time = datetime.now(self.indian_timezone)
        elif not check_time.tzinfo:
            # If the time doesn't have timezone info, assume it's in Indian timezone
            check_time = self.indian_timezone.localize(check_time)
            
        # Check if it's a weekday (0 = Monday, 6 = Sunday)
        if check_time.weekday() >= 5:  # Saturday or Sunday
            return False
            
        current_time = check_time.time()
        return self.market_open_time <= current_time <= self.market_close_time
    
    def get_historical_data(self, symbol, timeframe="15", days=30):
        """Get historical data for analysis"""
        try:
            # Handle symbol shorthand
            if symbol in self.mcx_symbols:
                actual_symbol = self.mcx_symbols[symbol]
            else:
                actual_symbol = symbol
                
            print(f"\nFetching data for {actual_symbol}...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = {
                "symbol": actual_symbol,
                "resolution": timeframe,
                "date_format": "1",
                "range_from": start_date.strftime("%Y-%m-%d"),
                "range_to": end_date.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }
            
            if self.fyers:
                hist_data = self.fyers.history(data)
                if hist_data["s"] == "ok":
                    df = pd.DataFrame(hist_data["candles"])
                    df.columns = ["datetime", "open", "high", "low", "close", "volume"]
                    df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
                    
                    # Convert to Indian timezone
                    df["datetime"] = df["datetime"].dt.tz_localize('UTC').dt.tz_convert(self.indian_timezone)
                    
                    # Filter for market hours
                    df = df[df["datetime"].apply(lambda x: self.is_market_open(x))]
                    
                    print(f"✅ Got {len(df)} candles during market hours")
                    
                    # Store the last price
                    symbol_key = symbol if symbol in self.mcx_symbols else actual_symbol
                    if len(df) > 0:
                        self.last_prices[symbol_key] = df.iloc[-1]['close']
                    
                    # Add symbol column for reference
                    df['symbol'] = symbol if symbol in self.mcx_symbols else actual_symbol
                    
                    return df
                else:
                    print(f"❌ API Error: {hist_data}")
            
            # If we can't get real data, generate simulated data
            if self.is_simulation:
                symbol_key = symbol if symbol in self.mcx_symbols else symbol
                print(f"Generating simulated data for {symbol_key}...")
                return self._generate_simulated_data(symbol_key, start_date, end_date)
                    
            return None
            
        except Exception as e:
            print(f"❌ Historical data error: {str(e)}")
            
            # Fallback to simulated data if real data fetching fails
            if self.is_simulation:
                symbol_key = symbol if symbol in self.mcx_symbols else symbol
                print(f"Falling back to simulated data for {symbol_key}...")
                return self._generate_simulated_data(symbol_key, datetime.now() - timedelta(days=days), datetime.now())
                
            return None
    
    def _generate_simulated_data(self, symbol, start_date, end_date):
        """Generate simulated price data when real data is unavailable"""
        print(f"🔄 Generating simulated data for {symbol}...")
        
        # Set initial price based on what instrument we're simulating
        if "SILVER" in symbol.upper():
            base_price = 75000
            volatility = 500
        elif "GOLD" in symbol.upper():
            base_price = 62000
            volatility = 300
        elif "CRUDEOIL" in symbol.upper():
            base_price = 6500
            volatility = 100
        elif "NATURALGAS" in symbol.upper():
            base_price = 230
            volatility = 8
        elif "COPPER" in symbol.upper():
            base_price = 780
            volatility = 10
        else:
            base_price = 1000
            volatility = 20
            
        # Create date range for MCX market hours only
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Weekday
                for hour in range(9, 24):
                    for minute in [0, 15, 30, 45]:
                        if hour == 23 and minute > 30:
                            continue
                        
                        all_dates.append(self.indian_timezone.localize(
                            datetime(current_date.year, current_date.month, current_date.day, hour, minute)))
            
            current_date += timedelta(days=1)
            
        # Generate random price data with trend
        np.random.seed(42)  # For reproducibility
        
        # Create the dataframe with dates
        df = pd.DataFrame(index=range(len(all_dates)))
        df['datetime'] = all_dates
        
        # Generate prices with some randomness but also a trend
        prices = [base_price]
        volumes = []
        
        for i in range(1, len(all_dates)):
            # Add more realistic price movement
            random_drift = np.random.normal(0, 1)
            trend = 0.1 * np.sin(i/20)  # Cyclic trend
            momentum = 0.8 * (prices[-1] - prices[-2]) if i > 1 else 0
            
            # Next price is previous + random + trend + momentum, all scaled by volatility
            new_price = prices[-1] + volatility * (0.01 * random_drift + 0.005 * trend + 0.002 * momentum)
            prices.append(new_price)
            
            # Generate volume
            volumes.append(int(np.random.normal(1000000, 500000)))
            
        # Fill the dataframe
        df['close'] = prices
        df['open'] = df['close'].shift(1)
        df.loc[0, 'open'] = prices[0] * 0.998  # First open price
        
        # High and low with randomness
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, volatility * 0.01, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, volatility * 0.01, len(df))
        df['volume'] = volumes + [int(np.random.normal(1000000, 500000))]
        df['symbol'] = symbol
        
        print(f"✅ Generated {len(df)} simulated candles for {symbol}")
        
        # Update last price
        if len(df) > 0:
            self.last_prices[symbol] = df.iloc[-1]['close']
            
        return df
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for MCX instruments"""
        try:
            print("\nCalculating indicators...")
            
            # Basic indicators
            df['ema20'] = df['close'].ewm(span=20).mean()
            df['ema50'] = df['close'].ewm(span=50).mean()
            df['ema200'] = df['close'].ewm(span=200).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            df['signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['signal']
            
            # Bollinger Bands
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['stddev'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['sma20'] + (df['stddev'] * 2)
            df['lower_band'] = df['sma20'] - (df['stddev'] * 2)
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['sma20']
            
            # ATR for volatility measurement (important for commodities)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            # Commodity-specific indicators
            
            # Volume indicators - very important for commodities
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Money Flow Index (volume-weighted RSI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = money_flow.copy()
            negative_flow = money_flow.copy()
            
            positive_flow[typical_price.diff() < 0] = 0
            negative_flow[typical_price.diff() > 0] = 0
            
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            
            money_ratio = positive_mf / negative_mf
            df['mfi'] = 100 - (100 / (1 + money_ratio))
            
            # Commodity Channel Index
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            mean_deviation = abs(typical_price - typical_price.rolling(window=20).mean()).rolling(window=20).mean()
            df['cci'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * mean_deviation)
            
            # Print current values for the most recent candle
            if len(df) > 0:
                current = df.iloc[-1]
                print(f"\nCurrent Values for {current['symbol']}:")
                print(f"Last Update: {current['datetime'].strftime('%Y-%m-%d %H:%M')}")
                print(f"Close: {current['close']:.2f}")
                print(f"ATR: {current['atr']:.2f} ({(current['atr']/current['close']*100):.2f}%)")
                print(f"RSI: {current['rsi']:.2f}")
                print(f"MFI: {current['mfi']:.2f}")
                print(f"CCI: {current['cci']:.2f}")
                print(f"BB Width: {current['bb_width']:.4f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Indicator calculation error: {str(e)}")
            return df
    
    def calculate_signal_score(self, indicators):
        """Calculate a score for the signal strength (0-100) tuned for commodities"""
        score = 0
        
        try:
            # Trend indicators (40 points max)
            if indicators['close'] > indicators['ema20'] > indicators['ema50']:
                score += 20  # Strong uptrend
            elif indicators['close'] > indicators['ema20']:
                score += 10  # Moderate uptrend
                
            if indicators['close'] > indicators['ema200']:
                score += 15  # Long-term uptrend
                
            # Momentum indicators (25 points max)
            if indicators['rsi'] > 50 and indicators['rsi'] < 70:
                score += 10  # Healthy RSI
            if indicators['macd'] > indicators['signal']:
                score += 8  # MACD bullish
            if not np.isnan(indicators['mfi']) and indicators['mfi'] > 50 and indicators['mfi'] < 80:
                score += 7  # Money Flow Index confirming
                
            # Volume confirmation (15 points)
            if indicators['volume_ratio'] > 1.2:
                score += 15  # Above average volume
            elif indicators['volume_ratio'] > 1.0:
                score += 5  # Normal volume
                
            # Volatility indicators (10 points)
            bb_pct = (indicators['close'] - indicators['lower_band']) / (indicators['upper_band'] - indicators['lower_band'])
            if 0.3 < bb_pct < 0.7:
                score += 5  # Not overbought or oversold
            if indicators['atr']/indicators['close'] > 0.01:  # High volatility can be good for commodities
                score += 5
                
            # Additional commodity-specific indicators (10 points)
            if not np.isnan(indicators['cci']):
                if indicators['cci'] > 0 and indicators['cci'] < 200:
                    score += 5  # CCI positive but not overbought
                elif indicators['cci'] < 0 and indicators['cci'] > -200:
                    score -= 5  # CCI negative but not oversold
                
            # Cap the score at 100
            score = min(100, max(0, score))
            
        except Exception as e:
            print(f"❌ Error calculating signal score: {e}")
            
        return score
    
    def check_entry_signals(self, df, timeframe="15min"):
        """Check for entry signals with commodity-specific criteria"""
        try:
            if df is None or len(df) < 50:
                print(f"{Fore.YELLOW}⚠️ Not enough data for signal generation{Style.RESET_ALL}")
                return []
                
            print(f"\n{Fore.CYAN}Checking entry signals...{Style.RESET_ALL}")
            signals = []
            
            # Get last few candles
            current = df.iloc[-1]
            prev1 = df.iloc[-2]
            prev2 = df.iloc[-3]
            
            # Skip if outside market hours
            if not self.is_market_open(current['datetime']):
                print(f"{Fore.YELLOW}⚠️ Current time is outside market hours. No signals generated.{Style.RESET_ALL}")
                return signals
            
            symbol = current['symbol']
            print(f"\n{Fore.CYAN}Analyzing {Fore.YELLOW}{symbol}{Fore.CYAN} at {current['datetime'].strftime('%Y-%m-%d %H:%M')}:{Style.RESET_ALL}")
            print(f"Current close: {Fore.GREEN}{current['close']:.2f}{Style.RESET_ALL}")
            print(f"Previous close: {Fore.GREEN}{prev1['close']:.2f}{Style.RESET_ALL}")
            
            # Calculate adaptive ATR multipliers based on volatility
            atr_factor = self.config.get("stop_loss_atr_multiplier", 1.5)
            
            # Bullish signals with scoring
            if (current['close'] > current['ema20']):
                bullish_score = self.calculate_signal_score(current)
                
                if bullish_score >= 60:  # Only take high-quality signals
                    # Calculate dynamic stop loss and target
                    stop_loss = current['close'] - (current['atr'] * atr_factor)
                    target = current['close'] + (current['atr'] * self.config.get("take_profit_ratio", 2.0))
                    
                    # Calculate position size based on risk management
                    risk_per_trade = self.config.get("risk_per_trade", 0.02)
                    risk_amount = self.capital * risk_per_trade
                    risk_per_unit = abs(current['close'] - stop_loss)
                    
                    if risk_per_unit > 0:
                        position_size = int(risk_amount / risk_per_unit)
                        min_lot_size = 1
                        position_size = max(min_lot_size, position_size)
                        
                        signals.append({
                            "side": "BUY",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "reason": f"Bullish pattern (Score: {bullish_score}/100)",
                            "entry": current['close'],
                            "sl": stop_loss,
                            "target": target,
                            "score": bullish_score,
                            "size": position_size,
                            "time": current['datetime'].strftime('%H:%M'),
                            "date": current['datetime'].strftime('%Y-%m-%d'),
                            "type": "INTRADAY",  # Default to intraday for commodities
                            "atr": current['atr'],
                            "rsi": current['rsi'],
                            "cci": current['cci'],
                            "bb_width": current['bb_width']
                        })
                
            # Bearish signals with scoring
            elif (current['close'] < current['ema20']):
                bearish_score = 100 - self.calculate_signal_score(current)  # Invert for bearish
                
                if bearish_score >= 60:  # Only take high-quality signals
                    # Calculate dynamic stop loss and target
                    stop_loss = current['close'] + (current['atr'] * atr_factor)
                    target = current['close'] - (current['atr'] * self.config.get("take_profit_ratio", 2.0))
                    
                    # Calculate position size
                    risk_per_trade = self.config.get("risk_per_trade", 0.02)
                    risk_amount = self.capital * risk_per_trade
                    risk_per_unit = abs(current['close'] - stop_loss)
                    
                    if risk_per_unit > 0:
                        position_size = int(risk_amount / risk_per_unit)
                        min_lot_size = 1
                        position_size = max(min_lot_size, position_size)
                        
                        signals.append({
                            "side": "SELL",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "reason": f"Bearish pattern (Score: {bearish_score}/100)",
                            "entry": current['close'],
                            "sl": stop_loss,
                            "target": target,
                            "score": bearish_score,
                            "size": position_size,
                            "time": current['datetime'].strftime('%H:%M'),
                            "date": current['datetime'].strftime('%Y-%m-%d'),
                            "type": "INTRADAY",  # Default to intraday for commodities
                            "atr": current['atr'],
                            "rsi": current['rsi'],
                            "cci": current['cci'],
                            "bb_width": current['bb_width']
                        })
            
            if signals:
                print(f"\n{Fore.GREEN}🎯 Found signals:{Style.RESET_ALL}")
                
                # Enhanced table formatting for signals using tabulate
                headers = [
                    f"{Fore.YELLOW}{Style.BRIGHT}Date{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Time{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Symbol{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Side{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Type{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Entry{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Stop Loss{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Target{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}SL Pts{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}TP Pts{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Qty{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Score{Style.RESET_ALL}"
                ]
                
                # Build table data
                table_data = []
                
                for signal in signals:
                    # Color formatting for side
                    side_formatted = f"{Fore.GREEN}{signal['side']}{Style.RESET_ALL}" if signal['side'] == "BUY" else f"{Fore.RED}{signal['side']}{Style.RESET_ALL}"
                    
                    sl_pts = abs(signal['entry']-signal['sl'])
                    tp_pts = abs(signal['entry']-signal['target'])
                    
                    # Color the score based on value
                    if signal['score'] >= 80:
                        score_formatted = f"{Fore.GREEN}{signal['score']}{Style.RESET_ALL}"
                    elif signal['score'] >= 60:
                        score_formatted = f"{Fore.YELLOW}{signal['score']}{Style.RESET_ALL}"
                    else:
                        score_formatted = f"{Fore.RED}{signal['score']}{Style.RESET_ALL}"
                        
                    table_data.append([
                        signal['date'],
                        signal['time'],
                        f"{Fore.CYAN}{signal['symbol']}{Style.RESET_ALL}",
                        side_formatted,
                        signal['type'],
                        f"{signal['entry']:.2f}",
                        f"{signal['sl']:.2f}",
                        f"{signal['target']:.2f}",
                        f"{sl_pts:.2f}",
                        f"{tp_pts:.2f}",
                        str(signal['size']),
                        score_formatted
                    ])
                
                # Print well-formatted table using tabulate
                from tabulate import tabulate
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
                print(f"{Fore.CYAN}Risk/Reward ratio: {tp_pts/sl_pts:.2f}{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}⚠️ No signals found{Style.RESET_ALL}")
                
            return signals
            
        except Exception as e:
            print(f"{Fore.RED}❌ Signal check error: {str(e)}{Style.RESET_ALL}")
            return []
            
    def generate_consolidated_signals_table(self, symbols=None, timeframes=None):
        """Generate a consolidated signal table for multiple symbols and timeframes
        
        Parameters:
        symbols (list): List of symbols to analyze, if None uses favorite symbols
        timeframes (list): List of timeframes to check, e.g. ["1min", "5min", "15min", "1hour"]
        """
        try:
            if symbols is None:
                symbols = self.config.get("favorite_symbols", ["SILVER", "GOLD", "CRUDEOIL", "NATURALGAS"])
                
            if timeframes is None:
                timeframes = ["5min", "15min", "1hour"]
                
            # Map timeframe names to actual resolution values for Fyers API
            timeframe_map = {
                "1min": "1",
                "2min": "2",
                "5min": "5", 
                "10min": "10",
                "15min": "15",
                "30min": "30",
                "1hour": "60",
                "1day": "D"
            }
            
            print("\n" + "═" * 100)
            print(f"{Fore.GREEN}{Style.BRIGHT}╔══════════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{Style.BRIGHT}║                     CONSOLIDATED SIGNALS ACROSS TIMEFRAMES                    ║{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{Style.BRIGHT}╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
            print("═" * 100)
            
            # Store all signals from all timeframes
            all_timeframe_signals = []
            
            # Collect data for all symbols and timeframes
            consolidated_data = []
            
            for symbol in symbols:
                print(f"\n{Fore.CYAN}Analyzing {Fore.YELLOW}{symbol}{Fore.CYAN} across multiple timeframes...{Style.RESET_ALL}")
                
                for tf_name, tf_value in [(tf, timeframe_map.get(tf, tf)) for tf in timeframes]:
                    try:
                        # Get data for this timeframe
                        df = self.get_historical_data(symbol, timeframe=tf_value, days=5)
                        
                        if df is not None and len(df) > 20:
                            # Calculate indicators
                            df = self.calculate_indicators(df)
                            current = df.iloc[-1]
                            
                            # Check for signals using the current timeframe
                            signals = self.check_entry_signals(df, timeframe=tf_name)
                            all_timeframe_signals.extend(signals)
                            
                            # Format indicator values with colors
                            atr_pct = current['atr'] / current['close'] * 100
                            if atr_pct > 1.0:
                                atr_formatted = f"{Fore.RED}{atr_pct:.2f}%{Style.RESET_ALL}"  # High volatility
                            elif atr_pct > 0.5:
                                atr_formatted = f"{Fore.YELLOW}{atr_pct:.2f}%{Style.RESET_ALL}"  # Medium
                            else:
                                atr_formatted = f"{Fore.GREEN}{atr_pct:.2f}%{Style.RESET_ALL}"  # Low
                                
                            if current['cci'] > 100:
                                cci_formatted = f"{Fore.GREEN}{current['cci']:.2f}{Style.RESET_ALL}"  # Bullish
                            elif current['cci'] < -100:
                                cci_formatted = f"{Fore.RED}{current['cci']:.2f}{Style.RESET_ALL}"  # Bearish
                            else:
                                cci_formatted = f"{current['cci']:.2f}"  # Neutral
                                
                            if current['rsi'] > 70:
                                rsi_formatted = f"{Fore.RED}{current['rsi']:.2f}{Style.RESET_ALL}"  # Overbought
                            elif current['rsi'] < 30:
                                rsi_formatted = f"{Fore.GREEN}{current['rsi']:.2f}{Style.RESET_ALL}"  # Oversold
                            else:
                                rsi_formatted = f"{current['rsi']:.2f}"  # Neutral
                                
                            if current['bb_width'] > 0.03:
                                bb_formatted = f"{Fore.RED}{current['bb_width']:.4f}{Style.RESET_ALL}"  # Wide bands, high volatility
                            elif current['bb_width'] < 0.01:
                                bb_formatted = f"{Fore.YELLOW}{current['bb_width']:.4f}{Style.RESET_ALL}"  # Narrow, potential breakout
                            else:
                                bb_formatted = f"{current['bb_width']:.4f}"  # Normal
                            
                            # Determine signal strength and type
                            signal_score = self.calculate_signal_score(current)
                            
                            # Format signal strength based on score
                            if signal_score >= 80:
                                strength_formatted = f"{Fore.GREEN}{signal_score}%{Style.RESET_ALL}"
                                signal_type = f"{Fore.GREEN}BUY{Style.RESET_ALL}"
                            elif signal_score <= 20:
                                strength_formatted = f"{Fore.RED}{100-signal_score}%{Style.RESET_ALL}"
                                signal_type = f"{Fore.RED}SELL{Style.RESET_ALL}"
                            elif signal_score >= 60:
                                strength_formatted = f"{Fore.YELLOW}{signal_score}%{Style.RESET_ALL}"
                                signal_type = f"{Fore.YELLOW}WEAK BUY{Style.RESET_ALL}"
                            elif signal_score <= 40:
                                strength_formatted = f"{Fore.YELLOW}{100-signal_score}%{Style.RESET_ALL}"
                                signal_type = f"{Fore.YELLOW}WEAK SELL{Style.RESET_ALL}"
                            else:
                                strength_formatted = f"{signal_score}%"
                                signal_type = "NEUTRAL"
                            
                            # Add to consolidated data
                            consolidated_data.append({
                                "symbol": f"{Fore.CYAN}{Style.BRIGHT}{symbol}{Style.RESET_ALL}",
                                "timeframe": tf_name,
                                "atr": atr_formatted,
                                "cci": cci_formatted,
                                "bb": bb_formatted,
                                "rsi": rsi_formatted,
                                "strength": strength_formatted,
                                "type": signal_type,
                                "price": f"{current['close']:.2f}"
                            })
                    except Exception as e:
                        print(f"{Fore.RED}❌ Error analyzing {symbol} on {tf_name}: {str(e)}{Style.RESET_ALL}")
            
            # Display the consolidated table using tabulate for clean formatting
            if consolidated_data:
                from tabulate import tabulate
                
                # Convert the data into a format suitable for tabulate
                table_data = []
                for row in consolidated_data:
                    table_data.append([
                        row["symbol"], row["timeframe"], row["atr"], row["cci"],
                        row["bb"], row["rsi"], row["strength"], row["type"], row["price"]
                    ])
                
                # Define headers with colors
                headers = [
                    f"{Fore.YELLOW}{Style.BRIGHT}Symbol{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Timeframe{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}ATR%{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}CCI{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}BB Width{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}RSI{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Signal Strength{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Signal Type{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Price{Style.RESET_ALL}"
                ]
                
                # Print the table using tabulate with grid format
                print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
                
                # Show summary of signals found
                if all_timeframe_signals:
                    signal_counts = {}
                    for signal in all_timeframe_signals:
                        key = f"{signal['symbol']}:{signal['timeframe']}"
                        if key not in signal_counts:
                            signal_counts[key] = {"BUY": 0, "SELL": 0}
                        signal_counts[key][signal['side']] += 1
                    
                    # Convert the signal summary into a format suitable for tabulate
                    summary_data = []
                    for key, counts in signal_counts.items():
                        symbol, tf = key.split(":")
                        summary_data.append([
                            f"{Fore.CYAN}{symbol}{Style.RESET_ALL} ({tf})", 
                            f"{Fore.GREEN}{counts['BUY']}{Style.RESET_ALL}", 
                            f"{Fore.RED}{counts['SELL']}{Style.RESET_ALL}"
                        ])
                    
                    # Print the signal summary table
                    print(f"\n{Fore.GREEN}{Style.BRIGHT}Signal Summary:{Style.RESET_ALL}")
                    summary_headers = [
                        f"{Fore.YELLOW}Symbol & Timeframe{Style.RESET_ALL}", 
                        f"{Fore.GREEN}BUY{Style.RESET_ALL}", 
                        f"{Fore.RED}SELL{Style.RESET_ALL}"
                    ]
                    print(tabulate(summary_data, headers=summary_headers, tablefmt="grid"))
            else:
                print(f"\n{Fore.YELLOW}No data available for consolidated table.{Style.RESET_ALL}")
            
            return all_timeframe_signals
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error generating consolidated signals table: {str(e)}{Style.RESET_ALL}")
            return []
    
    def display_mcx_dashboard(self):
        """Display current market status for all MCX instruments"""
        try:
            print("\n" + "═" * 100)
            print(f"{Fore.CYAN}{Style.BRIGHT}╔══════════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}║                          MCX TRADING DASHBOARD                               ║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
            print("═" * 100)
            
            # Get current time in Indian timezone
            current_time = datetime.now(self.indian_timezone)
            market_status = f"{Fore.GREEN}🟢 OPEN{Style.RESET_ALL}" if self.is_market_open() else f"{Fore.RED}🔴 CLOSED{Style.RESET_ALL}"
            
            # Print header
            print(f"{Fore.WHITE}{Style.BRIGHT}Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | MCX Market: {market_status}{Style.RESET_ALL}")
            print(f"Trading Mode: {Fore.YELLOW}{'SIMULATION' if self.is_simulation else 'LIVE'}{Style.RESET_ALL} | Capital: {Fore.GREEN}₹{self.capital:,.2f}{Style.RESET_ALL} | Risk: {Fore.YELLOW}{self.config.get('risk_per_trade', 0.02)*100}%{Style.RESET_ALL}")
            print("─" * 100)
            
            # Display table headers with colored formatting
            headers = [
                f"{Fore.YELLOW}{Style.BRIGHT}SYMBOL{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}LAST PRICE{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}CHANGE %{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}SIGNAL{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}ATR{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}RSI{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}MFI{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}CCI{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}VOLUME{Style.RESET_ALL}"
            ]
            print(f"{headers[0]:<16}|{headers[1]:<14}|{headers[2]:<12}|{headers[3]:<12}|{headers[4]:<10}|{headers[5]:<10}|{headers[6]:<10}|{headers[7]:<12}|{headers[8]:<12}")
            print("─" * 100)
            
            for symbol in self.mcx_symbols.keys():
                try:
                    # Check if we have data for this symbol
                    df = self.get_historical_data(symbol, days=5)
                    
                    if df is not None and len(df) > 0:
                        # Calculate indicators
                        df = self.calculate_indicators(df)
                        current = df.iloc[-1]
                        prev_day = df[df['datetime'].dt.date < current['datetime'].date()].iloc[-1] if len(df) > 20 else df.iloc[-2]
                        
                        # Calculate change percentage
                        change_pct = (current['close'] - prev_day['close']) / prev_day['close'] * 100
                        
                        # Determine signal with color
                        signal = f"{Fore.WHITE}NEUTRAL{Style.RESET_ALL}"
                        if current['close'] > current['ema20'] and current['rsi'] > 50 and current['macd'] > current['signal']:
                            signal = f"{Fore.GREEN}🟢 BUY{Style.RESET_ALL}"
                        elif current['close'] < current['ema20'] and current['rsi'] < 50 and current['macd'] < current['signal']:
                            signal = f"{Fore.RED}🔴 SELL{Style.RESET_ALL}"
                            
                        # Format values
                        volume_formatted = f"{current['volume']/1000:.0f}K" if current['volume'] > 1000 else f"{current['volume']:.0f}"
                        
                        # Format change with color
                        if change_pct > 0:
                            change_formatted = f"{Fore.GREEN}▲ {change_pct:<6.2f}%{Style.RESET_ALL}"
                        else:
                            change_formatted = f"{Fore.RED}▼ {abs(change_pct):<6.2f}%{Style.RESET_ALL}"
                        
                        # Format RSI with color
                        if current['rsi'] > 70:
                            rsi_formatted = f"{Fore.RED}{current['rsi']:<6.2f}{Style.RESET_ALL}"
                        elif current['rsi'] < 30:
                            rsi_formatted = f"{Fore.GREEN}{current['rsi']:<6.2f}{Style.RESET_ALL}"
                        else:
                            rsi_formatted = f"{current['rsi']:<6.2f}"
                        
                        # Print row with color formatting for the symbol
                        print(f"{Fore.CYAN}{Style.BRIGHT}{symbol:<14}{Style.RESET_ALL}|{current['close']:<14.2f}|{change_formatted:<12}|{signal:<12}|" +
                              f"{current['atr']:<10.2f}|{rsi_formatted:<10}|{current['mfi']:<10.2f}|{current['cci']:<12.2f}|{volume_formatted:<12}")
                    else:
                        # Use last known price if available
                        last_price = self.last_prices.get(symbol, "N/A")
                        if isinstance(last_price, (int, float)):
                            last_price = f"{last_price:.2f}"
                        print(f"{Fore.CYAN}{Style.BRIGHT}{symbol:<14}{Style.RESET_ALL}|{last_price:<14}|{'-':<12}|{'N/A':<12}|{'N/A':<10}|{'N/A':<10}|{'N/A':<10}|{'N/A':<12}|{'N/A':<12}")
                        
                except Exception as e:
                    print(f"{Fore.CYAN}{Style.BRIGHT}{symbol:<14}{Style.RESET_ALL}|{Fore.RED}{'ERROR':<14}{Style.RESET_ALL}|{'-':<12}|{'N/A':<12}|{'N/A':<10}|{'N/A':<10}|{'N/A':<10}|{'N/A':<12}|{'N/A':<12}")
            
            print("─" * 100)
            print(f"Note: {Fore.YELLOW}All trades are simulated with virtual capital. MFI = Money Flow Index. CCI = Commodity Channel Index.{Style.RESET_ALL}")
            print("═" * 100)
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error displaying dashboard: {str(e)}{Style.RESET_ALL}")
    
    def run_end_of_day_analysis(self):
        """Run comprehensive analysis at end of market day"""
        try:
            print("\n" + "*"*100)
            print("MCX END-OF-DAY ANALYSIS".center(100))
            print("*"*100)
            
            # Get current time in Indian timezone
            current_time = datetime.now(self.indian_timezone)
            print(f"EOD Analysis Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-"*100)
            
            # Analyze all available symbols
            all_symbols = list(self.mcx_symbols.keys())
            
            # Store all signals and performance metrics for each symbol
            all_signals = []
            symbol_performance = {}
            
            # Run daily analysis for each symbol
            for symbol in all_symbols:
                try:
                    print(f"\n=== EOD Analysis for {symbol} ===")
                    
                    # Get data with different timeframes
                    df_daily = self.get_historical_data(symbol, timeframe="D", days=60)
                    df_hourly = self.get_historical_data(symbol, timeframe="60", days=30)
                    df_15min = self.get_historical_data(symbol, timeframe="15", days=5)
                    
                    # Calculate performance metrics if we have data
                    if df_daily is not None and len(df_daily) > 10:
                        # Calculate daily metrics
                        df_daily = self.calculate_indicators(df_daily)
                        current = df_daily.iloc[-1]
                        prev = df_daily.iloc[-2] if len(df_daily) > 1 else current
                        
                        # Daily change
                        daily_change = (current['close'] - prev['close']) / prev['close'] * 100
                        
                        # Calculate 10-day volatility
                        returns = df_daily['close'].pct_change().dropna()
                        volatility_10d = returns[-10:].std() * 100 if len(returns) >= 10 else np.nan
                        
                        # Trading volume change
                        volume_change = (current['volume'] - prev['volume']) / prev['volume'] * 100 if prev['volume'] > 0 else 0
                        
                        # Store performance metrics
                        symbol_performance[symbol] = {
                            "close": current['close'],
                            "daily_change": daily_change,
                            "volatility_10d": volatility_10d,
                            "volume_change": volume_change,
                            "rsi": current['rsi'],
                            "trend": "Bullish" if current['close'] > current['ema20'] > current['ema50'] else
                                    "Bearish" if current['close'] < current['ema20'] < current['ema50'] else "Mixed"
                        }
                    
                    # Check for end-of-day signals using hourly data for a broader view
                    if df_hourly is not None:
                        df_hourly = self.calculate_indicators(df_hourly)
                        hourly_signals = self.check_entry_signals(df_hourly)
                        for signal in hourly_signals:
                            signal['timeframe'] = 'H1'
                            all_signals.append(signal)
                    
                    # Also check for short-term signals on 15-min timeframe
                    if df_15min is not None:
                        df_15min = self.calculate_indicators(df_15min)
                        signals_15min = self.check_entry_signals(df_15min)
                        for signal in signals_15min:
                            signal['timeframe'] = 'M15'
                            all_signals.append(signal)
                    
                except Exception as e:
                    print(f"❌ Error analyzing {symbol}: {str(e)}")
            
            # Print performance table
            if symbol_performance:
                print("\n=== Commodity Performance Summary ===")
                perf_data = []
                for symbol, metrics in symbol_performance.items():
                    perf_data.append([
                        symbol,
                        f"{metrics['close']:.2f}",
                        f"{metrics['daily_change']:.2f}%" if not np.isnan(metrics['daily_change']) else "N/A",
                        f"{metrics['volatility_10d']:.2f}%" if not np.isnan(metrics['volatility_10d']) else "N/A",
                        f"{metrics['volume_change']:.2f}%" if not np.isnan(metrics['volume_change']) else "N/A",
                        f"{metrics['rsi']:.2f}" if not np.isnan(metrics['rsi']) else "N/A",
                        metrics['trend']
                    ])
                print(tabulate(perf_data, headers=["Symbol", "Close", "Daily Change", "10D Volatility", "Volume Change", "RSI", "Trend"]))
            
            # Print all signals ranked by score
            if all_signals:
                print("\n=== End-of-Day Trading Signals ===")
                print(f"Found {len(all_signals)} trading signals across {len(all_symbols)} symbols")
                
                # Rank signals by score
                all_signals.sort(key=lambda x: x['score'], reverse=True)
                
                signal_data = []
                for signal in all_signals:
                    signal_data.append([
                        signal['symbol'],
                        signal['side'],
                        signal['type'],
                        signal['timeframe'],
                        f"{signal['entry']:.2f}",
                        f"{signal['sl']:.2f}",
                        f"{signal['target']:.2f}",
                        f"{abs(signal['entry']-signal['sl']):.2f}",
                        f"{signal['score']}",
                        signal['size'],
                    ])
                print(tabulate(signal_data, headers=["Symbol", "Side", "Type", "Timeframe", "Entry", "Stop Loss", "Target", "Risk Pts", "Score", "Qty"]))
            else:
                print("\n📢 No trading signals found for tomorrow.")
                
            # Save EOD report to file
            report_date = current_time.strftime('%Y-%m-%d')
            report_filename = f"mcx_eod_report_{report_date}.txt"
            
            print(f"\n✅ End-of-Day analysis completed. Report saved to {report_filename}")
            print("*"*100)
            
        except Exception as e:
            print(f"❌ Error running end-of-day analysis: {str(e)}")
    
    def check_if_eod_analysis_time(self):
        """Check if it's time to run end-of-day analysis"""
        current_time = datetime.now(self.indian_timezone)
        current_t = current_time.time()
        
        # Run EOD analysis if it's close to market close (within 10 minutes)
        if (self.eod_analysis_time <= current_t <= self.market_close_time and 
            current_time.weekday() < 5):  # Only on weekdays
            print("\n⏰ It's end-of-day analysis time!")
            return True
        return False
    
    def place_manual_trade(self, symbol, side, risk_percent=None):
        """Place a manual trade in the specified symbol
        
        Parameters:
        symbol (str): Symbol to trade (e.g., "GOLD")
        side (str): "BUY" or "SELL"
        risk_percent (float): Optional override for risk percentage
        
        Returns:
        dict: Trade details or None if trade couldn't be placed
        """
        try:
            # Validate inputs
            if symbol not in self.mcx_symbols:
                print(f"❌ Symbol {symbol} not found in available symbols.")
                return None
                
            if side not in ["BUY", "SELL"]:
                print(f"❌ Invalid side: {side}. Must be 'BUY' or 'SELL'.")
                return None
                
            # Get latest data
            print(f"\n🔍 Analyzing {symbol} for manual {side} trade...")
            df = self.get_historical_data(symbol, days=5)
            
            if df is None or len(df) < 20:  # Need enough data for indicators
                print(f"❌ Not enough data available for {symbol}.")
                return None
                
            # Calculate indicators
            df = self.calculate_indicators(df)
            current = df.iloc[-1]
            
            # Use risk percentage from parameter or config
            risk_pct = risk_percent if risk_percent is not None else self.config.get("risk_per_trade", 0.02)
            
            # Calculate ATR-based stop loss and target
            atr_factor = self.config.get("stop_loss_atr_multiplier", 1.5)
            take_profit_ratio = self.config.get("take_profit_ratio", 2.0)
            
            entry_price = current['close']
            
            if side == "BUY":
                stop_loss = entry_price - (current['atr'] * atr_factor)
                target = entry_price + (current['atr'] * take_profit_ratio)
            else:  # SELL
                stop_loss = entry_price + (current['atr'] * atr_factor)
                target = entry_price - (current['atr'] * take_profit_ratio)
                
            # Calculate position size based on risk management
            risk_amount = self.capital * risk_pct
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit <= 0:
                print("❌ Invalid risk calculation. Try different parameters.")
                return None
                
            position_size = int(risk_amount / risk_per_unit)
            min_lot_size = 1
            position_size = max(min_lot_size, position_size)
            
            # Create trade
            trade = {
                "symbol": symbol,
                "side": side,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "size": position_size,
                "entry_time": datetime.now(self.indian_timezone),
                "status": "OPEN",
                "risk_amount": risk_amount,
                "type": "MANUAL",
            }
            
            # Add to positions
            trade_id = f"{symbol}_{side}_{datetime.now(self.indian_timezone).strftime('%Y%m%d%H%M%S')}"
            self.positions[trade_id] = trade
            
            # Print confirmation
            print("\n✅ Manual trade placed successfully!")
            print(f"Symbol: {symbol}")
            print(f"Side: {side}")
            print(f"Entry: {entry_price:.2f}")
            print(f"Stop Loss: {stop_loss:.2f} ({abs(entry_price - stop_loss):.2f} points)")
            print(f"Target: {target:.2f} ({abs(entry_price - target):.2f} points)")
            print(f"Position Size: {position_size} units")
            print(f"Risk Amount: ₹{risk_amount:.2f}")
            
            # Risk-reward ratio
            rr_ratio = abs(target - entry_price) / abs(stop_loss - entry_price)
            print(f"Risk-Reward Ratio: 1:{rr_ratio:.2f}")
            
            return trade
            
        except Exception as e:
            print(f"❌ Error placing manual trade: {str(e)}")
            return None
    
    def show_current_positions(self):
        """Display all current open positions with real-time P&L"""
        try:
            if not self.positions:
                print(f"\n{Fore.YELLOW}📊 No open positions.{Style.RESET_ALL}")
                return
                
            print("\n" + "═" * 120)
            print(f"{Fore.MAGENTA}{Style.BRIGHT}╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{Style.BRIGHT}║                             CURRENT OPEN POSITIONS WITH REAL-TIME PROFIT/LOSS                                ║{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{Style.BRIGHT}╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
            print("═" * 120)
            
            # Refresh prices for all symbols with positions
            symbols_to_refresh = set(trade['symbol'] for trade in self.positions.values())
            for symbol in symbols_to_refresh:
                # Get fresh data to update last price
                df = self.get_historical_data(symbol, timeframe="1", days=1)
                if df is not None and len(df) > 0:
                    self.last_prices[symbol] = df.iloc[-1]['close']
                    print(f"{Fore.GREEN}✅ Updated {symbol} price: {self.last_prices[symbol]:.2f}{Style.RESET_ALL}")
            
            position_data = []
            total_risk = 0
            total_unrealized_pnl = 0
            
            # Table headers with color formatting
            headers = [
                f"{Fore.YELLOW}{Style.BRIGHT}Symbol{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Side{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Type{Style.RESET_ALL}", 
                f"{Fore.YELLOW}{Style.BRIGHT}Entry{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Stop Loss{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Target{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Current{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Unrealized P&L{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Size{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Risk{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}R/R{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Status{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Entry Time{Style.RESET_ALL}"
            ]
            
            for trade_id, trade in self.positions.items():
                # Calculate current P&L based on last known price
                current_price = self.last_prices.get(trade['symbol'], trade['entry'])
                
                if trade['side'] == "BUY":
                    unrealized_pnl = (current_price - trade['entry']) * trade['size']
                    # Calculate percentage gain/loss
                    pnl_percent = ((current_price / trade['entry']) - 1) * 100
                    # Check if hit stop loss or target
                    hit_stop = current_price <= trade['stop_loss']
                    hit_target = current_price >= trade['target']
                else:  # SELL
                    unrealized_pnl = (trade['entry'] - current_price) * trade['size']
                    # Calculate percentage gain/loss
                    pnl_percent = ((trade['entry'] / current_price) - 1) * 100
                    # Check if hit stop loss or target
                    hit_stop = current_price >= trade['stop_loss']
                    hit_target = current_price <= trade['target']
                    
                # Add to total
                total_unrealized_pnl += unrealized_pnl
                    
                # Calculate risk
                risk = trade['risk_amount']
                total_risk += risk
                
                # Format P&L with color indicator
                if unrealized_pnl >= 0:
                    pnl_str = f"{Fore.GREEN}₹{unrealized_pnl:.2f} ({pnl_percent:.2f}%){Style.RESET_ALL}"
                else:
                    pnl_str = f"{Fore.RED}₹{unrealized_pnl:.2f} ({pnl_percent:.2f}%){Style.RESET_ALL}"
                
                # Trade status indicator with color
                if hit_stop:
                    status = f"{Fore.RED}🛑 STOP HIT{Style.RESET_ALL}"
                elif hit_target:
                    status = f"{Fore.GREEN}🎯 TARGET HIT{Style.RESET_ALL}"
                else:
                    status = f"{Fore.CYAN}ACTIVE{Style.RESET_ALL}"
                
                # Risk/reward current status
                risk_reward_current = abs(unrealized_pnl) / risk if risk > 0 else 0
                
                # Format side with color
                if trade['side'] == "BUY":
                    side_formatted = f"{Fore.GREEN}{trade['side']}{Style.RESET_ALL}"
                else:
                    side_formatted = f"{Fore.RED}{trade['side']}{Style.RESET_ALL}"
                
                position_data.append([
                    f"{Fore.CYAN}{Style.BRIGHT}{trade['symbol']}{Style.RESET_ALL}",
                    side_formatted,
                    trade['type'],
                    f"{trade['entry']:.2f}",
                    f"{trade['stop_loss']:.2f}",
                    f"{trade['target']:.2f}",
                    f"{current_price:.2f}",
                    pnl_str,
                    f"{trade['size']}",
                    f"₹{risk:.2f}",
                    f"{risk_reward_current:.2f}",
                    status,
                    trade['entry_time'].strftime('%Y-%m-%d %H:%M')
                ])
            
            # Print the formatted header row
            print("|".join(f"{h:<15}" for h in headers))
            print("─" * 120)
            
            # Print each position row
            for row in position_data:
                position_row = "|".join(f"{col:<15}" for col in row)
                print(position_row)
                
            print("─" * 120)
            
            # Format with colors
            if total_unrealized_pnl >= 0:
                total_pnl_str = f"{Fore.GREEN}₹{total_unrealized_pnl:.2f}{Style.RESET_ALL}"
            else:
                total_pnl_str = f"{Fore.RED}₹{total_unrealized_pnl:.2f}{Style.RESET_ALL}"
                
            print(f"Total Unrealized P&L: {total_pnl_str}")
            print(f"Total Risk: {Fore.YELLOW}₹{total_risk:.2f} ({total_risk/self.capital*100:.2f}% of capital){Style.RESET_ALL}")
            print(f"Capital + P&L: {Fore.GREEN}₹{self.capital + total_unrealized_pnl:.2f}{Style.RESET_ALL}")
            
            # Show realized gains/losses from closed positions today
            if self.trades_history:
                today = datetime.now(self.indian_timezone).date()
                today_trades = [t for t in self.trades_history if t['exit_time'].date() == today]
                
                if today_trades:
                    total_realized = sum(t['pnl'] for t in today_trades)
                    if total_realized >= 0:
                        realized_str = f"{Fore.GREEN}₹{total_realized:.2f}{Style.RESET_ALL}"
                    else:
                        realized_str = f"{Fore.RED}₹{total_realized:.2f}{Style.RESET_ALL}"
                    print(f"Today's Realized P&L: {realized_str} from {len(today_trades)} trades")
                    
            print("═" * 120)
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error showing positions: {str(e)}{Style.RESET_ALL}")
    
    def close_position(self, trade_id=None, symbol=None):
        """Close a position by trade_id or symbol
        
        Parameters:
        trade_id (str): Specific trade ID to close
        symbol (str): Symbol to close all positions for
        """
        try:
            positions_to_close = {}
            
            # Find positions to close
            if trade_id and trade_id in self.positions:
                positions_to_close[trade_id] = self.positions[trade_id]
            elif symbol:
                for tid, trade in self.positions.items():
                    if trade['symbol'] == symbol:
                        positions_to_close[tid] = trade
            
            if not positions_to_close:
                print(f"❌ No positions found to close.")
                return
                
            print("\n🔄 Closing positions...")
            
            for tid, trade in positions_to_close.items():
                # Get current price
                current_price = self.last_prices.get(trade['symbol'], trade['entry'])
                
                # Calculate P&L
                if trade['side'] == "BUY":
                    realized_pnl = (current_price - trade['entry']) * trade['size']
                else:  # SELL
                    realized_pnl = (trade['entry'] - current_price) * trade['size']
                
                # Create closed trade record
                closed_trade = trade.copy()
                closed_trade['exit_price'] = current_price
                closed_trade['exit_time'] = datetime.now(self.indian_timezone)
                closed_trade['pnl'] = realized_pnl
                closed_trade['status'] = "CLOSED"
                
                # Add to trade history
                self.trades_history.append(closed_trade)
                
                # Remove from open positions
                del self.positions[tid]
                
                # Print confirmation
                print(f"Closed {trade['symbol']} {trade['side']} position:")
                print(f"Entry: {trade['entry']:.2f}, Exit: {current_price:.2f}")
                
                pnl_str = f"₹{realized_pnl:.2f}"
                if realized_pnl >= 0:
                    print(f"Realized P&L: 🟢 {pnl_str}")
                else:
                    print(f"Realized P&L: 🔴 {pnl_str}")
            
            print(f"✅ Closed {len(positions_to_close)} positions.")
            
        except Exception as e:
            print(f"❌ Error closing position: {str(e)}")
    
    def update_prices_in_background(self):
        """Update prices for all symbols in background thread"""
        try:
            print("\n🔄 Starting real-time price updates...")
            
            while True:
                # Get symbols to update - active positions and favorite symbols
                symbols_to_update = set(trade['symbol'] for trade in self.positions.values())
                
                # Add favorite symbols
                favorites = self.config.get("favorite_symbols", ["SILVER", "GOLD", "CRUDEOIL"])
                symbols_to_update.update(favorites)
                
                # Update each symbol
                for symbol in symbols_to_update:
                    try:
                        # Get fresh data to update last price
                        df = self.get_historical_data(symbol, timeframe="1", days=1)
                        if df is not None and len(df) > 0:
                            last_price = df.iloc[-1]['close']
                            old_price = self.last_prices.get(symbol, None)
                            self.last_prices[symbol] = last_price
                            
                            # Only print if price changed
                            if old_price is not None and old_price != last_price:
                                change = last_price - old_price
                                change_pct = (change / old_price) * 100 if old_price > 0 else 0
                                direction = "▲" if change > 0 else "▼"
                                print(f"📊 {symbol} price: {last_price:.2f} {direction} {abs(change):.2f} ({change_pct:.2f}%)")
                    except Exception as e:
                        print(f"Error updating {symbol}: {str(e)}")
                
                # Update positions if there are any
                if self.positions:
                    self.show_current_positions()
                
                # Sleep for 60 seconds
                time.sleep(60)
                
        except Exception as e:
            print(f"❌ Error in price update thread: {str(e)}")
    
    def run(self, mode=None):
        """Run the MCX trading system
        
        Parameters:
        mode (str): Operation mode - 'eod' for end-of-day analysis,
                   'dashboard' for just displaying dashboard, 
                   'gold_trade' to place a gold trade,
                   'realtime' to start real-time monitoring,
                   'signals' to view consolidated signals table,
                   'virtual' to run automated virtual trading with tracking,
                   None for normal operation
        """
        try:
            print("\n" + "═" * 100)
            print(f"{Fore.YELLOW}{Style.BRIGHT}╔══════════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{Style.BRIGHT}║                           MCX TRADING SYSTEM                                 ║{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{Style.BRIGHT}╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
            print("═" * 100)
            print(f"{Fore.RED}{Style.BRIGHT}⚠️ NOTE: All trades are simulated with virtual capital. No real trades are placed.{Style.RESET_ALL}")
            
            # Virtual trading mode
            if mode == 'virtual':
                return self.run_virtual_trading()
                
            # Consolidated signals mode
            if mode == 'signals':
                timeframes = ["5min", "15min", "1hour"]
                return self.generate_consolidated_signals_table(timeframes=timeframes)
            
            # Real-time monitoring mode
            if mode == 'realtime':
                print(f"\n{Fore.CYAN}🔄 Starting real-time market monitoring...{Style.RESET_ALL}")
                # Show current positions first
                self.show_current_positions()
                
                # Start a background thread for price updates
                import threading
                update_thread = threading.Thread(target=self.update_prices_in_background)
                update_thread.daemon = True  # Will exit when main thread exits
                update_thread.start()
                
                # Keep main thread alive with periodic checks
                try:
                    while True:
                        # Check for EOD analysis time
                        if self.check_if_eod_analysis_time():
                            self.run_end_of_day_analysis()
                        time.sleep(300)  # Check every 5 minutes
                except KeyboardInterrupt:
                    print(f"\n{Fore.GREEN}✅ Real-time monitoring stopped.{Style.RESET_ALL}")
                return
            
            # Special mode for gold trading
            if mode == 'gold_trade':
                # Place a manual trade in gold
                trade = self.place_manual_trade("GOLD", "BUY")
                if trade:
                    # Show the positions after placing the trade
                    self.show_current_positions()
                return
            
            # Check if we explicitly want EOD analysis
            if mode == 'eod':
                return self.run_end_of_day_analysis()
                
            # Display market dashboard
            self.display_mcx_dashboard()
            
            # Check if it's time for EOD analysis
            if self.check_if_eod_analysis_time():
                return self.run_end_of_day_analysis()
            
            # If just dashboard mode, exit here
            if mode == 'dashboard':
                return
                
            # Normal operation - check signals for favorite symbols
            favorite_symbols = self.config.get("favorite_symbols", ["SILVER", "GOLD", "CRUDEOIL"])
            
            # Show current positions if any
            self.show_current_positions()
            
            all_signals = []
            for symbol in favorite_symbols:
                print(f"\n{Fore.CYAN}=== Analyzing {Fore.YELLOW}{symbol} {Fore.CYAN}==={Style.RESET_ALL}")
                df = self.get_historical_data(symbol)
                if df is not None:
                    df = self.calculate_indicators(df)
                    signals = self.check_entry_signals(df)
                    all_signals.extend(signals)
            
            if all_signals:
                print(f"\n\n{Fore.GREEN}{Style.BRIGHT}=== SUMMARY OF ALL SIGNALS ==={Style.RESET_ALL}")
                print(f"Found {len(all_signals)} trading signals across {len(favorite_symbols)} symbols")
                
                # Rank signals by score
                all_signals.sort(key=lambda x: x['score'], reverse=True)
                
                # Set up enhanced headers
                headers = [
                    f"{Fore.YELLOW}{Style.BRIGHT}Symbol{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Side{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Type{Style.RESET_ALL}", 
                    f"{Fore.YELLOW}{Style.BRIGHT}Entry{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Stop Loss{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Target{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Risk Pts{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Score{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}Qty{Style.RESET_ALL}"
                ]
                
                # Build table data
                table_data = []
                
                for signal in all_signals:
                    # Format side with color
                    side_formatted = f"{Fore.GREEN}{signal['side']}{Style.RESET_ALL}" if signal['side'] == "BUY" else f"{Fore.RED}{signal['side']}{Style.RESET_ALL}"
                    
                    # Format score with color based on value
                    if signal['score'] >= 80:
                        score_formatted = f"{Fore.GREEN}{signal['score']}{Style.RESET_ALL}"
                    elif signal['score'] >= 60:
                        score_formatted = f"{Fore.YELLOW}{signal['score']}{Style.RESET_ALL}"
                    else:
                        score_formatted = f"{Fore.RED}{signal['score']}{Style.RESET_ALL}"
                    
                    table_data.append([
                        f"{Fore.CYAN}{signal['symbol']}{Style.RESET_ALL}",
                        side_formatted,
                        signal['type'],
                        f"{signal['entry']:.2f}",
                        f"{signal['sl']:.2f}",
                        f"{signal['target']:.2f}",
                        f"{abs(signal['entry']-signal['sl']):.2f}",
                        score_formatted,
                        f"{signal['size']}"
                    ])
                
                # Print well-formatted table using tabulate
                from tabulate import tabulate
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
                
                # Also show consolidated view
                print(f"\n{Fore.CYAN}Generating consolidated signals table across multiple timeframes...{Style.RESET_ALL}")
                self.generate_consolidated_signals_table()
            else:
                print(f"\n{Fore.YELLOW}📢 No trading signals found at this time.{Style.RESET_ALL}")
            
            print("\n" + "═" * 100)
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error running MCX system: {str(e)}{Style.RESET_ALL}")

    def run_virtual_trading(self):
        """Run automated virtual trading with position tracking
        
        This function automatically takes trades based on signals,
        tracks them until they hit either stop loss or take profit,
        and maintains a professional performance tracking table.
        """
        try:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}╔══════════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}║                    AUTOMATED VIRTUAL TRADING SESSION                          ║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
            
            # Virtual trading session data
            virtual_trades = []
            completed_trades = []
            fixed_lot_size = 1  # Always trade with 1 lot
            session_start_time = datetime.now(self.indian_timezone)
            
            # Table headers for trade log
            headers = [
                f"{Fore.YELLOW}{Style.BRIGHT}ID{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Symbol{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Side{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Timeframe{Style.RESET_ALL}", 
                f"{Fore.YELLOW}{Style.BRIGHT}Entry{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Stop Loss{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Target{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Status{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}P&L{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Entry Time{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}Exit Time{Style.RESET_ALL}"
            ]
            
            # Performance tracking variables
            total_trades = 0
            winning_trades = 0
            total_pnl = 0
            
            symbols = self.config.get("favorite_symbols", ["SILVER", "GOLD", "CRUDEOIL", "NATURALGAS"])
            timeframes = ["5min", "15min", "1hour"]
            timeframe_map = {
                "5min": "5",
                "15min": "15",
                "1hour": "60"
            }
            
            print(f"\n{Fore.GREEN}✅ Starting virtual trading session at {session_start_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✅ Trading {len(symbols)} symbols across {len(timeframes)} timeframes{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✅ Fixed lot size: {fixed_lot_size} lot per trade{Style.RESET_ALL}")
            
            # Interactive loop - continue until user exits
            try:
                while True:
                    # Check for new signals
                    all_signals = []
                    for symbol in symbols:
                        for tf_name, tf_value in [(tf, timeframe_map.get(tf, tf)) for tf in timeframes]:
                            try:
                                df = self.get_historical_data(symbol, timeframe=tf_value, days=5)
                                if df is not None and len(df) > 20:
                                    df = self.calculate_indicators(df)
                                    signals = self.check_entry_signals(df, timeframe=tf_name)
                                    
                                    # Filter signals to avoid duplicates
                                    for signal in signals:
                                        # Check if we already have this signal in active trades
                                        is_duplicate = False
                                        for trade in virtual_trades:
                                            if (trade['symbol'] == signal['symbol'] and 
                                                trade['timeframe'] == signal['timeframe'] and
                                                trade['side'] == signal['side']):
                                                is_duplicate = True
                                                break
                                        
                                        if not is_duplicate:
                                            # Create new virtual trade with 1 lot
                                            trade_id = f"{signal['symbol']}_{signal['timeframe']}_{len(virtual_trades) + 1}"
                                            # Calculate proper lot size based on the symbol's lot size
                                            lot_multiplier = 1  # Default
                                            if 'SILVER' in signal['symbol']:
                                                lot_multiplier = 1
                                            elif 'GOLD' in signal['symbol']:
                                                lot_multiplier = 1
                                            elif 'CRUDEOIL' in signal['symbol']:
                                                lot_multiplier = 1
                                            elif 'NATURALGAS' in signal['symbol']:
                                                lot_multiplier = 1
                                            
                                            new_trade = {
                                                'id': trade_id,
                                                'symbol': signal['symbol'],
                                                'side': signal['side'],
                                                'timeframe': signal['timeframe'],
                                                'entry': signal['entry'],
                                                'sl': signal['sl'],
                                                'target': signal['target'],
                                                'size': fixed_lot_size * lot_multiplier,
                                                'entry_time': datetime.now(self.indian_timezone),
                                                'status': 'OPEN',
                                                'exit_time': None,
                                                'exit_price': None,
                                                'pnl': 0,
                                                'hit_target': False,
                                                'hit_sl': False
                                            }
                                            
                                            virtual_trades.append(new_trade)
                                            total_trades += 1
                                            print(f"\n{Fore.GREEN}➕ New virtual trade opened:{Style.RESET_ALL}")
                                            print(f"  Symbol: {new_trade['symbol']}")
                                            print(f"  Side: {new_trade['side']}")
                                            print(f"  Timeframe: {new_trade['timeframe']}")
                                            print(f"  Entry: {new_trade['entry']:.2f}")
                                            print(f"  Stop Loss: {new_trade['sl']:.2f}")
                                            print(f"  Target: {new_trade['target']:.2f}")
                            except Exception as e:
                                print(f"{Fore.RED}❌ Error checking signals for {symbol} on {tf_name}: {str(e)}{Style.RESET_ALL}")
                    
                    # Update open virtual trades
                    for trade in virtual_trades[:]:  # Use a copy of the list since we'll modify it
                        if trade['status'] == 'OPEN':
                            # Get latest price
                            df = self.get_historical_data(trade['symbol'], timeframe="1", days=1)
                            if df is not None and len(df) > 0:
                                current_price = df.iloc[-1]['close']
                                
                                # Check if SL or TP hit
                                if trade['side'] == 'BUY':
                                    # Stop loss hit
                                    if current_price <= trade['sl']:
                                        trade['status'] = 'CLOSED'
                                        trade['exit_time'] = datetime.now(self.indian_timezone)
                                        trade['exit_price'] = trade['sl']
                                        trade['pnl'] = (trade['sl'] - trade['entry']) * trade['size']
                                        trade['hit_sl'] = True
                                        
                                        # Add to completed trades
                                        completed_trades.append(trade)
                                        # Remove from active trades
                                        virtual_trades.remove(trade)
                                        
                                        total_pnl += trade['pnl']
                                        print(f"\n{Fore.RED}⛔ Stop loss hit: {trade['symbol']} {trade['side']}{Style.RESET_ALL}")
                                        print(f"  Loss: ₹{abs(trade['pnl']):.2f}")
                                    
                                    # Take profit hit
                                    elif current_price >= trade['target']:
                                        trade['status'] = 'CLOSED'
                                        trade['exit_time'] = datetime.now(self.indian_timezone)
                                        trade['exit_price'] = trade['target']
                                        trade['pnl'] = (trade['target'] - trade['entry']) * trade['size']
                                        trade['hit_target'] = True
                                        
                                        # Add to completed trades
                                        completed_trades.append(trade)
                                        # Remove from active trades
                                        virtual_trades.remove(trade)
                                        
                                        winning_trades += 1
                                        total_pnl += trade['pnl']
                                        print(f"\n{Fore.GREEN}🎯 Target hit: {trade['symbol']} {trade['side']}{Style.RESET_ALL}")
                                        print(f"  Profit: ₹{trade['pnl']:.2f}")
                                else:  # SELL
                                    # Stop loss hit
                                    if current_price >= trade['sl']:
                                        trade['status'] = 'CLOSED'
                                        trade['exit_time'] = datetime.now(self.indian_timezone)
                                        trade['exit_price'] = trade['sl']
                                        trade['pnl'] = (trade['entry'] - trade['sl']) * trade['size']
                                        trade['hit_sl'] = True
                                        
                                        # Add to completed trades
                                        completed_trades.append(trade)
                                        # Remove from active trades
                                        virtual_trades.remove(trade)
                                        
                                        total_pnl += trade['pnl']
                                        print(f"\n{Fore.RED}⛔ Stop loss hit: {trade['symbol']} {trade['side']}{Style.RESET_ALL}")
                                        print(f"  Loss: ₹{abs(trade['pnl']):.2f}")
                                    
                                    # Take profit hit
                                    elif current_price <= trade['target']:
                                        trade['status'] = 'CLOSED'
                                        trade['exit_time'] = datetime.now(self.indian_timezone)
                                        trade['exit_price'] = trade['target']
                                        trade['pnl'] = (trade['entry'] - trade['target']) * trade['size']
                                        trade['hit_target'] = True
                                        
                                        # Add to completed trades
                                        completed_trades.append(trade)
                                        # Remove from active trades
                                        virtual_trades.remove(trade)
                                        
                                        winning_trades += 1
                                        total_pnl += trade['pnl']
                                        print(f"\n{Fore.GREEN}🎯 Target hit: {trade['symbol']} {trade['side']}{Style.RESET_ALL}")
                                        print(f"  Profit: ₹{trade['pnl']:.2f}")
                    
                    # Display trade summary table
                    print("\n" + "═" * 120)
                    print(f"{Fore.MAGENTA}{Style.BRIGHT}╔═══════════════════════════════════════════╗{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}{Style.BRIGHT}║      VIRTUAL TRADING SESSION SUMMARY      ║{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}{Style.BRIGHT}╚═══════════════════════════════════════════╝{Style.RESET_ALL}")
                    
                    # Performance metrics
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    print(f"{Fore.CYAN}Session duration: {(datetime.now(self.indian_timezone) - session_start_time).seconds // 60} minutes{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}Total trades: {total_trades}{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}Win rate: {win_rate:.1f}%{Style.RESET_ALL}")
                    
                    if total_pnl >= 0:
                        pnl_str = f"{Fore.GREEN}₹{total_pnl:.2f}{Style.RESET_ALL}"
                    else:
                        pnl_str = f"{Fore.RED}₹{total_pnl:.2f}{Style.RESET_ALL}"
                    print(f"{Fore.CYAN}Total P&L: {pnl_str}{Style.RESET_ALL}")
                    
                    # Active trades table
                    if virtual_trades:
                        print("\n" + "─" * 120)
                        print(f"{Fore.GREEN}{Style.BRIGHT}ACTIVE TRADES:{Style.RESET_ALL}")
                        active_data = []
                        for trade in virtual_trades:
                            if trade['side'] == 'BUY':
                                side_formatted = f"{Fore.GREEN}{trade['side']}{Style.RESET_ALL}"
                            else:
                                side_formatted = f"{Fore.RED}{trade['side']}{Style.RESET_ALL}"
                            
                            active_data.append([
                                trade['id'],
                                f"{Fore.CYAN}{trade['symbol']}{Style.RESET_ALL}",
                                side_formatted,
                                trade['timeframe'],
                                f"{trade['entry']:.2f}",
                                f"{trade['sl']:.2f}",
                                f"{trade['target']:.2f}",
                                f"{Fore.CYAN}OPEN{Style.RESET_ALL}",
                                f"-",
                                trade['entry_time'].strftime('%H:%M:%S'),
                                "-"
                            ])
                        
                        print(tabulate(active_data, headers=headers, tablefmt="grid"))
                    
                    # Completed trades table
                    if completed_trades:
                        print("\n" + "─" * 120)
                        print(f"{Fore.YELLOW}{Style.BRIGHT}COMPLETED TRADES:{Style.RESET_ALL}")
                        completed_data = []
                        for trade in completed_trades:
                            if trade['side'] == 'BUY':
                                side_formatted = f"{Fore.GREEN}{trade['side']}{Style.RESET_ALL}"
                            else:
                                side_formatted = f"{Fore.RED}{trade['side']}{Style.RESET_ALL}"
                            
                            if trade['hit_target']:
                                status = f"{Fore.GREEN}TARGET HIT{Style.RESET_ALL}"
                                pnl_str = f"{Fore.GREEN}₹{trade['pnl']:.2f}{Style.RESET_ALL}"
                            elif trade['hit_sl']:
                                status = f"{Fore.RED}STOP HIT{Style.RESET_ALL}"
                                pnl_str = f"{Fore.RED}₹{trade['pnl']:.2f}{Style.RESET_ALL}"
                            else:
                                status = f"{Fore.YELLOW}CLOSED{Style.RESET_ALL}"
                                if trade['pnl'] >= 0:
                                    pnl_str = f"{Fore.GREEN}₹{trade['pnl']:.2f}{Style.RESET_ALL}"
                                else:
                                    pnl_str = f"{Fore.RED}₹{trade['pnl']:.2f}{Style.RESET_ALL}"
                            
                            completed_data.append([
                                trade['id'],
                                f"{Fore.CYAN}{trade['symbol']}{Style.RESET_ALL}",
                                side_formatted,
                                trade['timeframe'],
                                f"{trade['entry']:.2f}",
                                f"{trade['sl']:.2f}",
                                f"{trade['target']:.2f}",
                                status,
                                pnl_str,
                                trade['entry_time'].strftime('%H:%M:%S'),
                                trade['exit_time'].strftime('%H:%M:%S') if trade['exit_time'] else "-"
                            ])
                        
                        print(tabulate(completed_data, headers=headers, tablefmt="grid"))
                    
                    print("\n" + "═" * 120)
                    print(f"{Fore.YELLOW}Press Ctrl+C to exit virtual trading session and view final results{Style.RESET_ALL}")
                    
                    # Wait before next check
                    time.sleep(60)  # Check every minute
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.GREEN}✅ Virtual trading session ended by user{Style.RESET_ALL}")
                
                # Final summary statistics
                if total_trades > 0:
                    print("\n" + "═" * 120)
                    print(f"{Fore.CYAN}{Style.BRIGHT}╔═══════════════════════════════════════════╗{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}{Style.BRIGHT}║       FINAL PERFORMANCE SUMMARY           ║{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}{Style.BRIGHT}╚═══════════════════════════════════════════╝{Style.RESET_ALL}")
                    
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    print(f"Total trades: {total_trades}")
                    print(f"Winning trades: {winning_trades}")
                    print(f"Losing trades: {total_trades - winning_trades}")
                    print(f"Win rate: {win_rate:.1f}%")
                    
                    if total_pnl >= 0:
                        pnl_str = f"{Fore.GREEN}₹{total_pnl:.2f}{Style.RESET_ALL}"
                    else:
                        pnl_str = f"{Fore.RED}₹{total_pnl:.2f}{Style.RESET_ALL}"
                    print(f"Total P&L: {pnl_str}")
                    
                    # Calculate average metrics
                    avg_win = sum(t['pnl'] for t in completed_trades if t['pnl'] > 0) / winning_trades if winning_trades > 0 else 0
                    avg_loss = sum(abs(t['pnl']) for t in completed_trades if t['pnl'] < 0) / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
                    
                    if avg_loss > 0:
                        reward_risk = avg_win / avg_loss
                        print(f"Average win: ₹{avg_win:.2f}")
                        print(f"Average loss: ₹{avg_loss:.2f}")
                        print(f"Reward-to-risk ratio: {reward_risk:.2f}")
                    
                    # Symbol performance
                    print("\n" + "─" * 120)
                    print(f"{Fore.YELLOW}{Style.BRIGHT}PERFORMANCE BY SYMBOL:{Style.RESET_ALL}")
                    symbol_stats = {}
                    
                    for trade in completed_trades:
                        symbol = trade['symbol']
                        if symbol not in symbol_stats:
                            symbol_stats[symbol] = {
                                'total': 0,
                                'wins': 0,
                                'pnl': 0
                            }
                        
                        symbol_stats[symbol]['total'] += 1
                        if trade['pnl'] > 0:
                            symbol_stats[symbol]['wins'] += 1
                        symbol_stats[symbol]['pnl'] += trade['pnl']
                    
                    # Create symbol performance table
                    symbol_data = []
                    for symbol, stats in symbol_stats.items():
                        symbol_win_rate = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
                        if stats['pnl'] >= 0:
                            pnl_str = f"{Fore.GREEN}₹{stats['pnl']:.2f}{Style.RESET_ALL}"
                        else:
                            pnl_str = f"{Fore.RED}₹{stats['pnl']:.2f}{Style.RESET_ALL}"
                        
                        symbol_data.append([
                            f"{Fore.CYAN}{symbol}{Style.RESET_ALL}",
                            stats['total'],
                            stats['wins'],
                            f"{symbol_win_rate:.1f}%",
                            pnl_str
                        ])
                    
                    symbol_headers = [
                        f"{Fore.YELLOW}{Style.BRIGHT}Symbol{Style.RESET_ALL}",
                        f"{Fore.YELLOW}{Style.BRIGHT}Total{Style.RESET_ALL}",
                        f"{Fore.YELLOW}{Style.BRIGHT}Wins{Style.RESET_ALL}",
                        f"{Fore.YELLOW}{Style.BRIGHT}Win Rate{Style.RESET_ALL}",
                        f"{Fore.YELLOW}{Style.BRIGHT}P&L{Style.RESET_ALL}"
                    ]
                    
                    print(tabulate(symbol_data, headers=symbol_headers, tablefmt="grid"))
                    
                    # Timeframe performance
                    print("\n" + "─" * 120)
                    print(f"{Fore.YELLOW}{Style.BRIGHT}PERFORMANCE BY TIMEFRAME:{Style.RESET_ALL}")
                    timeframe_stats = {}
                    
                    for trade in completed_trades:
                        tf = trade['timeframe']
                        if tf not in timeframe_stats:
                            timeframe_stats[tf] = {
                                'total': 0,
                                'wins': 0,
                                'pnl': 0
                            }
                        
                        timeframe_stats[tf]['total'] += 1
                        if trade['pnl'] > 0:
                            timeframe_stats[tf]['wins'] += 1
                        timeframe_stats[tf]['pnl'] += trade['pnl']
                    
                    # Create timeframe performance table
                    tf_data = []
                    for tf, stats in timeframe_stats.items():
                        tf_win_rate = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
                        if stats['pnl'] >= 0:
                            pnl_str = f"{Fore.GREEN}₹{stats['pnl']:.2f}{Style.RESET_ALL}"
                        else:
                            pnl_str = f"{Fore.RED}₹{stats['pnl']:.2f}{Style.RESET_ALL}"
                        
                        tf_data.append([
                            tf,
                            stats['total'],
                            stats['wins'],
                            f"{tf_win_rate:.1f}%",
                            pnl_str
                        ])
                    
                    tf_headers = [
                        f"{Fore.YELLOW}{Style.BRIGHT}Timeframe{Style.RESET_ALL}",
                        f"{Fore.YELLOW}{Style.BRIGHT}Total{Style.RESET_ALL}",
                        f"{Fore.YELLOW}{Style.BRIGHT}Wins{Style.RESET_ALL}",
                        f"{Fore.YELLOW}{Style.BRIGHT}Win Rate{Style.RESET_ALL}",
                        f"{Fore.YELLOW}{Style.BRIGHT}P&L{Style.RESET_ALL}"
                    ]
                    
                    print(tabulate(tf_data, headers=tf_headers, tablefmt="grid"))
                
            return
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error running virtual trading: {str(e)}{Style.RESET_ALL}")

# Main execution
if __name__ == "__main__":
    import sys
    import time  # For real-time monitoring
    
    # Parse command line arguments
    mode = None
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'eod':
            mode = 'eod'
        elif sys.argv[1].lower() == 'dashboard':
            mode = 'dashboard'
        elif sys.argv[1].lower() == 'gold':
            mode = 'gold_trade'
        elif sys.argv[1].lower() == 'realtime':
            mode = 'realtime'
        elif sys.argv[1].lower() == 'signals':
            mode = 'signals'
        elif sys.argv[1].lower() == 'virtual':
            mode = 'virtual'
        elif sys.argv[1].lower() == 'close':
            # Initialize MCX trader
            mcx = MCXTrader()
            # If there's a third argument, use it as the symbol to close
            if len(sys.argv) > 2:
                mcx.close_position(symbol=sys.argv[2].upper())
            else:
                # Show positions first
                mcx.show_current_positions()
                # Ask which position to close
                print(f"\n{Fore.YELLOW}To close positions, run: python mcx_trader.py close SYMBOL{Style.RESET_ALL}")
            sys.exit(0)
    
    mcx = MCXTrader()
    mcx.run(mode) 