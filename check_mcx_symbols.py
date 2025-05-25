from fyers_apiv3 import fyersModel
import json

# Connect to Fyers
with open("access_token.txt", "r") as f:
    token = f.read().strip()
    
fyers = fyersModel.FyersModel(
    client_id="JAOZFJL8IO-100",
    is_async=False,
    token=token,
    log_path=""
)

# Test if these symbols are valid with the quotes API
test_symbols = [
    "MCX:SILVERMIC25JUNFUT",
    "MCX:GOLDM25JUNFUT",
    "MCX:CRUDEOILM25JUNFUT",  # Changed from CRUDEOILM23JUNFUT
    "MCX:NATURALGAS25JUNFUT",  # Changed from NATURALGAS27JUNFUT
    "MCX:COPPER25JUNFUT",      # Changed from COPPER30JUNFUT
    "MCX:ZINC25JUNFUT",        # Changed from ZINC30JUNFUT
    "MCX:LEAD25JUNFUT",        # Changed from LEAD30JUNFUT
    "MCX:ALUMINIUM25JUNFUT"    # Changed from ALUMINI30JUNFUT
]

# Try to get quotes for each symbol
for symbol in test_symbols:
    print(f"Testing {symbol}...")
    response = fyers.quotes({"symbols": symbol})
    print(f"Response: {response}")
    print("-" * 50)

# Update the MCX symbols file with corrected symbols
corrected_symbols = {
    "SILVER": "MCX:SILVERMIC25JUNFUT",
    "GOLD": "MCX:GOLDM25JUNFUT",
    "CRUDEOIL": "MCX:CRUDEOILM25JUNFUT", 
    "NATURALGAS": "MCX:NATURALGAS25JUNFUT",
    "COPPER": "MCX:COPPER25JUNFUT",
    "ZINC": "MCX:ZINC25JUNFUT",
    "LEAD": "MCX:LEAD25JUNFUT",
    "ALUMINIUM": "MCX:ALUMINIUM25JUNFUT"
}

# Save corrected symbols to file
with open("mcx_symbols.json", "w") as f:
    json.dump(corrected_symbols, f, indent=4)
    
print("Updated mcx_symbols.json with corrected symbols") 