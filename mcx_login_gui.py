#!/usr/bin/env python
"""
Fyers Login GUI
Run this file to start the login process
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from fyers_apiv3 import fyersModel
import requests
import hashlib
import os
import webbrowser
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
from colorama import init, Fore, Style
import json
import sys

init()

class DirectFyersAuth:
    """Direct authentication with Fyers API"""
    
    def __init__(self, config):
        self.config = config
        self.client_id = config["credentials"].get("client_id", "")
        self.app_id = config["credentials"].get("app_id", "")
        self.redirect_uri = config["credentials"].get("redirect_uri", "https://www.google.com/")
        self.secret_key = config["credentials"].get("secret_key", "")
        
        # Create logger if available
        try:
            from logger import Logger
            self.logger = Logger("DirectFyersAuth")
        except:
            # Basic logger fallback
            self.logger = type('', (), {})()
            self.logger.log_info = lambda msg: print(f"INFO: {msg}")
            self.logger.log_error = lambda msg: print(f"ERROR: {msg}")
    
    def get_access_token(self):
        """Get Fyers access token using App ID approach"""
        try:
            self.logger.log_info("Starting direct Fyers authentication")
            
            # Step 1: Get auth code
            session = fyersModel.SessionModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type="code",
                grant_type="authorization_code"
            )
            
            # Generate auth code URL
            auth_url = session.generate_authcode()
            self.logger.log_info(f"Auth URL generated: {auth_url}")
            
            # Open browser for user to log in
            webbrowser.open(auth_url)
            
            # Create dialog to input the auth code
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            # Show information dialog
            messagebox.showinfo(
                "Fyers Authentication", 
                "1. Login with your Fyers credentials in the browser\n"
                "2. Complete verification (PIN/OTP)\n"
                "3. You will be redirected to Google\n"
                "4. Copy the full URL from your browser address bar\n"
                "5. Paste it in the next dialog"
            )
            
            # Get the full redirect URL from user
            auth_url = simpledialog.askstring(
                "Fyers Authentication", 
                "Paste the FULL URL from your browser:",
                parent=root
            )
            
            if not auth_url:
                self.logger.log_error("User cancelled authentication")
                return None
                
            # Extract auth code from URL
            try:
                auth_code = auth_url.split("auth_code=")[1].split("&")[0]
                self.logger.log_info(f"Extracted auth code: {auth_code}")
            except:
                self.logger.log_error("Could not extract auth code from URL")
                messagebox.showerror(
                    "Authentication Error", 
                    "Could not extract auth code from the URL.\n"
                    "Please make sure you copied the complete URL."
                )
                return None
            
            # Step 2: Generate access token
            try:
                # Generate token
                session.set_token(auth_code)
                response = session.generate_token()
                
                if response.get("access_token"):
                    # Save token to file for future use
                    token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'access_token.txt')
                    with open(token_file, 'w') as f:
                        f.write(response["access_token"])
                    
                    # Also save json format (optional)
                    json_token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'access_token.json')
                    with open(json_token_file, 'w') as f:
                        json.dump(response, f)
                    
                    self.logger.log_info(f"Token generated successfully: {response}")
                    return response["access_token"]
                else:
                    self.logger.log_error(f"Failed to generate token: {response}")
                    error_message = response.get("message", "Unknown error")
                    messagebox.showerror("Token Error", f"Failed to generate token: {error_message}")
                    return None
                    
            except Exception as e:
                self.logger.log_error(f"Error generating token: {str(e)}")
                messagebox.showerror("Token Error", f"Failed to generate token: {str(e)}")
                return None
                
        except Exception as e:
            self.logger.log_error(f"Authentication error: {str(e)}")
            return None

# For backward compatibility, keep the original class name
class FyersLoginGUI(DirectFyersAuth):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fyers Login")
        self.root.geometry("600x700")
        self.root.configure(bg='#f0f0f0')
        
        # Set token path in mcx_master directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.token_file = os.path.join(script_dir, "access_token.txt")
        
        # Credentials
        self.client_id = "JAOZFJL8IO-100"
        self.secret_key = "WAKAQ5SLYW"
        self.redirect_uri = "https://trade.fyers.in/api-login/redirect-uri/index.html"
        
        self.create_widgets()
        
    def create_widgets(self):
        # Style
        style = ttk.Style()
        style.configure('TButton', padding=10, font=('Helvetica', 10))
        style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Fyers Trading Login", font=('Helvetica', 16, 'bold'))
        title.pack(pady=20)
        
        # Status Text
        self.status_text = tk.Text(main_frame, height=20, width=60, font=('Courier', 10))
        self.status_text.pack(pady=20)
        self.status_text.config(state='disabled')
        
        # URL Entry
        self.url_var = tk.StringVar()
        url_frame = ttk.Frame(main_frame)
        url_frame.pack(fill=tk.X, pady=10)
        
        url_label = ttk.Label(url_frame, text="Paste URL here:")
        url_label.pack()
        
        self.url_entry = ttk.Entry(url_frame, textvariable=self.url_var, width=50)
        self.url_entry.pack(pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        self.login_btn = ttk.Button(btn_frame, text="1. Start Login", command=self.start_login)
        self.login_btn.pack(side=tk.LEFT, padx=5)
        
        self.submit_btn = ttk.Button(btn_frame, text="2. Submit URL", command=self.submit_url, state='disabled')
        self.submit_btn.pack(side=tk.LEFT, padx=5)
        
    def log(self, message):
        """Update GUI text only"""
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, f"\n{message}")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')
        self.root.update()

    def start_login(self):
        try:
            self.log("\n=== Starting Login Process ===")
            
            # Initialize session with all required parameters
            session = fyersModel.SessionModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type="code",
                grant_type="authorization_code",
                state="sample_state"
            )

            # Generate auth URL
            auth_url = session.generate_authcode()
            self.log("\nOpening login URL in browser...")
            webbrowser.open(auth_url)
            
            self.log("\nPlease:")
            self.log("1. Login in the browser")
            self.log("2. Copy the URL after login")
            self.log("3. Paste it below and click Submit URL")
            
            self.submit_btn.config(state='normal')
            self.login_btn.config(state='disabled')
            
        except Exception as e:
            self.log(f"\n❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
            
    def submit_url(self):
        try:
            self.log("\nProcessing login URL...")
            url = self.url_var.get()
            
            if not url:
                raise ValueError("Please paste the login URL first")
            
            # Extract auth code
            auth_code = url.split("auth_code=")[1].split("&")[0]
            self.log(f"\nAuth code extracted: {auth_code[:10]}...")
            
            # Generate hash
            app_id_hash = hashlib.sha256((self.client_id + ":" + self.secret_key).encode()).hexdigest()
            
            # Get access token
            self.log("\nGetting access token...")
            session = fyersModel.SessionModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type="code",
                grant_type="authorization_code",
                state="sample_state"
            )
            session.set_token(auth_code)
            response = session.generate_token()
            
            if "access_token" in response:
                access_token = response["access_token"]
                self.log("\nGot access token!")
                
                # Save token
                with open(self.token_file, "w") as f:
                    f.write(access_token)
                self.log(f"\n✅ Token saved at: {self.token_file}")
                
                # Test API
                self.log("\nTesting API connection...")
                
                # Get script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Create Fyers model with logging disabled
                fyers = fyersModel.FyersModel(
                    client_id=self.client_id,
                    token=access_token,
                    log_path=os.path.join(script_dir, "logs")
                )

                # Create logs directory if it doesn't exist
                os.makedirs(os.path.join(script_dir, "logs"), exist_ok=True)
                
                # Set the token for validation
                validation = f"{self.client_id}:{access_token}"
                fyers.token = validation

                profile = fyers.get_profile()
                if profile["s"] == "ok":
                    data = profile["data"]
                    self.log("\nProfile data:")
                    self.log("-" * 30)
                    self.log(f"Name: {data['name']}")
                    self.log(f"Email: {data['email_id']}")
                    self.log(f"Mobile: {data['mobile_number']}")
                    self.log(f"PAN: {data['PAN']}")
                    self.log(f"Fyers ID: {data['fy_id']}")
                    self.log("\n✅ Login Successful!")
                    messagebox.showinfo("Success", "Login Successful!")
                else:
                    self.log("\n❌ API Error: " + profile["message"])
            else:
                self.log("\n❌ Failed to get access token!")
                self.log("Error: " + str(response))
                
        except Exception as e:
            self.log(f"\n❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
            
    def run(self):
        self.root.mainloop()
        

if __name__ == "__main__":
    app = FyersLoginGUI()
    app.run() 