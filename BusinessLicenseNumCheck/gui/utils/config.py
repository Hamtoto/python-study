import os
import json
import sys

def get_user_data_folder():
    """
    Windows: %APPDATA%\YourAppName
    macOS: ~/Library/Application Support/YourAppName
    Linux (추가할 경우): ~/.config/YourAppName
    """
    app_name = "Ethan Toy Tool"  # 앱이름
    
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        return os.path.join(base, app_name)
    elif sys.platform.startswith("darwin"):
        base = os.path.expanduser("~/Library/Application Support")
        return os.path.join(base, app_name)
    else:  # Linux or 기타
        base = os.path.expanduser("~/.config")
        return os.path.join(base, app_name)

SETTINGS_FILE = os.path.join(get_user_data_folder(), 'settings.json')

def ensure_settings_directory():
    settings_dir = os.path.dirname(SETTINGS_FILE)
    if not os.path.exists(settings_dir):
        os.makedirs(settings_dir, exist_ok=True)

def load_settings():
    ensure_settings_directory()
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_settings(settings):
    try:
        ensure_settings_directory()
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving settings: {e}")
