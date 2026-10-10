# -*- coding: utf-8 -*-
"""Hardware Authentication Module for Onmyoji Tactical Helper.

This module retrieves unique hardware UUID features (WMIC/system UUID) to generate
a hashed Machine ID, verifies authorization against a remote Google Sheets license database,
and renders a Windows 11 Fluent-style activation UI for unauthorized machines.
"""

import os
import hashlib
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox
import urllib.request
from typing import NoReturn

# Google Sheets License Authorization List CSV URL
AUTH_CSV_URL = os.getenv(
    "AUTH_CSV_URL", 
    "https://docs.google.com/spreadsheets/d/e/YOUR_AUTH_SHEET_ID/pub?output=csv"
)

DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() in ("true", "1")

def get_machine_id() -> str:
    """Retrieves hardware features from system UUID to generate a unique Machine ID.

    Returns:
        16-character uppercase SHA-256 hashed unique hardware identifier string.
    """
    try:
        uuid = subprocess.check_output('wmic csproduct get uuid').decode().split('\n')[1].strip()
    except Exception:
        import uuid as uuid_lib
        uuid = str(uuid_lib.getnode())
    raw_id = f"ONMYOJI_{uuid}"
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16].upper()


def check_activation() -> None:
    """Verifies machine authorization against remote license database.

    Intercepts unauthorized hardware and displays activation GUI if license check fails.
    """
    if DEMO_MODE:
        print("🔓 [Auth] 運行於 Demo 模式，已跳過硬件鎖驗證。")
        return
    machine_id = get_machine_id()
    auth_hash = hashlib.sha256(machine_id.encode()).hexdigest()
    timestamp = int(time.time())
    no_cache_url = (
        f"{AUTH_CSV_URL}&t={timestamp}"
        if "?" in AUTH_CSV_URL
        else f"{AUTH_CSV_URL}?t={timestamp}"
    )

    try:
        req = urllib.request.Request(
            no_cache_url,
            headers={
                'User-Agent': 'Mozilla/5.0',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            csv_data = response.read().decode('utf-8')
            if auth_hash in csv_data:
                return  # Verification passed, grant access
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning("網絡錯誤", f"無法連線至驗證伺服器，請檢查網絡連線。\n(錯誤代碼: {e})")
        sys.exit()

    # License verification failed: Display Windows 11 style activation UI
    show_unauthorized_and_exit(machine_id)


def show_unauthorized_and_exit(machine_id: str) -> NoReturn:
    """Displays Windows 11 Fluent-style activation UI and terminates application upon exit.

    Args:
        machine_id: 16-character hardware identifier string to display.
    """
    root = tk.Tk()
    root.title("產品啟動")

    window_width = 520
    window_height = 290
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    pos_x = int((screen_width - window_width) / 2)
    pos_y = int((screen_height - window_height) / 2)

    root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    # Windows 11 Dark Mode Palette
    bg_color = "#202020"
    card_bg = "#2D2D2D"
    text_primary = "#FFFFFF"
    text_secondary = "#CCCCCC"
    accent_blue = "#4CC2FF"
    btn_gray = "#333333"

    root.configure(bg=bg_color)

    # Header section
    tk.Label(
        root,
        text="需要啟動系統",
        font=("Microsoft JhengHei", 18, "bold"),
        fg=text_primary,
        bg=bg_color
    ).pack(pady=(25, 5), anchor="w", padx=35)

    tk.Label(
        root,
        text="此設備尚未綁定數位授權。請提供您的專屬識別碼以啟用功能。",
        font=("Microsoft JhengHei", 10),
        fg=text_secondary,
        bg=bg_color,
        wraplength=450,
        justify="left"
    ).pack(anchor="w", padx=35, pady=(0, 15))

    # Machine ID Card UI
    card = tk.Frame(root, bg=card_bg, bd=0)
    card.pack(fill=tk.X, padx=35, pady=5, ipadx=10, ipady=12)

    tk.Label(
        card,
        text="裝置識別碼 (Machine ID)",
        font=("Microsoft JhengHei", 9),
        fg=text_secondary,
        bg=card_bg
    ).pack(anchor="w", padx=10, pady=(0, 2))

    entry = tk.Entry(
        card,
        justify='left',
        font=("Consolas", 13, "bold"),
        bg=card_bg,
        fg=accent_blue,
        readonlybackground=card_bg,
        insertbackground=text_primary,
        bd=0,
        highlightthickness=0
    )
    entry.insert(0, machine_id)
    entry.config(state='readonly')
    entry.pack(anchor="w", padx=10, fill=tk.X)

    # Contact Info
    tk.Label(
        root,
        text="獲取授權請聯絡 Discord: alexc7832",
        font=("Microsoft JhengHei", 9),
        fg=text_secondary,
        bg=bg_color
    ).pack(anchor="w", padx=35, pady=(10, 0))

    # Bottom Buttons Container
    bottom_frame = tk.Frame(root, bg=bg_color)
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20, padx=35)

    def copy_to_clipboard() -> None:
        root.clipboard_clear()
        root.clipboard_append(machine_id)
        root.update()
        copy_btn.config(text="已複製 ✓", bg="#3B9ECA")
        root.after(2000, lambda: copy_btn.config(text="複製識別碼", bg=accent_blue))

    copy_btn = tk.Button(
        bottom_frame,
        text="複製識別碼",
        command=copy_to_clipboard,
        font=("Microsoft JhengHei", 10, "bold"),
        bg=accent_blue,
        fg="#000000",
        activebackground="#3B9ECA",
        relief=tk.FLAT,
        width=12,
        cursor="hand2"
    )
    copy_btn.pack(side=tk.RIGHT, padx=(10, 0))

    close_btn = tk.Button(
        bottom_frame,
        text="關閉",
        command=sys.exit,
        font=("Microsoft JhengHei", 10),
        bg=btn_gray,
        fg=text_primary,
        activebackground="#444444",
        activeforeground=text_primary,
        relief=tk.FLAT,
        width=8,
        cursor="hand2"
    )
    close_btn.pack(side=tk.RIGHT)

    root.mainloop()
    sys.exit()