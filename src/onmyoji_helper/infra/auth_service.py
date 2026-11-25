"""Hardware Authentication Service for Onmyoji Tactical Helper.

Supports dual-track execution:
1. Private Production Mode (when `ONMYOJI_AUTH_CSV_URL` is configured in `.env`): validates
   the SHA-256 Machine ID against the remote Google Sheets license whitelist (preserving full
   backwards compatibility with `wmic csproduct get uuid`, plus `MachineGuid` fallback).
2. Public Sandbox / Demo Mode (when `.env` is absent or `--demo` is passed): bypasses private
   license enforcement cleanly so recruiters and contributors can evaluate the app immediately.
"""

import csv
import hashlib
import io
import logging
import subprocess
import sys
import time
import tkinter as tk
import urllib.request
from tkinter import messagebox
from typing import NoReturn

from onmyoji_helper.infra.env_loader import get_clean_env

logger = logging.getLogger("OnmyojiHelper")

CREATE_NO_WINDOW: int = 0x08000000


def get_auth_csv_url() -> str:
    """Resolves the license CSV endpoint from `.env` or environment variables."""
    return get_clean_env("ONMYOJI_AUTH_CSV_URL")


def _read_wmic_uuid() -> str:
    """Attempts primary legacy WMIC hardware UUID lookup to preserve existing Machine IDs."""
    try:
        flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
        output = subprocess.check_output(
            "wmic csproduct get uuid",
            creationflags=flags,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        ).decode(errors="ignore")
        lines = [ln.strip() for ln in output.splitlines() if ln.strip() and ln.strip().upper() != "UUID"]
        if lines:
            candidate = lines[0]
            if candidate and set(candidate.replace("-", "").upper()) != {"F"}:
                return candidate
    except Exception:
        pass
    return ""


def _read_registry_machine_guid() -> str:
    """Fallback hardware identifier from Windows Cryptography MachineGuid (Win11 24H2+ safe)."""
    if sys.platform != "win32":
        return ""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as key:
            guid, _ = winreg.QueryValueEx(key, "MachineGuid")
            return str(guid).strip()
    except Exception:
        return ""


def get_machine_id() -> str:
    """Retrieves hardware UUID features to generate a 16-char uppercase SHA-256 Machine ID."""
    hw_uuid = _read_wmic_uuid()
    if not hw_uuid:
        hw_uuid = _read_registry_machine_guid()
    if not hw_uuid:
        import uuid as uuid_lib

        hw_uuid = str(uuid_lib.getnode())

    raw_id = f"ONMYOJI_{hw_uuid}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16].upper()


def _extract_authorized_hashes(csv_text: str) -> set[str]:
    """Parses normalized SHA-256 license hashes from remote CSV text."""
    hashes: set[str] = set()
    reader = csv.reader(io.StringIO(csv_text))
    for row in reader:
        for cell in row:
            cleaned = cell.strip().lstrip("'").lower()
            if cleaned:
                hashes.add(cleaned)
    return hashes


def check_activation(demo_mode: bool = False) -> None:
    """Verifies machine authorization when configured, or enters Public Sandbox Mode cleanly."""
    auth_url = get_auth_csv_url()
    if demo_mode or not auth_url:
        logger.info("🧪 [Auth] Public Sandbox / Demo Mode active (no private license endpoint required).")
        return

    machine_id = get_machine_id()
    auth_hash = hashlib.sha256(machine_id.encode("utf-8")).hexdigest().lower()
    timestamp = int(time.time())
    no_cache_url = f"{auth_url}&t={timestamp}" if "?" in auth_url else f"{auth_url}?t={timestamp}"

    try:
        req = urllib.request.Request(
            no_cache_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            csv_data = response.read().decode("utf-8")
            valid_hashes = _extract_authorized_hashes(csv_data)
            if auth_hash in valid_hashes or auth_hash in csv_data.lower():
                logger.info("✅ [Auth] 授權驗證通過，歡迎使用陰陽師鬥技助手！")
                return
    except Exception as exc:
        logger.warning(f"⚠️ [Auth] 無法連線至授權伺服器 ({exc})。")
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning("網絡錯誤", f"無法連線至驗證伺服器，請檢查網絡連線。\n(錯誤代碼: {exc})")
        root.destroy()
        sys.exit(1)

    show_unauthorized_and_exit(machine_id)


def show_unauthorized_and_exit(machine_id: str) -> NoReturn:
    """Displays Windows 11 Fluent-style activation UI and terminates application upon exit."""
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

    bg_color = "#202020"
    card_bg = "#2D2D2D"
    text_primary = "#FFFFFF"
    text_secondary = "#CCCCCC"
    accent_blue = "#4CC2FF"
    btn_gray = "#333333"

    root.configure(bg=bg_color)

    tk.Label(
        root,
        text="需要啟動系統",
        font=("Microsoft JhengHei", 18, "bold"),
        fg=text_primary,
        bg=bg_color,
    ).pack(pady=(25, 5), anchor="w", padx=35)

    tk.Label(
        root,
        text="此設備尚未綁定數位授權。請提供您的專屬識別碼以啟用功能。",
        font=("Microsoft JhengHei", 10),
        fg=text_secondary,
        bg=bg_color,
        wraplength=450,
        justify="left",
    ).pack(anchor="w", padx=35, pady=(0, 15))

    card = tk.Frame(root, bg=card_bg, bd=0)
    card.pack(fill=tk.X, padx=35, pady=5, ipadx=10, ipady=12)

    tk.Label(
        card,
        text="裝置識別碼 (Machine ID)",
        font=("Microsoft JhengHei", 9),
        fg=text_secondary,
        bg=card_bg,
    ).pack(anchor="w", padx=10, pady=(0, 2))

    entry = tk.Entry(
        card,
        justify="left",
        font=("Consolas", 13, "bold"),
        bg=card_bg,
        fg=accent_blue,
        readonlybackground=card_bg,
        insertbackground=text_primary,
        bd=0,
        highlightthickness=0,
    )
    entry.insert(0, machine_id)
    entry.config(state="readonly")
    entry.pack(anchor="w", padx=10, fill=tk.X)

    tk.Label(
        root,
        text="獲取授權請聯絡 Discord: alexc7832",
        font=("Microsoft JhengHei", 9),
        fg=text_secondary,
        bg=bg_color,
    ).pack(anchor="w", padx=35, pady=(10, 0))

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
        cursor="hand2",
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
        cursor="hand2",
    )
    close_btn.pack(side=tk.RIGHT)

    root.mainloop()
    sys.exit()
