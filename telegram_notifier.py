"""
telegram_notifier.py -- delivers a brief via the Telegram Bot API.

Shaped like IQRF_V2's own BaseAlertChannel contract
(send(message, level="INFO") -> None) on purpose -- structurally
compatible, no import of it -- so if the user ever wants to wire
this into IQRF_V2's own AlertManager themselves, this class can be
dropped into AlertManager(channels=[...]) with no changes needed on
either side. See README.md's "Wiring into IQRF_V2" section; this
project does not do that wiring itself.

Dry-run by default: with no TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID
configured, send() prints exactly what would have been sent instead
of making a network call, so the rest of the pipeline is fully
runnable and testable without a bot token.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

_API_BASE = "https://api.telegram.org"


class TelegramNotifier:
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        dry_run: Optional[bool] = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        # Explicit dry_run always wins; otherwise dry-run iff either
        # credential is missing -- never silently attempt a call that
        # can only fail.
        self.dry_run = dry_run if dry_run is not None else not (self.bot_token and self.chat_id)
        self.history = []  # every send() call, dry-run or real, for tests/inspection

    def send(self, message: str, level: str = "INFO") -> None:
        self.history.append({"message": message, "level": level, "dry_run": self.dry_run})

        if self.dry_run:
            print(f"[telegram dry-run, level={level}] {message}")
            return

        url = f"{_API_BASE}/bot{self.bot_token}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": self.chat_id, "text": message},
            timeout=15,
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(f"Telegram API returned ok=false: {body}")
