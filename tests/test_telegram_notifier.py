import pytest

from telegram_notifier import TelegramNotifier


def test_defaults_to_dry_run_without_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    notifier = TelegramNotifier()
    assert notifier.dry_run is True


def test_dry_run_send_records_history_without_network_call(capsys):
    notifier = TelegramNotifier(dry_run=True)
    notifier.send("hello", level="WARNING")
    out = capsys.readouterr().out
    assert "hello" in out
    assert "level=WARNING" in out
    assert notifier.history == [{"message": "hello", "level": "WARNING", "dry_run": True}]


def test_explicit_dry_run_true_wins_even_with_credentials(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    notifier = TelegramNotifier(dry_run=True)
    assert notifier.dry_run is True


def test_real_credentials_set_dry_run_false(monkeypatch):
    notifier = TelegramNotifier(bot_token="fake-token", chat_id="12345")
    assert notifier.dry_run is False


def test_real_send_posts_to_telegram_api(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def fake_post(url, json, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["timeout"] = timeout
        return FakeResponse()

    import telegram_notifier
    monkeypatch.setattr(telegram_notifier.requests, "post", fake_post)

    notifier = TelegramNotifier(bot_token="fake-token", chat_id="12345")
    notifier.send("real send test")

    assert calls["url"] == "https://api.telegram.org/botfake-token/sendMessage"
    assert calls["json"] == {"chat_id": "12345", "text": "real send test"}


def test_real_send_raises_on_telegram_api_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": False, "description": "chat not found"}

    import telegram_notifier
    monkeypatch.setattr(telegram_notifier.requests, "post", lambda *a, **k: FakeResponse())

    notifier = TelegramNotifier(bot_token="fake-token", chat_id="12345")
    with pytest.raises(RuntimeError, match="chat not found"):
        notifier.send("will fail")
