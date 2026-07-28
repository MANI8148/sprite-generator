import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional


class UsageTracker:
    def __init__(self, path: str = "data/billing/usage.json"):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def _load(self) -> dict:
        if not os.path.isfile(self.path):
            return {}
        with open(self.path, "r") as f:
            return json.load(f)

    def _save(self, data: dict):
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _date_key(self, dt=None) -> str:
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%Y-%m-%d")

    def _month_key(self, dt=None) -> str:
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%Y-%m")

    def record_generation(self, user_id: str, credits_spent: int):
        with self._lock:
            data = self._load()
            today = self._date_key()
            month = self._month_key()
            user_daily = data.setdefault(user_id, {}).setdefault(today, {"generations": 0, "credits_used": 0})
            user_daily["generations"] += 1
            user_daily["credits_used"] += credits_spent
            monthly = data.setdefault("_monthly", {}).setdefault(month, {"generations": 0, "credits_used": 0})
            monthly["generations"] += 1
            monthly["credits_used"] += credits_spent
            self._save(data)

    def get_user_daily_usage(self, user_id: str, date: Optional[str] = None) -> dict:
        with self._lock:
            data = self._load()
            key = date or self._date_key()
            entry = data.get(user_id, {}).get(key, {})
            return {"generations": entry.get("generations", 0), "credits_used": entry.get("credits_used", 0)}

    def get_user_monthly_usage(self, user_id: str, month: Optional[str] = None) -> dict:
        with self._lock:
            data = self._load()
            key = month or self._month_key()
            total_generations = 0
            total_credits = 0
            for day_key, day_data in data.get(user_id, {}).items():
                if day_key.startswith(key) and day_key != "_monthly":
                    total_generations += day_data.get("generations", 0)
                    total_credits += day_data.get("credits_used", 0)
            return {"generations": total_generations, "credits_used": total_credits}

    def get_user_all_usage(self, user_id: str) -> dict:
        with self._lock:
            data = self._load()
            user_data = data.get(user_id, {})
            total = {"generations": 0, "credits_used": 0}
            days = []
            for day_key in sorted(user_data.keys()):
                if day_key == "_monthly":
                    continue
                day_info = user_data[day_key]
                total["generations"] += day_info.get("generations", 0)
                total["credits_used"] += day_info.get("credits_used", 0)
                days.append({"date": day_key, **day_info})
            return {"total": total, "days": days}

    def get_system_totals(self) -> dict:
        with self._lock:
            data = self._load()
            monthly = data.get("_monthly", {})
            total_generations = 0
            total_credits_used = 0
            for month_data in monthly.values():
                total_generations += month_data.get("generations", 0)
                total_credits_used += month_data.get("credits_used", 0)
            return {"total_generations": total_generations, "total_credits_used": total_credits_used}

    def get_all_user_ids(self) -> list:
        with self._lock:
            data = self._load()
            return [uid for uid in data.keys() if uid != "_monthly"]


_default_usage_tracker = UsageTracker()


def get_usage_tracker() -> UsageTracker:
    return _default_usage_tracker


def set_usage_tracker(tracker: UsageTracker):
    global _default_usage_tracker
    _default_usage_tracker = tracker
