import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

GENERATION_COST = int(os.environ.get("BILLING_GENERATION_COST", "1"))
FREE_CREDITS = int(os.environ.get("BILLING_FREE_CREDITS", "100"))


@dataclass
class TransactionRecord:
    user_id: str
    amount: int
    reason: str
    timestamp: str = ""
    transaction_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.transaction_id:
            import uuid
            self.transaction_id = str(uuid.uuid4())[:12]


@dataclass
class CreditBalance:
    user_id: str
    balance: int
    transactions: list = field(default_factory=list)


class CreditManager:
    def __init__(self, ledger_path: str = "data/billing/ledger.json"):
        self.ledger_path = ledger_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.ledger_path) or ".", exist_ok=True)

    def _load_ledger(self) -> dict:
        if not os.path.isfile(self.ledger_path):
            return {}
        with open(self.ledger_path, "r") as f:
            return json.load(f)

    def _save_ledger(self, ledger: dict):
        with open(self.ledger_path, "w") as f:
            json.dump(ledger, f, indent=2)

    def get_balance(self, user_id: str) -> int:
        with self._lock:
            ledger = self._load_ledger()
            entry = ledger.get(user_id, {})
            return entry.get("balance", 0)

    def get_transactions(self, user_id: str, limit: int = 50) -> list:
        with self._lock:
            ledger = self._load_ledger()
            entry = ledger.get(user_id, {})
            txs = entry.get("transactions", [])
            return list(reversed(txs))[:limit]

    def add_credits(self, user_id: str, amount: int, reason: str = "topup") -> int:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        with self._lock:
            ledger = self._load_ledger()
            entry = ledger.setdefault(user_id, {"balance": 0, "transactions": []})
            entry["balance"] += amount
            txn = TransactionRecord(
                user_id=user_id,
                amount=amount,
                reason=reason,
            )
            entry["transactions"].append({
                "transaction_id": txn.transaction_id,
                "amount": txn.amount,
                "reason": txn.reason,
                "timestamp": txn.timestamp,
            })
            self._save_ledger(ledger)
            return entry["balance"]

    def deduct_credits(self, user_id: str, amount: int, reason: str = "generation") -> bool:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        with self._lock:
            ledger = self._load_ledger()
            entry = ledger.setdefault(user_id, {"balance": 0, "transactions": []})
            if entry["balance"] < amount:
                return False
            entry["balance"] -= amount
            txn = TransactionRecord(
                user_id=user_id,
                amount=-amount,
                reason=reason,
            )
            entry["transactions"].append({
                "transaction_id": txn.transaction_id,
                "amount": txn.amount,
                "reason": txn.reason,
                "timestamp": txn.timestamp,
            })
            self._save_ledger(ledger)
            return True

    def ensure_user_exists(self, user_id: str):
        with self._lock:
            ledger = self._load_ledger()
            if user_id not in ledger:
                ledger[user_id] = {"balance": FREE_CREDITS, "transactions": []}
                txn = TransactionRecord(
                    user_id=user_id,
                    amount=FREE_CREDITS,
                    reason="signup_bonus",
                )
                ledger[user_id]["transactions"].append({
                    "transaction_id": txn.transaction_id,
                    "amount": txn.amount,
                    "reason": txn.reason,
                    "timestamp": txn.timestamp,
                })
                self._save_ledger(ledger)

    def get_generation_cost(self) -> int:
        return GENERATION_COST


_default_credit_manager = CreditManager()


def get_credit_manager() -> CreditManager:
    return _default_credit_manager


def set_credit_manager(cm: CreditManager):
    global _default_credit_manager
    _default_credit_manager = cm
