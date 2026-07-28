import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.modules.billing import CreditManager, get_credit_manager, set_credit_manager, UsageTracker, get_usage_tracker, set_usage_tracker
from backend.modules.billing.payments import StripePaymentGateway, set_payment_gateway, get_payment_gateway, CREDIT_PACKAGES
from backend.modules.auth import AuthHandler, set_auth_handler, get_auth_handler
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter, get_rate_limiter
from backend.api.routes import set_pipeline, set_generator_loaded, set_storage, set_library, _batch_jobs
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.storage.file_storage import FileStorage
from backend.modules.storage.asset_library import AssetLibrary


AUTH_HEADER = "Authorization"


@pytest.fixture(autouse=True)
def test_setup():
    tmp = tempfile.mkdtemp()

    old_auth = get_auth_handler()
    auth_handler = AuthHandler(users_path=os.path.join(tmp, "users.json"))
    set_auth_handler(auth_handler)

    old_credits = get_credit_manager()
    cm = CreditManager(ledger_path=os.path.join(tmp, "ledger.json"))
    set_credit_manager(cm)

    old_limiter = get_rate_limiter()
    limiter = RateLimiter(max_requests=1000, window_seconds=60)
    set_rate_limiter(limiter)

    old_gateway = get_payment_gateway()
    test_gateway = StripePaymentGateway(api_key="sk_test_mock", webhook_secret="whsec_mock")
    set_payment_gateway(test_gateway)

    set_generator_loaded(False)
    set_storage(FileStorage(base_dir=os.path.join(tmp, "storage")))
    set_library(AssetLibrary(base_dir=os.path.join(tmp, "lib")))
    _batch_jobs.clear()

    yield

    set_payment_gateway(old_gateway)
    set_rate_limiter(old_limiter)
    set_credit_manager(old_credits)
    set_auth_handler(old_auth)


@pytest.fixture
def authed_client():
    pipe = AssetPipeline()
    from tests.test_api import FakeGenerator
    pipe.set_generator(FakeGenerator(num_images=1))
    set_pipeline(pipe)
    set_generator_loaded(True)

    tc = TestClient(app)
    resp = tc.post("/auth/register", json={
        "username": "billinguser",
        "password": "testpass123",
    })
    assert resp.status_code == 201
    token = resp.json()["access_token"]

    tc.headers = {"Authorization": f"Bearer {token}"}
    return tc


@pytest.fixture
def unauthed_client():
    pipe = AssetPipeline()
    from tests.test_api import FakeGenerator
    pipe.set_generator(FakeGenerator(num_images=1))
    set_pipeline(pipe)
    set_generator_loaded(True)
    return TestClient(app)


class TestCreditManager:
    def test_initial_balance_is_zero(self):
        cm = get_credit_manager()
        assert cm.get_balance("nonexistent") == 0

    def test_ensure_user_exists_grants_free_credits(self):
        cm = get_credit_manager()
        cm.ensure_user_exists("new_user")
        assert cm.get_balance("new_user") == 100

    def test_add_credits(self):
        cm = get_credit_manager()
        cm.ensure_user_exists("user_a")
        new_bal = cm.add_credits("user_a", 50, reason="purchase")
        assert new_bal == 150
        assert cm.get_balance("user_a") == 150

    def test_add_credits_negative_raises(self):
        cm = get_credit_manager()
        with pytest.raises(ValueError, match="Amount must be positive"):
            cm.add_credits("user_x", -10)

    def test_deduct_credits_success(self):
        cm = get_credit_manager()
        cm.ensure_user_exists("user_b")
        result = cm.deduct_credits("user_b", 10, reason="generation")
        assert result is True
        assert cm.get_balance("user_b") == 90

    def test_deduct_credits_insufficient(self):
        cm = get_credit_manager()
        cm.ensure_user_exists("user_c")
        result = cm.deduct_credits("user_c", 999, reason="generation")
        assert result is False
        assert cm.get_balance("user_c") == 100

    def test_deduct_credits_negative_raises(self):
        cm = get_credit_manager()
        with pytest.raises(ValueError, match="Amount must be positive"):
            cm.deduct_credits("user_y", -5)

    def test_transaction_history(self):
        cm = get_credit_manager()
        cm.ensure_user_exists("user_d")
        cm.add_credits("user_d", 50, reason="purchase")
        cm.deduct_credits("user_d", 30, reason="generation")
        txs = cm.get_transactions("user_d")
        assert len(txs) == 3
        assert txs[0]["reason"] == "generation"
        assert txs[0]["amount"] == -30
        assert txs[1]["reason"] == "purchase"
        assert txs[1]["amount"] == 50
        assert txs[2]["reason"] == "signup_bonus"
        assert txs[2]["amount"] == 100

    def test_multiple_users_independent(self):
        cm = get_credit_manager()
        cm.ensure_user_exists("alice")
        cm.ensure_user_exists("bob")
        cm.add_credits("alice", 200)
        cm.deduct_credits("bob", 10)
        assert cm.get_balance("alice") == 300
        assert cm.get_balance("bob") == 90

    def test_persistence_across_manager_instances(self):
        tmp = tempfile.mkdtemp()
        ledger_path = os.path.join(tmp, "ledger.json")
        cm1 = CreditManager(ledger_path=ledger_path)
        cm1.ensure_user_exists("persist_user")
        cm1.add_credits("persist_user", 25)

        cm2 = CreditManager(ledger_path=ledger_path)
        assert cm2.get_balance("persist_user") == 125
        txs = cm2.get_transactions("persist_user")
        assert len(txs) == 2

    def test_get_generation_cost_default(self):
        cm = get_credit_manager()
        assert cm.get_generation_cost() == 1

    def test_transaction_limit(self):
        cm = get_credit_manager()
        cm.ensure_user_exists("tx_user")
        for i in range(60):
            cm.add_credits("tx_user", 1, reason=f"test_{i}")
        txs = cm.get_transactions("tx_user", limit=10)
        assert len(txs) == 10

    def test_filter_transactions_by_reason(self):
        cm = get_credit_manager()
        cm.ensure_user_exists("filter_user")
        cm.add_credits("filter_user", 100, reason="purchase")
        cm.deduct_credits("filter_user", 10, reason="generation")
        cm.add_credits("filter_user", 50, reason="bonus")
        gen_txs = cm.get_transactions("filter_user", reason="generation")
        assert len(gen_txs) == 1
        assert gen_txs[0]["amount"] == -10
        assert gen_txs[0]["reason"] == "generation"
        purchase_txs = cm.get_transactions("filter_user", reason="purchase")
        assert len(purchase_txs) == 1
        assert purchase_txs[0]["amount"] == 100

    def test_filter_transactions_by_limit_with_reason(self):
        cm = get_credit_manager()
        cm.ensure_user_exists("multi_tx")
        for i in range(5):
            cm.add_credits("multi_tx", 10, reason="test")
            cm.deduct_credits("multi_tx", 5, reason="generation")
        gen_txs = cm.get_transactions("multi_tx", limit=2, reason="generation")
        assert len(gen_txs) == 2
        for t in gen_txs:
            assert t["reason"] == "generation"

    def test_refund_credits_adds_balance(self):
        cm = get_credit_manager()
        cm.ensure_user_exists("refund_user")
        cm.deduct_credits("refund_user", 30, reason="generation")
        before = cm.get_balance("refund_user")
        assert before == 70
        new_bal = cm.refund_credits("refund_user", 30, reason="refund_generation_failed")
        assert new_bal == 100
        assert cm.get_balance("refund_user") == 100

    def test_refund_credits_records_transaction(self):
        cm = get_credit_manager()
        cm.ensure_user_exists("refund_tx")
        cm.deduct_credits("refund_tx", 50, reason="generation")
        cm.refund_credits("refund_tx", 50, original_txn_id="txn_abc", reason="refund")
        txs = cm.get_transactions("refund_tx")
        assert len(txs) == 3
        newest = txs[0]
        assert newest["reason"] == "refund"
        assert newest["amount"] == 50
        assert newest.get("refunds_original_txn_id") == "txn_abc"

    def test_refund_negative_amount_raises(self):
        cm = get_credit_manager()
        with pytest.raises(ValueError, match="Amount must be positive"):
            cm.refund_credits("user_x", -10)

    def test_refund_nonexistent_user_creates_entry(self):
        cm = get_credit_manager()
        bal = cm.refund_credits("new_user_refund", 100, reason="signup_bonus")
        assert bal == 100
        assert cm.get_balance("new_user_refund") == 100


class TestBillingAPIAuth:
    def test_balance_requires_auth(self, unauthed_client):
        resp = unauthed_client.get("/billing/balance")
        assert resp.status_code == 401

    def test_topup_requires_auth(self, unauthed_client):
        resp = unauthed_client.post("/billing/topup", json={"amount": 50})
        assert resp.status_code == 401

    def test_transactions_requires_auth(self, unauthed_client):
        resp = unauthed_client.get("/billing/transactions")
        assert resp.status_code == 401

    def test_cost_estimate_does_not_require_auth(self, unauthed_client):
        resp = unauthed_client.get("/billing/cost-estimate?num_frames=4")
        assert resp.status_code == 200


class TestBillingAPI:
    def test_get_balance(self, authed_client):
        resp = authed_client.get("/billing/balance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] != ""
        assert data["balance"] == 100
        assert data["generation_cost"] == 1

    def test_topup_adds_credits(self, authed_client):
        resp = authed_client.post("/billing/topup", json={"amount": 50})
        assert resp.status_code == 200
        data = resp.json()
        assert data["amount_added"] == 50
        assert data["balance"] == 150

        bal = authed_client.get("/billing/balance")
        assert bal.json()["balance"] == 150

    def test_topup_zero_amount_returns_422(self, authed_client):
        resp = authed_client.post("/billing/topup", json={"amount": 0})
        assert resp.status_code == 422

    def test_topup_negative_amount_returns_422(self, authed_client):
        resp = authed_client.post("/billing/topup", json={"amount": -10})
        assert resp.status_code == 422

    def test_get_transactions(self, authed_client):
        authed_client.post("/billing/topup", json={"amount": 50, "reason": "gift"})
        resp = authed_client.get("/billing/transactions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["transactions"]) == 2
        assert data["transactions"][0]["reason"] == "gift"
        assert data["transactions"][0]["amount"] == 50
        assert data["transactions"][1]["reason"] == "signup_bonus"
        assert data["transactions"][1]["amount"] == 100

    def test_cost_estimate_default(self, authed_client):
        resp = authed_client.get("/billing/cost-estimate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["generation_cost"] == 1
        assert data["num_frames"] == 1
        assert data["total_cost"] == 1

    def test_cost_estimate_multiple_frames(self, authed_client):
        resp = authed_client.get("/billing/cost-estimate?num_frames=8")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_frames"] == 8
        assert data["total_cost"] == 8

    def test_refund_endpoint_refunds_credits(self, authed_client):
        resp = authed_client.post("/billing/refund", json={"amount": 30})
        assert resp.status_code == 200
        data = resp.json()
        assert data["amount_refunded"] == 30
        assert data["balance"] == 130

    def test_refund_endpoint_requires_auth(self, unauthed_client):
        resp = unauthed_client.post("/billing/refund", json={"amount": 50})
        assert resp.status_code == 401

    def test_refund_endpoint_negative_amount_returns_422(self, authed_client):
        resp = authed_client.post("/billing/refund", json={"amount": -10})
        assert resp.status_code == 422

    def test_refund_endpoint_zero_amount_returns_422(self, authed_client):
        resp = authed_client.post("/billing/refund", json={"amount": 0})
        assert resp.status_code == 422

    def test_refund_endpoint_with_original_txn_id(self, authed_client):
        txs = authed_client.get("/billing/transactions").json()["transactions"]
        txn_id = txs[0]["transaction_id"]
        resp = authed_client.post("/billing/refund", json={
            "amount": 50,
            "original_txn_id": txn_id,
            "reason": "customer_refund",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["amount_refunded"] == 50
        assert data["balance"] == 150

    def test_refund_shows_in_transactions(self, authed_client):
        authed_client.post("/billing/refund", json={"amount": 25, "reason": "adjustment"})
        txs = authed_client.get("/billing/transactions").json()["transactions"]
        refund_txs = [t for t in txs if t["reason"] == "adjustment"]
        assert len(refund_txs) == 1
        assert refund_txs[0]["amount"] == 25

    def test_get_transactions_with_reason_filter(self, authed_client):
        authed_client.post("/billing/topup", json={"amount": 50, "reason": "gift"})
        txs = authed_client.get("/billing/transactions?reason=gift").json()["transactions"]
        assert len(txs) == 1
        assert txs[0]["reason"] == "gift"

    def test_get_transactions_with_limit(self, authed_client):
        for i in range(5):
            authed_client.post("/billing/topup", json={"amount": 10, "reason": f"batch_{i}"})
        txs = authed_client.get("/billing/transactions?limit=2").json()["transactions"]
        assert len(txs) == 2

    def test_balance_isolation_between_users(self):
        pipe = AssetPipeline()
        from tests.test_api import FakeGenerator
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)

        tc = TestClient(app)

        r1 = tc.post("/auth/register", json={"username": "user1", "password": "pass1234"})
        t1 = r1.json()["access_token"]
        r2 = tc.post("/auth/register", json={"username": "user2", "password": "pass5678"})
        t2 = r2.json()["access_token"]

        b1 = tc.get("/billing/balance", headers={"Authorization": f"Bearer {t1}"}).json()
        b2 = tc.get("/billing/balance", headers={"Authorization": f"Bearer {t2}"}).json()
        assert b1["balance"] == 100
        assert b2["balance"] == 100

        tc.post("/billing/topup", json={"amount": 30}, headers={"Authorization": f"Bearer {t1}"})
        b1 = tc.get("/billing/balance", headers={"Authorization": f"Bearer {t1}"}).json()
        b2 = tc.get("/billing/balance", headers={"Authorization": f"Bearer {t2}"}).json()
        assert b1["balance"] == 130
        assert b2["balance"] == 100


class TestBillingGenerationIntegration:
    def test_generation_deducts_credits(self, authed_client):
        resp = authed_client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
        })
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        from tests.test_api import poll_job
        poll_job(authed_client, job_id)

        bal = authed_client.get("/billing/balance").json()
        assert bal["balance"] == 99

    def test_generation_insufficient_credits_returns_402(self, authed_client):
        bal = authed_client.get("/billing/balance").json()
        user_id = bal["user_id"]
        assert bal["balance"] == 100

        cm = get_credit_manager()
        cm.deduct_credits(user_id, 100)

        resp = authed_client.post("/generate", json={
            "asset_type": "character",
            "num_frames": 1,
        })
        assert resp.status_code == 402
        assert "Insufficient credits" in resp.json()["detail"]

    def test_generation_without_auth_still_works(self, unauthed_client):
        resp = unauthed_client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
        })
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        from tests.test_api import poll_job
        result = poll_job(unauthed_client, job_id)
        assert result["status"] == "done"

    def test_generation_cost_reflects_num_frames(self, authed_client):
        resp = authed_client.post("/generate", json={
            "asset_type": "character",
            "num_frames": 3,
        })
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        from tests.test_api import poll_job
        poll_job(authed_client, job_id)

        bal = authed_client.get("/billing/balance").json()
        assert bal["balance"] == 97

    def test_cached_generation_does_not_deduct_credits(self, authed_client):
        resp1 = authed_client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "palette": "retro_16",
            "seed": 42,
        })
        assert resp1.status_code == 202

        from tests.test_api import poll_job
        poll_job(authed_client, resp1.json()["job_id"])

        bal_after_first = authed_client.get("/billing/balance").json()["balance"]
        assert bal_after_first == 99

        resp2 = authed_client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "palette": "retro_16",
            "seed": 42,
        })
        assert resp2.status_code == 202
        poll_job(authed_client, resp2.json()["job_id"])

        bal_after_second = authed_client.get("/billing/balance").json()["balance"]
        assert bal_after_second == 99

    def test_batch_generation_deducts_credits(self, authed_client):
        resp = authed_client.post("/generate/batch", json={
            "items": [
                {"asset_type": "character", "view": "front"},
                {"asset_type": "enemy", "view": "side"},
            ]
        })
        assert resp.status_code == 202
        data = resp.json()

        from tests.test_api import poll_batch
        poll_batch(authed_client, data["batch_id"])

        bal = authed_client.get("/billing/balance").json()
        assert bal["balance"] == 98

    def test_batch_insufficient_credits_returns_402(self, authed_client):
        bal = authed_client.get("/billing/balance").json()
        user_id = bal["user_id"]
        assert bal["balance"] == 100

        cm = get_credit_manager()
        cm.deduct_credits(user_id, 99)

        resp = authed_client.post("/generate/batch", json={
            "items": [
                {"asset_type": "character"},
                {"asset_type": "enemy"},
            ]
        })
        assert resp.status_code == 402
        assert "Insufficient credits" in resp.json()["detail"]

    def test_batch_without_auth_still_works(self, unauthed_client):
        resp = unauthed_client.post("/generate/batch", json={
            "items": [{"asset_type": "character"}]
        })
        assert resp.status_code == 202
        data = resp.json()

        from tests.test_api import poll_batch
        result = poll_batch(unauthed_client, data["batch_id"])
        assert result["completed"] == 1


class TestStripePaymentGateway:
    def test_not_available_without_key(self):
        gw = StripePaymentGateway(api_key="")
        assert not gw.available

    def test_available_with_key(self):
        gw = StripePaymentGateway(api_key="sk_test_abc123")
        assert gw.available

    def test_create_checkout_session_without_key_returns_none(self):
        gw = StripePaymentGateway(api_key="")
        result = gw.create_checkout_session("starter", "user_1")
        assert result is None

    def test_get_packages_returns_all(self):
        gw = StripePaymentGateway(api_key="sk_test_abc")
        pkgs = gw.get_packages()
        assert "starter" in pkgs
        assert "pro" in pkgs
        assert "studio" in pkgs
        assert pkgs["starter"]["credits"] == 100

    def test_create_checkout_session_unknown_package(self):
        gw = StripePaymentGateway(api_key="sk_test_abc")
        result = gw.create_checkout_session("nonexistent", "user_1")
        assert result is None

    def test_create_checkout_session_calls_stripe(self):
        gw = StripePaymentGateway(api_key="sk_test_abc")
        import unittest.mock as mock
        fake_session = mock.MagicMock()
        fake_session.id = "cs_test_abc123"
        fake_session.url = "https://checkout.stripe.com/pay/cs_test_abc123"
        with mock.patch("stripe.checkout.Session.create", return_value=fake_session):
            result = gw.create_checkout_session("starter", "user_42")
            assert result is not None
            assert result["session_id"] == "cs_test_abc123"
            assert "checkout.stripe.com" in result["url"]

    def test_handle_webhook_without_key_returns_none(self):
        gw = StripePaymentGateway(api_key="", webhook_secret="")
        result = gw.handle_webhook(b"{}", "sig")
        assert result is None

    def test_handle_webhook_valid_payment(self):
        gw = StripePaymentGateway(api_key="sk_test_abc", webhook_secret="whsec_test")
        import unittest.mock as mock
        fake_event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test_xyz",
                    "payment_status": "paid",
                    "client_reference_id": "user_99",
                    "metadata": {"package": "pro", "credits": "500"},
                }
            },
        }
        with mock.patch("stripe.Webhook.construct_event", return_value=fake_event):
            result = gw.handle_webhook(b"{}", "sig")
            assert result is not None
            assert result["user_id"] == "user_99"
            assert result["credits"] == 500
            assert result["package"] == "pro"

    def test_handle_webhook_unpaid_ignored(self):
        gw = StripePaymentGateway(api_key="sk_test_abc", webhook_secret="whsec_test")
        import unittest.mock as mock
        fake_event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test_unpaid",
                    "payment_status": "unpaid",
                    "client_reference_id": "user_99",
                    "metadata": {"package": "pro", "credits": "500"},
                }
            },
        }
        with mock.patch("stripe.Webhook.construct_event", return_value=fake_event):
            result = gw.handle_webhook(b"{}", "sig")
            assert result is None

    def test_handle_webhook_wrong_event_type(self):
        gw = StripePaymentGateway(api_key="sk_test_abc", webhook_secret="whsec_test")
        import unittest.mock as mock
        fake_event = {"type": "payment_intent.succeeded", "data": {"object": {}}}
        with mock.patch("stripe.Webhook.construct_event", return_value=fake_event):
            result = gw.handle_webhook(b"{}", "sig")
            assert result is None

    def test_handle_webhook_signature_failure(self):
        gw = StripePaymentGateway(api_key="sk_test_abc", webhook_secret="whsec_test")
        import unittest.mock as mock
        with mock.patch("stripe.Webhook.construct_event", side_effect=ValueError("bad sig")):
            result = gw.handle_webhook(b"{}", "bad_sig")
            assert result is None

    def test_handle_webhook_no_user_id(self):
        gw = StripePaymentGateway(api_key="sk_test_abc", webhook_secret="whsec_test")
        import unittest.mock as mock
        fake_event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test_no_user",
                    "payment_status": "paid",
                    "client_reference_id": "",
                    "metadata": {"package": "pro", "credits": "500"},
                }
            },
        }
        with mock.patch("stripe.Webhook.construct_event", return_value=fake_event):
            result = gw.handle_webhook(b"{}", "sig")
            assert result is None


class TestPaymentAPI:
    def test_list_packages(self, authed_client):
        resp = authed_client.get("/billing/packages")
        assert resp.status_code == 200
        data = resp.json()
        assert "packages" in data
        keys = [p["key"] for p in data["packages"]]
        assert "starter" in keys
        assert "pro" in keys
        assert "studio" in keys

    def test_create_checkout_requires_auth(self, unauthed_client):
        resp = unauthed_client.post("/billing/create-checkout-session", json={
            "package": "starter",
        })
        assert resp.status_code == 401

    def test_create_checkout_unknown_package(self, authed_client):
        resp = authed_client.post("/billing/create-checkout-session", json={
            "package": "nonexistent",
        })
        assert resp.status_code == 400

    def test_create_checkout_calls_stripe(self, authed_client):
        import unittest.mock as mock
        fake_session = mock.MagicMock()
        fake_session.id = "cs_test_mock_123"
        fake_session.url = "https://checkout.stripe.com/pay/cs_test_mock_123"
        with mock.patch("stripe.checkout.Session.create", return_value=fake_session):
            resp = authed_client.post("/billing/create-checkout-session", json={
                "package": "pro",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "cs_test_mock_123"
        assert "checkout.stripe.com" in data["url"]

    def test_webhook_adds_credits(self, authed_client):
        import unittest.mock as mock
        bal_before = authed_client.get("/billing/balance").json()
        user_id = bal_before["user_id"]
        fake_event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test_wh",
                    "payment_status": "paid",
                    "client_reference_id": user_id,
                    "metadata": {"package": "pro", "credits": "500"},
                }
            },
        }
        with mock.patch("stripe.Webhook.construct_event", return_value=fake_event):
            resp = authed_client.post(
                "/billing/webhook",
                content=b"{}",
                headers={"stripe-signature": "test_sig"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["credits_added"] == 500

        bal = authed_client.get("/billing/balance").json()
        assert bal["balance"] == 600

    def test_webhook_unpaid_ignored(self, authed_client):
        import unittest.mock as mock
        fake_event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test_unpaid",
                    "payment_status": "unpaid",
                    "client_reference_id": "billinguser",
                    "metadata": {"package": "pro", "credits": "500"},
                }
            },
        }
        with mock.patch("stripe.Webhook.construct_event", return_value=fake_event):
            resp = authed_client.post(
                "/billing/webhook",
                content=b"{}",
                headers={"stripe-signature": "test_sig"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ignored"

    def test_webhook_without_gateway_returns_503(self):
        old_gw = get_payment_gateway()
        set_payment_gateway(StripePaymentGateway(api_key=""))
        try:
            pipe = AssetPipeline()
            from tests.test_api import FakeGenerator
            pipe.set_generator(FakeGenerator(num_images=1))
            set_pipeline(pipe)
            set_generator_loaded(True)
            tc = TestClient(app)
            resp = tc.post("/billing/webhook", content=b"{}", headers={"stripe-signature": "sig"})
            assert resp.status_code == 503
        finally:
            set_payment_gateway(old_gw)


class TestUsageTracker:
    def test_initial_usage_is_zero(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        ut = UsageTracker(path=os.path.join(tmp, "usage.json"))
        assert ut.get_user_daily_usage("user_x") == {"generations": 0, "credits_used": 0}
        assert ut.get_user_monthly_usage("user_x") == {"generations": 0, "credits_used": 0}

    def test_record_generation_increments(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        ut = UsageTracker(path=os.path.join(tmp, "usage.json"))
        ut.record_generation("user_a", 5)
        assert ut.get_user_daily_usage("user_a") == {"generations": 1, "credits_used": 5}

    def test_multiple_generations_accumulate(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        ut = UsageTracker(path=os.path.join(tmp, "usage.json"))
        ut.record_generation("user_a", 3)
        ut.record_generation("user_a", 7)
        daily = ut.get_user_daily_usage("user_a")
        assert daily["generations"] == 2
        assert daily["credits_used"] == 10

    def test_separate_users_independent(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        ut = UsageTracker(path=os.path.join(tmp, "usage.json"))
        ut.record_generation("user_a", 10)
        ut.record_generation("user_b", 20)
        assert ut.get_user_daily_usage("user_a") == {"generations": 1, "credits_used": 10}
        assert ut.get_user_daily_usage("user_b") == {"generations": 1, "credits_used": 20}

    def test_system_totals(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        ut = UsageTracker(path=os.path.join(tmp, "usage.json"))
        ut.record_generation("user_a", 10)
        ut.record_generation("user_b", 20)
        totals = ut.get_system_totals()
        assert totals["total_generations"] == 2
        assert totals["total_credits_used"] == 30

    def test_get_all_user_ids(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        ut = UsageTracker(path=os.path.join(tmp, "usage.json"))
        ut.record_generation("user_a", 1)
        ut.record_generation("user_b", 1)
        assert sorted(ut.get_all_user_ids()) == ["user_a", "user_b"]

    def test_get_user_all_usage_structure(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        ut = UsageTracker(path=os.path.join(tmp, "usage.json"))
        ut.record_generation("user_a", 5)
        ut.record_generation("user_a", 3)
        all_usage = ut.get_user_all_usage("user_a")
        assert all_usage["total"]["generations"] == 2
        assert all_usage["total"]["credits_used"] == 8
        assert len(all_usage["days"]) == 1

    def test_persistence(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "usage.json")
        ut1 = UsageTracker(path=path)
        ut1.record_generation("user_persist", 7)
        ut2 = UsageTracker(path=path)
        assert ut2.get_user_daily_usage("user_persist") == {"generations": 1, "credits_used": 7}
        assert ut2.get_system_totals()["total_generations"] == 1

    def test_date_specific_query(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        ut = UsageTracker(path=os.path.join(tmp, "usage.json"))
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ut.record_generation("user_date", 5)
        result = ut.get_user_daily_usage("user_date", date=today)
        assert result["generations"] == 1
        result_wrong = ut.get_user_daily_usage("user_date", date="2099-01-01")
        assert result_wrong["generations"] == 0


class TestBillingAdminAPI:
    def test_admin_list_users_requires_admin(self, unauthed_client):
        resp = unauthed_client.get("/billing/admin/users")
        assert resp.status_code == 401

    def test_admin_list_users_non_admin_returns_403(self, authed_client):
        resp = authed_client.get("/billing/admin/users")
        assert resp.status_code == 403

    def test_admin_list_users_as_admin(self):
        pipe = AssetPipeline()
        from tests.test_api import FakeGenerator
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)

        tc = TestClient(app)
        token = self._register_admin(tc)

        resp = tc.get("/billing/admin/users", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert "users" in data
        assert len(data["users"]) >= 0

    def _register_admin(self, tc):
        r = tc.post("/auth/register", json={"username": "admin", "password": "adminpass123"})
        assert r.status_code == 201
        return r.json()["access_token"]

    def test_admin_adjust_adds_credits(self):
        pipe = AssetPipeline()
        from tests.test_api import FakeGenerator
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)

        tc = TestClient(app)
        admin_token = self._register_admin(tc)

        resp = tc.post(
            "/billing/admin/adjust",
            json={"user_id": "target_user_adj1", "amount": 200, "reason": "admin_grant"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["amount_changed"] == 200
        assert data["balance"] >= 200

    def test_admin_adjust_deducts_credits(self):
        pipe = AssetPipeline()
        from tests.test_api import FakeGenerator
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)

        tc = TestClient(app)
        admin_token = self._register_admin(tc)

        cm = get_credit_manager()
        cm.ensure_user_exists("deduct_target")
        cm.add_credits("deduct_target", 100)
        before = cm.get_balance("deduct_target")

        resp = tc.post(
            "/billing/admin/adjust",
            json={"user_id": "deduct_target", "amount": -30, "reason": "penalty"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["amount_changed"] == -30
        assert data["balance"] == before - 30

    def test_admin_adjust_zero_returns_422(self):
        pipe = AssetPipeline()
        from tests.test_api import FakeGenerator
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)

        tc = TestClient(app)
        admin_token = self._register_admin(tc)

        resp = tc.post(
            "/billing/admin/adjust",
            json={"user_id": "someuser", "amount": 0},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 422

    def test_admin_usage_requires_admin(self, unauthed_client):
        resp = unauthed_client.get("/billing/admin/usage")
        assert resp.status_code == 401

    def test_admin_usage_returns_system_totals(self):
        pipe = AssetPipeline()
        from tests.test_api import FakeGenerator
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)

        import tempfile
        tmp = tempfile.mkdtemp()
        ut = UsageTracker(path=os.path.join(tmp, "usage_admin.json"))
        old_tracker = get_usage_tracker()
        set_usage_tracker(ut)
        try:
            ut.record_generation("user1", 10)
            ut.record_generation("user2", 20)

            tc = TestClient(app)
            admin_token = self._register_admin(tc)

            resp = tc.get("/billing/admin/usage", headers={"Authorization": f"Bearer {admin_token}"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_generations"] == 2
            assert data["total_credits_used"] == 30
        finally:
            set_usage_tracker(old_tracker)

    def test_admin_usage_with_user_id(self):
        pipe = AssetPipeline()
        from tests.test_api import FakeGenerator
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)

        import tempfile
        tmp = tempfile.mkdtemp()
        ut = UsageTracker(path=os.path.join(tmp, "usage_user.json"))
        old_tracker = get_usage_tracker()
        set_usage_tracker(ut)
        try:
            ut.record_generation("specific_user", 15)

            tc = TestClient(app)
            admin_token = self._register_admin(tc)

            resp = tc.get(
                "/billing/admin/usage?user_id=specific_user",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["user_id"] == "specific_user"
            assert data["daily"]["generations"] == 1
            assert data["daily"]["credits_used"] == 15
            assert data["monthly"]["generations"] == 1
            assert data["all_time"]["total"]["generations"] == 1
        finally:
            set_usage_tracker(old_tracker)

    def test_generation_records_usage(self):
        pipe = AssetPipeline()
        from tests.test_api import FakeGenerator
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)

        import tempfile
        tmp = tempfile.mkdtemp()
        ut = UsageTracker(path=os.path.join(tmp, "usage_gen.json"))
        old_tracker = get_usage_tracker()
        set_usage_tracker(ut)
        try:
            tc = TestClient(app)
            r = tc.post("/auth/register", json={"username": "gen_user1", "password": "genpass123"})
            assert r.status_code == 201
            token = r.json()["access_token"]
            tc.headers = {"Authorization": f"Bearer {token}"}

            resp = tc.post("/generate", json={
                "asset_type": "character",
                "num_frames": 2,
            })
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            from tests.test_api import poll_job
            poll_job(tc, job_id)

            bal = tc.get("/billing/balance").json()
            assert bal["balance"] == 98

            usage = ut.get_user_daily_usage(bal["user_id"])
            assert usage["generations"] == 1
            assert usage["credits_used"] == 2
        finally:
            set_usage_tracker(old_tracker)
