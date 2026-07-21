import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.modules.billing import CreditManager, get_credit_manager, set_credit_manager
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
