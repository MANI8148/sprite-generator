import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

CREDITS_PER_UNIT_CENTS = int(os.environ.get("STRIPE_CREDITS_PER_UNIT_CENTS", "1"))

CREDIT_PACKAGES: dict[str, dict] = {
    "starter": {
        "credits": 100,
        "amount_cents": 499,
        "description": "100 credits — $4.99",
        "price_id": os.environ.get("STRIPE_PRICE_STARTER", ""),
    },
    "pro": {
        "credits": 500,
        "amount_cents": 1999,
        "description": "500 credits — $19.99",
        "price_id": os.environ.get("STRIPE_PRICE_PRO", ""),
    },
    "studio": {
        "credits": 2000,
        "amount_cents": 4999,
        "description": "2000 credits — $49.99",
        "price_id": os.environ.get("STRIPE_PRICE_STUDIO", ""),
    },
}


class StripePaymentGateway:
    def __init__(
        self,
        api_key: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("STRIPE_API_KEY", "")
        self.webhook_secret = webhook_secret or os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        self._available = bool(self.api_key)
        if self._available:
            import stripe as _stripe
            _stripe.api_key = self.api_key

    @property
    def available(self) -> bool:
        return self._available

    def create_checkout_session(
        self,
        package_key: str,
        user_id: str,
        success_url: str = "https://example.com/success",
        cancel_url: str = "https://example.com/cancel",
    ) -> Optional[dict]:
        if not self._available:
            return None
        package = CREDIT_PACKAGES.get(package_key)
        if package is None:
            return None
        import stripe as _stripe
        session = _stripe.checkout.Session.create(
            mode="payment",
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": f"Sprite Generator — {package['description']}"},
                    "unit_amount": package["amount_cents"],
                },
                "quantity": 1,
            }],
            client_reference_id=user_id,
            metadata={"package": package_key, "credits": str(package["credits"])},
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return {"session_id": session.id, "url": session.url}

    def handle_webhook(self, payload: bytes, sig_header: str) -> Optional[dict]:
        if not self._available or not self.webhook_secret:
            return None
        import stripe as _stripe
        try:
            event = _stripe.Webhook.construct_event(payload, sig_header, self.webhook_secret)
        except (_stripe.error.SignatureVerificationError, ValueError):
            logger.warning("Stripe webhook signature verification failed")
            return None
        if event["type"] == "checkout.session.completed":
            session = event["data"]["object"]
            payment_status = session.get("payment_status", "")
            if payment_status != "paid":
                return None
            user_id = session.get("client_reference_id", "")
            credits_str = session.get("metadata", {}).get("credits", "0")
            try:
                credits = int(credits_str)
            except (ValueError, TypeError):
                credits = 0
            if user_id and credits > 0:
                session_id = getattr(session, "id", None) or session.get("id", "")
                return {
                    "user_id": user_id,
                    "credits": credits,
                    "package": session.get("metadata", {}).get("package", ""),
                    "stripe_session_id": session_id,
                }
        return None

    def get_packages(self) -> dict:
        return {
            key: {
                "credits": pkg["credits"],
                "amount_cents": pkg["amount_cents"],
                "description": pkg["description"],
            }
            for key, pkg in CREDIT_PACKAGES.items()
        }


_default_payment_gateway: Optional[StripePaymentGateway] = None


def get_payment_gateway() -> StripePaymentGateway:
    global _default_payment_gateway
    if _default_payment_gateway is None:
        _default_payment_gateway = StripePaymentGateway()
    return _default_payment_gateway


def set_payment_gateway(gw: StripePaymentGateway) -> None:
    global _default_payment_gateway
    _default_payment_gateway = gw
