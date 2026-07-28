import os
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
from typing import Optional

from backend.modules.auth import get_current_user, OptionalAuth, TokenData
from backend.modules.billing import CreditManager, get_credit_manager
from backend.modules.billing.usage import UsageTracker, get_usage_tracker
from backend.modules.billing.payments import StripePaymentGateway, get_payment_gateway, CREDIT_PACKAGES

router = APIRouter(prefix="/billing", tags=["billing"])

ADMIN_USERNAME = os.environ.get("BILLING_ADMIN_USERNAME", "admin")


def require_admin(current_user: TokenData = Depends(get_current_user)):
    if current_user.username != ADMIN_USERNAME:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


class BalanceResponse(BaseModel):
    user_id: str
    balance: int
    generation_cost: int


class TopupRequest(BaseModel):
    amount: int
    reason: str = "topup"


class TopupResponse(BaseModel):
    user_id: str
    balance: int
    amount_added: int


class TransactionEntry(BaseModel):
    transaction_id: str
    amount: int
    reason: str
    timestamp: str


class TransactionHistoryResponse(BaseModel):
    user_id: str
    transactions: list


class CostEstimateResponse(BaseModel):
    generation_cost: int
    num_frames: int
    total_cost: int


class RefundRequest(BaseModel):
    amount: int
    original_txn_id: str = ""
    reason: str = "refund"


@router.get("/balance", response_model=BalanceResponse)
def get_balance(
    current_user: TokenData = Depends(get_current_user),
    credits: CreditManager = Depends(get_credit_manager),
):
    credits.ensure_user_exists(current_user.user_id)
    balance = credits.get_balance(current_user.user_id)
    return BalanceResponse(
        user_id=current_user.user_id,
        balance=balance,
        generation_cost=credits.get_generation_cost(),
    )


@router.post("/topup", response_model=TopupResponse)
def topup(
    req: TopupRequest,
    current_user: TokenData = Depends(get_current_user),
    credits: CreditManager = Depends(get_credit_manager),
):
    credits.ensure_user_exists(current_user.user_id)
    if req.amount <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Amount must be positive",
        )
    new_balance = credits.add_credits(current_user.user_id, req.amount, reason=req.reason)
    return TopupResponse(
        user_id=current_user.user_id,
        balance=new_balance,
        amount_added=req.amount,
    )


@router.get("/transactions", response_model=TransactionHistoryResponse)
def get_transactions(
    current_user: TokenData = Depends(get_current_user),
    credits: CreditManager = Depends(get_credit_manager),
    reason: Optional[str] = None,
    limit: int = 50,
):
    credits.ensure_user_exists(current_user.user_id)
    txs = credits.get_transactions(current_user.user_id, limit=limit, reason=reason)
    return TransactionHistoryResponse(
        user_id=current_user.user_id,
        transactions=[TransactionEntry(**t) for t in txs],
    )


@router.get("/cost-estimate", response_model=CostEstimateResponse)
def cost_estimate(
    num_frames: int = 1,
    credits: CreditManager = Depends(get_credit_manager),
):
    unit_cost = credits.get_generation_cost()
    return CostEstimateResponse(
        generation_cost=unit_cost,
        num_frames=num_frames,
        total_cost=unit_cost * num_frames,
    )


@router.post("/refund")
def refund_credits(
    req: RefundRequest,
    current_user: TokenData = Depends(get_current_user),
    credits: CreditManager = Depends(get_credit_manager),
):
    if req.amount <= 0:
        raise HTTPException(status_code=422, detail="Amount must be positive")
    credits.ensure_user_exists(current_user.user_id)
    new_balance = credits.refund_credits(
        current_user.user_id,
        req.amount,
        original_txn_id=req.original_txn_id,
        reason=req.reason,
    )
    return {
        "user_id": current_user.user_id,
        "balance": new_balance,
        "amount_refunded": req.amount,
    }


class PackageResponse(BaseModel):
    key: str
    credits: int
    amount_cents: int
    description: str


class PackagesListResponse(BaseModel):
    packages: list


@router.get("/packages")
def list_packages():
    pkgs = []
    for key, pkg in CREDIT_PACKAGES.items():
        pkgs.append(PackageResponse(
            key=key,
            credits=pkg["credits"],
            amount_cents=pkg["amount_cents"],
            description=pkg["description"],
        ))
    return PackagesListResponse(packages=pkgs)


class CheckoutRequest(BaseModel):
    package: str
    success_url: str = "https://example.com/success"
    cancel_url: str = "https://example.com/cancel"


class CheckoutResponse(BaseModel):
    session_id: str
    url: str


@router.post("/create-checkout-session", response_model=CheckoutResponse)
def create_checkout_session(
    req: CheckoutRequest,
    current_user: TokenData = Depends(get_current_user),
    gateway: StripePaymentGateway = Depends(get_payment_gateway),
):
    if not gateway.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment gateway not configured. Set STRIPE_API_KEY environment variable.",
        )
    if req.package not in CREDIT_PACKAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown package '{req.package}'. Available: {list(CREDIT_PACKAGES.keys())}",
        )
    result = gateway.create_checkout_session(
        package_key=req.package,
        user_id=current_user.user_id,
        success_url=req.success_url,
        cancel_url=req.cancel_url,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create checkout session",
        )
    return CheckoutResponse(**result)


class WebhookResponse(BaseModel):
    status: str
    credits_added: int = 0
    user_id: str = ""


@router.post("/webhook", response_model=WebhookResponse)
async def stripe_webhook(
    request: Request,
    gateway: StripePaymentGateway = Depends(get_payment_gateway),
    credits: CreditManager = Depends(get_credit_manager),
):
    if not gateway.available:
        raise HTTPException(status_code=503, detail="Payment gateway not configured")
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    result = gateway.handle_webhook(payload, sig_header)
    if result is None:
        return WebhookResponse(status="ignored")
    credits.ensure_user_exists(result["user_id"])
    credits.add_credits(result["user_id"], result["credits"], reason=f"stripe_purchase:{result.get('package', 'unknown')}")
    return WebhookResponse(
        status="completed",
        credits_added=result["credits"],
        user_id=result["user_id"],
    )


class UserBalanceEntry(BaseModel):
    user_id: str
    balance: int


class AllUsersResponse(BaseModel):
    users: list


@router.get("/admin/users")
def admin_list_users(
    current_user: TokenData = Depends(require_admin),
    credits: CreditManager = Depends(get_credit_manager),
):
    users = credits.list_all_users()
    return AllUsersResponse(users=[UserBalanceEntry(**u) for u in users])


class AdminAdjustRequest(BaseModel):
    user_id: str
    amount: int
    reason: str = "admin_adjustment"


class AdminAdjustResponse(BaseModel):
    user_id: str
    balance: int
    amount_changed: int


@router.post("/admin/adjust")
def admin_adjust_credits(
    req: AdminAdjustRequest,
    current_user: TokenData = Depends(require_admin),
    credits: CreditManager = Depends(get_credit_manager),
):
    if req.amount == 0:
        raise HTTPException(status_code=422, detail="Amount must not be zero")
    credits.ensure_user_exists(req.user_id)
    if req.amount > 0:
        new_bal = credits.add_credits(req.user_id, req.amount, reason=req.reason)
    else:
        success = credits.deduct_credits(req.user_id, abs(req.amount), reason=req.reason)
        if not success:
            raise HTTPException(status_code=400, detail="Insufficient credits")
        new_bal = credits.get_balance(req.user_id)
    return AdminAdjustResponse(user_id=req.user_id, balance=new_bal, amount_changed=req.amount)


class UsageSummaryResponse(BaseModel):
    user_id: str
    daily: dict
    monthly: dict
    all_time: dict


class SystemUsageResponse(BaseModel):
    total_generations: int
    total_credits_used: int


@router.get("/admin/usage")
def admin_usage(
    current_user: TokenData = Depends(require_admin),
    usage: UsageTracker = Depends(get_usage_tracker),
    user_id: Optional[str] = None,
):
    if user_id:
        daily = usage.get_user_daily_usage(user_id)
        monthly = usage.get_user_monthly_usage(user_id)
        all_time = usage.get_user_all_usage(user_id)
        return UsageSummaryResponse(user_id=user_id, daily=daily, monthly=monthly, all_time=all_time)
    totals = usage.get_system_totals()
    return SystemUsageResponse(**totals)
