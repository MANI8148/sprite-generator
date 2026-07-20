from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from backend.modules.auth import get_current_user, OptionalAuth, TokenData
from backend.modules.billing import CreditManager, get_credit_manager

router = APIRouter(prefix="/billing", tags=["billing"])


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
):
    credits.ensure_user_exists(current_user.user_id)
    txs = credits.get_transactions(current_user.user_id)
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
