# scripts/brent_options_iv_utils.py

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional


# -------------------------
# Normal distribution utils
# -------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


# -------------------------
# Day count
# -------------------------

def year_fraction(d0: date, d1: date, basis: int = 365) -> float:
    return max((d1 - d0).days, 0) / float(basis)


# -------------------------
# Black-76 core
# -------------------------

def black76_d1_d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return d1, d2


def black76_price(F: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """
    Black-76 futures option price.
    r is continuously compounded.
    """
    if F <= 0 or K <= 0:
        return float("nan")

    if T <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return intrinsic

    disc = math.exp(-r * T)

    if sigma <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return disc * intrinsic

    d1, d2 = black76_d1_d2(F, K, T, sigma)

    if is_call:
        return disc * (F * norm_cdf(d1) - K * norm_cdf(d2))
    else:
        return disc * (K * norm_cdf(-d2) - F * norm_cdf(-d1))


def black76_delta(F: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """
    Delta w.r.t futures price (discounted).
    """
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0

    disc = math.exp(-r * T)
    d1, _ = black76_d1_d2(F, K, T, sigma)

    if is_call:
        return disc * norm_cdf(d1)
    else:
        return -disc * norm_cdf(-d1)


@dataclass(frozen=True)
class Black76Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float


def black76_greeks(F: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> Black76Greeks:
    price = black76_price(F, K, T, r, sigma, is_call)

    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return Black76Greeks(price=price, delta=0.0, gamma=0.0, vega=0.0, theta=0.0)

    disc = math.exp(-r * T)
    vol_sqrt = sigma * math.sqrt(T)
    d1, _ = black76_d1_d2(F, K, T, sigma)
    pdf_d1 = norm_pdf(d1)

    # delta
    delta = black76_delta(F, K, T, r, sigma, is_call)

    # gamma
    gamma = disc * pdf_d1 / (F * vol_sqrt)

    # vega (per 1.00 vol)
    vega = disc * F * pdf_d1 * math.sqrt(T)

    # theta (continuous approximation)
    theta = -disc * (F * pdf_d1 * sigma) / (2.0 * math.sqrt(T)) + r * price

    return Black76Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)


# -------------------------
# Implied volatility (bisection)
# -------------------------

def implied_vol_black76(
    price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    *,
    vol_low: float = 1e-6,
    vol_high: float = 3.0,
    tol: float = 1e-8,
    max_iter: int = 120,
) -> Optional[float]:
    """
    Robust bisection IV. Returns None if cannot solve.
    """
    if price is None or F <= 0 or K <= 0 or T <= 0:
        return None
    if price <= 0:
        return None

    disc = math.exp(-r * T)
    intrinsic = disc * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    upper = disc * (F if is_call else K)

    if price < intrinsic - 1e-10 or price > upper + 1e-10:
        return None

    lo, hi = vol_low, vol_high
    f_lo = black76_price(F, K, T, r, lo, is_call) - price
    f_hi = black76_price(F, K, T, r, hi, is_call) - price

    # Expand hi if needed
    expand = 0
    while f_hi < 0 and expand < 12:
        hi *= 1.7
        f_hi = black76_price(F, K, T, r, hi, is_call) - price
        expand += 1

    if f_lo > 0 or f_hi < 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = black76_price(F, K, T, r, mid, is_call) - price

        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid

        if f_mid > 0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


# -------------------------
# Strike-from-delta inversion
# -------------------------

def strike_from_delta_black76(
    target_delta: float,
    F: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool,
    *,
    K_low: float,
    K_high: float,
    tol: float = 1e-6,
    max_iter: int = 120,
) -> Optional[float]:
    """
    Solve for strike K such that Black-76 delta = target_delta.
    Delta decreases monotonically with strike K.
    """
    if F <= 0 or T <= 0 or sigma <= 0 or K_low <= 0 or K_high <= 0:
        return None
    if K_low >= K_high:
        return None

    d_lo = black76_delta(F, K_low, T, r, sigma, is_call)
    d_hi = black76_delta(F, K_high, T, r, sigma, is_call)

    d_max = max(d_lo, d_hi)
    d_min = min(d_lo, d_hi)
    if not (d_min - 1e-12 <= target_delta <= d_max + 1e-12):
        return None

    lo, hi = K_low, K_high

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        d_mid = black76_delta(F, mid, T, r, sigma, is_call)

        if abs(d_mid - target_delta) < tol or (hi - lo) < tol:
            return mid

        if d_mid > target_delta:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)

