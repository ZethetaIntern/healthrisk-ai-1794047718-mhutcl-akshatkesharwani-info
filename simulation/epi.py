"""SEIR epidemic model and its link to insurance claims (used for the pandemic scenario and the early-warning module)."""
import numpy as np


def seir(r0=2.5, incubation_days=5.5, infectious_days=6.0, n=1_000_000, i0=10, days=365, dt=0.25):
    """dS/dt = -b S I/N, dE/dt = b S I/N - sigma E, dI/dt = sigma E - gamma I, dR/dt = gamma I   with b = R0 x gamma."""
    sigma, gamma = 1 / incubation_days, 1 / infectious_days
    beta = r0 * gamma
    S, E, I, R = n - i0, 0.0, float(i0), 0.0
    out = []
    for step in range(int(days / dt)):
        new_e = beta * S * I / n * dt
        new_i, new_r = sigma * E * dt, gamma * I * dt
        S, E, I, R = S - new_e, E + new_e - new_i, I + new_i - new_r, R + new_r
        if step % int(1 / dt) == 0:
            out.append((S, E, I, R))
    arr = np.array(out)
    return {"peak_infected": float(arr[:, 2].max()), "peak_day": int(arr[:, 2].argmax()), "attack_rate": float(arr[-1, 3] / n), "curve": arr}


def claims_impact(members=50_000, r0=2.5, treat_share=0.035, icu_cost=75_000, hosp_share_of_cases=0.15, mild_cost=1_500):
    """Rough claims for an insured book; the share needing treatment is capped to the brief's 2-5% range. Unit costs are assumptions."""
    ar = seir(r0=r0)["attack_rate"]
    treated = members * min(max(ar * treat_share / 0.35, 0.02), 0.05)
    severe = treated * hosp_share_of_cases
    return {"attack_rate": round(ar, 3), "members_treated": round(treated), "severe_cases": round(severe), "incremental_claims_musd": round((severe * icu_cost + (treated - severe) * mild_cost) / 1e6, 1)}


def doubling_time(r0, incubation_days=5.5, infectious_days=6.0):
    """Early exponential growth: the dominant eigenvalue of the SEIR linearisation gives the growth rate."""
    sigma, gamma = 1 / incubation_days, 1 / infectious_days
    lam = np.linalg.eigvals(np.array([[-sigma, r0 * gamma], [sigma, -gamma]])).real.max()
    return float(np.log(2) / lam) if lam > 0 else float("inf")
