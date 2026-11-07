"""HealthRisk Lab - quarterly simulation of a $500M health-financial portfolio.

Books: insurance ($150M, 50,000 lives), 15 hospital bonds ($150M), 20 pharma stocks ($100M), credit facility to 10 systems ($100M).
Modes: 1 Actuarial (20 quarters), 2 Hospital Credit (40), 3 Pharma (20), 4 Integrated / Master (40).
1000-point scoring: Insurance 250, Hospital credit 250, Pharma alpha 200, Credit facility 150, Cross-asset 100, Speed and quality 50.
All numbers are teaching assumptions, not calibrated to real markets."""
import numpy as np

# name: (claims_mult, hosp_pd_mult, hosp_exposed_share, pharma_shocks{segment: return}, facility_draw, duration_q, early_signal)
SCENARIOS = {
    "Pandemic Outbreak": (1.35, 1.9, 0.6, {"vaccine": 0.40, "elective": -0.25, "oncology": -0.05, "other": -0.05}, 0.35, 4, True),
    "Seasonal Flu Surge": (1.08, 1.1, 0.4, {"vaccine": 0.08, "elective": -0.03, "oncology": 0.0, "other": 0.0}, 0.05, 1, True),
    "Drug Safety Crisis (black box warning)": (1.10, 1.1, 0.3, {"vaccine": 0.0, "elective": 0.0, "oncology": -0.35, "other": 0.05}, 0.05, 2, False),
    "Regulatory Reimbursement Cut (-3% Medicare)": (0.98, 1.6, 0.7, {"vaccine": 0.0, "elective": -0.05, "oncology": 0.0, "other": 0.0}, 0.15, 4, False),
    "Hospital System Merger": (1.0, 0.9, 0.2, {"vaccine": 0.0, "elective": 0.02, "oncology": 0.0, "other": 0.0}, 0.0, 2, False),
    "Gene Therapy Launch Claims Shock": (1.12, 1.0, 0.2, {"vaccine": 0.0, "elective": 0.0, "oncology": 0.10, "other": 0.0}, 0.0, 4, False),
    "Opioid-Style Litigation Wave": (1.06, 1.25, 0.3, {"vaccine": 0.0, "elective": 0.0, "oncology": 0.0, "other": -0.30}, 0.10, 6, False),
    "Cyberattack on Hospital Network": (1.02, 1.5, 0.3, {"vaccine": 0.0, "elective": -0.02, "oncology": 0.0, "other": 0.0}, 0.25, 1, False),
    "Antimicrobial Resistance Outbreak": (1.15, 1.4, 0.5, {"vaccine": 0.12, "elective": -0.08, "oncology": -0.03, "other": 0.0}, 0.20, 3, True),
    "Drug Pricing Regulation (IRA-style)": (0.97, 1.0, 0.1, {"vaccine": -0.05, "elective": 0.0, "oncology": -0.12, "other": -0.15}, 0.0, 4, False),
    "Rural Hospital Closure Wave": (1.03, 1.8, 0.4, {"vaccine": 0.0, "elective": -0.02, "oncology": 0.0, "other": 0.0}, 0.20, 4, False),
    "Positive Surprise: Cost Trend Falls": (0.94, 0.8, 0.3, {"vaccine": 0.0, "elective": 0.05, "oncology": 0.03, "other": 0.03}, 0.0, 3, False),
}
SEGMENTS = ["vaccine", "elective", "oncology", "other"]
NAIVE = {"reserve_load": 1.0, "premium_margin": 0.02, "hospital_screen": 0.08, "hospital_react": 0.2, "pharma_cap": 0.12, "pharma_tilt": 0.0,
         "facility_monitor": 0.2, "hedge": 0.0, "use_clinical": 0.0, "reaction_q": 2, "documented": 0}
AI = {"reserve_load": 1.12, "premium_margin": 0.03, "hospital_screen": 0.03, "hospital_react": 0.8, "pharma_cap": 0.06, "pharma_tilt": 0.8,
      "facility_monitor": 0.8, "hedge": 0.6, "use_clinical": 0.9, "reaction_q": 0, "documented": 1}
MODE_CATEGORIES = {1: ["insurance", "speed"], 2: ["hospital", "speed"], 3: ["pharma", "speed"], 4: ["insurance", "hospital", "pharma", "facility", "cross", "speed"]}
MAX_PTS = {"insurance": 250, "hospital": 250, "pharma": 200, "facility": 150, "cross": 100, "speed": 50}
MODE_QUARTERS = {1: 20, 2: 40, 3: 20, 4: 40}


def clip01(x):
    return float(np.clip(x, 0, 1))


def sig(x):
    return 1 / (1 + np.exp(-x))


def schedule(n_q, rng, rate=0.35):
    """Random events with a severity; outbreak-type events send an early signal one quarter ahead."""
    ev = []
    for q in range(1, n_q):
        if rng.random() < rate:
            ev.append({"q": q, "name": list(SCENARIOS)[rng.integers(len(SCENARIOS))], "severity": float(rng.uniform(0.7, 1.3))})
    return ev


def run(mode, policy, seed=0, events=None):
    rng = np.random.default_rng(seed)
    nq = MODE_QUARTERS[mode]
    events = events if events is not None else schedule(nq, rng)
    uc, react = policy["use_clinical"], int(policy["reaction_q"])
    prem_q, base_claims = 45.0, 37.0                                    # $M per quarter
    hosp_L = rng.normal(0, 1, 15)
    expo = np.full(15, 10.0)
    medicare_dep = rng.random(15) < 0.5
    seg = rng.choice(len(SEGMENTS), 20, p=[0.15, 0.2, 0.35, 0.3])
    beta = rng.uniform(0.6, 1.3, 20)
    w_ph = np.full(20, 5.0)
    fac_expo = np.full(10, 10.0)
    fac_L = rng.normal(0, 1, 10)
    ins_profit = mlr_ok = res_score = retention = prem_total = 0.0
    hosp_income = hosp_loss = q_good = 0.0
    defaults = deteriorated_cut = deteriorated = 0
    port_ret, bench_ret, hit_beats = [], [], []
    fac_loss = fac_detect = fac_breach = 0.0
    alerts = {}
    for e in events:
        if SCENARIOS[e["name"]][6] and e["q"] >= 1:
            alerts.setdefault(e["q"] - 1, []).append(e)                # early signal the quarter before
    log, pending = [], []
    cut_flag = np.zeros(15, bool)
    for q in range(nq):
        active = [e for e in events if e["q"] <= q < e["q"] + SCENARIOS[e["name"]][5]]
        cm = np.prod([1 + (SCENARIOS[e["name"]][0] - 1) * e["severity"] for e in active]) if active else 1.0
        hm = np.prod([1 + (SCENARIOS[e["name"]][1] - 1) * e["severity"] for e in active]) if active else 1.0
        seen_alert = alerts.get(q, [])
        for e in [x for x in events if x["q"] == q]:
            pending.append((q + react, e))
        responding = [e for (qq, e) in pending if qq <= q and e["q"] + SCENARIOS[e["name"]][5] > q]
        aware = (uc > 0.3 and len(seen_alert) > 0) or len(responding) > 0
        # insurance
        claims = base_claims * cm * np.exp(rng.normal(0, 0.025))
        required_ibnr = 0.35 * claims * (1.0 + 0.3 * (cm - 1))
        known_mult = 1 + (cm - 1) * (1.0 if aware or (q > 0 and uc > 0.3) else 0.3)
        held = 0.35 * base_claims * known_mult * policy["reserve_load"]
        res_score += clip01(1 - max(abs(held / required_ibnr - 1) - 0.05, 0) / 0.25)
        premium = prem_q * (1 + policy["premium_margin"])
        mlr = claims / premium
        mlr_ok += float(0.80 <= mlr <= 0.85)
        ins_profit += premium - claims - 0.08 * premium - 0.004 * max(held - required_ibnr, 0)
        prem_total += premium
        retention += 1 - 0.8 * max(policy["premium_margin"] - 0.04, 0) - 0.1 * float(mlr > 0.95)
        # hospital bonds
        hosp_L = 0.96 * hosp_L + rng.normal(0, 0.25, 15)
        pd_true = sig(-5.0 + 1.1 * hosp_L) * np.where(medicare_dep, hm, 1 + (hm - 1) * 0.4)
        est = pd_true * np.exp(rng.normal(0, 0.7 - 0.55 * uc, 15))
        if aware or q % 4 == 0:
            cut = (est > policy["hospital_screen"]) | (aware & medicare_dep & (hm > 1.2))
            cut_now = cut & (rng.random(15) < policy["hospital_react"])
            cut_flag |= cut_now
            expo = np.where(cut_now, expo * (1 - policy["hospital_react"]), expo)
        det = (hosp_L > 1.2) & (pd_true > 0.02)
        deteriorated += int(det.sum() > 0 and q % 4 == 0)
        deteriorated_cut += int((det & cut_flag).sum() > 0 and q % 4 == 0)
        d = rng.random(15) < pd_true
        loss_h = float((expo * 0.5 * d).sum())
        defaults += int(d.sum())
        expo = np.where(d, 0.0, expo)
        hosp_income += float((expo * (0.012 + 5 * pd_true)).sum()) / 4
        hosp_loss += loss_h
        q_good += float(pd_true[expo > 0].mean() < 0.012) if (expo > 0).any() else 1.0
        # pharma
        mkt = rng.normal(0.02, 0.07)
        shock = np.zeros(20)
        for e in active:
            if e["q"] == q:
                shock += np.array([SCENARIOS[e["name"]][3][SEGMENTS[s]] for s in seg]) * e["severity"]
        tilt_seg = np.zeros(20)
        if aware and uc > 0.3:
            tilt_seg = np.array([1.0 if s == 0 else -0.6 if s == 1 else 0 for s in seg]) * policy["pharma_tilt"] * 0.5
        w = np.clip(w_ph * (1 + tilt_seg), 0, policy["pharma_cap"] * 100)
        ev = rng.random(20) < 0.08
        p_true = rng.uniform(0.35, 0.65, 20)
        p_hat = np.clip(p_true + rng.normal(0, 0.18 * (1 - uc), 20), 0.05, 0.95)
        w_ev = np.where(ev, np.minimum(w * (1 + policy["pharma_tilt"] * (p_hat - 0.5) * 4), policy["pharma_cap"] * 100), w)
        outcome = np.where(rng.random(20) < p_true, 0.45, -0.55)
        r_i = beta * mkt * (1 - 0.5 * policy["hedge"]) + rng.normal(0, 0.05, 20) + shock + ev * outcome
        pr = float((w_ev * r_i).sum() / w_ev.sum() * (w_ev.sum() / 100))
        br = float(np.mean(beta * mkt + shock))
        port_ret.append(pr)
        bench_ret.append(br)
        hit_beats.append(float(pr > br))
        # credit facility
        fac_L = 0.95 * fac_L + rng.normal(0, 0.25, 10)
        p_breach = sig(-4.0 + 1.0 * fac_L) * hm * (1 + 0.5 * (cm - 1))
        breach = rng.random(10) < np.clip(p_breach, 0, 0.9)
        detected = breach & (rng.random(10) < policy["facility_monitor"] * (0.5 + 0.5 * uc))
        fac_loss += float((fac_expo * 0.35 * breach * (1 - detected * 0.7)).sum())
        fac_breach += breach.sum()
        fac_detect += detected.sum()
        log.append({"q": q, "events": [e["name"] for e in active], "claims_mult": round(cm, 2), "mlr": round(mlr, 3), "hosp_defaults": int(d.sum()), "pharma_return": round(pr, 4)})
    pr_a = np.array(port_ret)
    sharpe = pr_a.mean() / (pr_a.std() + 1e-9) * 2
    eq = np.cumprod(1 + pr_a)
    dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
    ins = {"mlr_compliance": 60 * mlr_ok / nq, "reserve_adequacy": 70 * res_score / nq, "profitability": 70 * clip01(ins_profit / prem_total / 0.10), "member_satisfaction": 50 * clip01(retention / nq)}
    hosp = {"default_avoidance": 80 * clip01(1 - hosp_loss / (0.05 * 150)), "spread_capture": 60 * clip01(hosp_income / (0.012 * 150 * nq / 4 * 1.4)),
            "early_warning": 70 * clip01(deteriorated_cut / max(deteriorated, 1)) if deteriorated else 70 * 0.7, "portfolio_quality": 40 * q_good / nq}
    ph = {"absolute_return": 50 * clip01(pr_a.mean() * 4 / 0.12), "sharpe": 60 * clip01(sharpe / 1.0), "hit_rate": 40 * clip01((np.mean(hit_beats) - 0.4) / 0.25), "drawdown_control": 50 * clip01(1 - abs(dd) / 0.25)}
    fac = {"loss_avoidance": 70 * clip01(1 - fac_loss / (0.06 * 100 * nq / 4 + 1e-9) * 2), "covenant_monitoring": 50 * clip01(fac_detect / max(fac_breach, 1)), "exposure_mgmt": 30 * clip01(0.5 + 0.5 * policy["facility_monitor"])}
    cross = {"correlation_mgmt": 40 * clip01(0.4 + 0.6 * policy["hedge"] + 0.2 * (1 - policy["pharma_cap"] * 5)), "scenario_response": 30 * clip01(1 - react / 3 + 0.4 * uc), "hedging": 30 * clip01(policy["hedge"] * 1.2)}
    speed = {"timeliness": 30 * clip01(1 - react / 3), "documentation": 20 * float(policy["documented"])}
    cats = {"insurance": ins, "hospital": hosp, "pharma": ph, "facility": fac, "cross": cross, "speed": speed}
    use = MODE_CATEGORIES[mode]
    raw = {c: sum(cats[c].values()) for c in use}
    total = sum(raw.values()) / sum(MAX_PTS[c] for c in use) * 1000
    rating = "HealthRisk Master" if total >= 900 else "HealthRisk Expert" if total >= 800 else "HealthRisk Specialist" if total >= 650 else "HealthRisk Analyst" if total >= 500 else "Trainee"
    return {"mode": mode, "quarters": nq, "total": round(total), "rating": rating, "category_points": {c: round(v, 1) for c, v in raw.items()},
            "max_points": {c: MAX_PTS[c] for c in use}, "detail": {c: {k: round(v, 1) for k, v in cats[c].items()} for c in use},
            "hospital_defaults": defaults, "events": events, "log": log, "sharpe": round(float(sharpe), 2), "max_drawdown_pct": round(dd * 100, 1), "mlr_in_band_share": round(mlr_ok / nq, 2)}


def ai_opponent(mode, seed=0, events=None):
    return run(mode, AI, seed, events)


def compare(mode, policy, seed=0):
    ev = schedule(MODE_QUARTERS[mode], np.random.default_rng(seed))
    me, ai = run(mode, policy, seed, ev), ai_opponent(mode, seed, ev)
    gap = {c: round(ai["category_points"][c] - me["category_points"][c], 1) for c in me["category_points"]}
    return {"player": me["total"], "ai": ai["total"], "gap_by_category": gap, "biggest_gaps": sorted(gap, key=gap.get, reverse=True)[:2]}


def replay_pandemic(policy, seed=0):
    """Historical-style replay: outbreak detected in quarter 1 (early signal), pandemic declared in quarter 2."""
    return run(4, policy, seed, [{"q": 2, "name": "Pandemic Outbreak", "severity": 1.0}])


def to_plain(x):
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def play_cli(mode=4):
    """Terminal version:  python -m simulation.lab"""
    pol = dict(NAIVE)
    for k in pol:
        v = input(f"{k} [{pol[k]}]: ").strip()
        if v:
            pol[k] = float(v)
    print(compare(mode, pol, 1))


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        print({m: (run(m, NAIVE, 1)["total"], ai_opponent(m, 1)["total"]) for m in (1, 2, 3, 4)})
    else:
        play_cli(int(input("Mode 1-4: ")))
