"""HealthRisk Lab demo: AI vs naive over 20 seeds, scenario catalogue, pandemic replay. Writes results/simulation.json."""
import json
import numpy as np
from simulation import lab as L

out = {"scenarios": len(L.SCENARIOS), "catalogue": {k: {"claims_multiplier": v[0], "hospital_pd_multiplier": v[1], "duration_q": v[5], "early_signal": v[6]} for k, v in L.SCENARIOS.items()}}
for m in (1, 2, 3, 4):
    a = [L.run(m, L.NAIVE, s)["total"] for s in range(20)]
    b = [L.ai_opponent(m, s)["total"] for s in range(20)]
    out[f"mode{m}"] = {"quarters": L.MODE_QUARTERS[m], "naive_mean": round(float(np.mean(a))), "ai_mean": round(float(np.mean(b))), "ai_beats_naive_share": round(float(np.mean(np.array(b) > np.array(a))), 2)}
ten_year = L.to_plain(L.run(4, L.AI, 7))
out["complete_10_year_run_master_mode_ai"] = {k: v for k, v in ten_year.items() if k not in ("log", "events")}
out["ten_year_event_list"] = [f"Q{e['q']}: {e['name']} (severity {e['severity']:.2f})" for e in ten_year["events"]]
rp, rn = L.to_plain(L.replay_pandemic(L.AI)), L.to_plain(L.replay_pandemic(L.NAIVE))
out["pandemic_replay"] = {"ai": {"total": rp["total"], "points": rp["category_points"], "hospital_defaults": rp["hospital_defaults"]}, "naive": {"total": rn["total"], "points": rn["category_points"], "hospital_defaults": rn["hospital_defaults"]}}
out["player_vs_ai_example"] = L.compare(4, L.NAIVE, 3)
json.dump(out, open("results/simulation.json", "w"), indent=1)
print({k: out[k] for k in ("mode1", "mode2", "mode3", "mode4")})
