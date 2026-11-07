# HealthRisk Lab - game design document

**Educational objective:** make learners combine clinical intelligence (outbreak signals, trial data, hospital quality) with insurance, credit and equity decisions under time pressure.

**Portfolio:** $500M = insurance book $150M (50,000 lives), 15 hospital bonds $150M, 20 pharma stocks $100M, credit facility to 10 systems $100M. Quarterly turns.

**Modes:** 1 Actuarial (20 quarters), 2 Hospital Credit (40), 3 Pharma (20), 4 Integrated / Master (40). Single-mode scores are rescaled to 1000.

**Player decisions:** reserve load, premium margin, hospital PD screen and reaction strength, pharma position cap and tilt, facility monitoring, hedge, use of clinical data, reaction delay, documentation.

## Scoring (1000 points)
| Category | Points | Components |
|---|---|---|
| Insurance | 250 | MLR in 80-85% band 60, reserve adequacy 70, profitability 70, member retention 50 |
| Hospital credit | 250 | default avoidance 80, spread capture 60, early warning 70, portfolio quality 40 |
| Pharma alpha | 200 | absolute return 50, Sharpe 60, hit rate 40, drawdown control 50 |
| Credit facility | 150 | loss avoidance 70, covenant monitoring 50, exposure management 30 |
| Cross-asset risk | 100 | correlation management 40, scenario response 30, hedging 30 |
| Speed and quality | 50 | timeliness 30, documentation 20 |

Ratings: Trainee < 500, Analyst >= 500, Specialist >= 650, Expert >= 800, Master >= 900.

## Scenarios
| Scenario | Claims multiplier | Hospital PD multiplier | Duration (quarters) | Early signal |
|---|---|---|---|---|
| Pandemic Outbreak | 1.35 | 1.9 | 4 | yes |
| Seasonal Flu Surge | 1.08 | 1.1 | 1 | yes |
| Drug Safety Crisis (black box warning) | 1.1 | 1.1 | 2 | no |
| Regulatory Reimbursement Cut (-3% Medicare) | 0.98 | 1.6 | 4 | no |
| Hospital System Merger | 1.0 | 0.9 | 2 | no |
| Gene Therapy Launch Claims Shock | 1.12 | 1.0 | 4 | no |
| Opioid-Style Litigation Wave | 1.06 | 1.25 | 6 | no |
| Cyberattack on Hospital Network | 1.02 | 1.5 | 1 | no |
| Antimicrobial Resistance Outbreak | 1.15 | 1.4 | 3 | yes |
| Drug Pricing Regulation (IRA-style) | 0.97 | 1.0 | 4 | no |
| Rural Hospital Closure Wave | 1.03 | 1.8 | 4 | no |
| Positive Surprise: Cost Trend Falls | 0.94 | 0.8 | 3 | no |

Pandemic worked example (replay): quarter 2 declaration; the AI scores 574 against 370 for the naive policy. Points by category (AI): {'insurance': 193.0, 'hospital': 146.0, 'pharma': 34.9, 'facility': 62.0, 'cross': 87.6, 'speed': 50.0}.

**AI opponent logic:** uses clinical data for sharper risk estimates (noise falls with `use_clinical`), reacts in the same quarter, holds a larger reserve, cuts weak hospital exposure and tilts pharma toward event beneficiaries. It is a rule set, not a learned policy.
