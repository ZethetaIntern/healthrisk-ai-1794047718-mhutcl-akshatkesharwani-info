"""Public-data acquisition (UNTESTED: the build machine had no internet). Each function returns None on failure.
MIMIC-IV and HCUP need credentialed access or purchase and are NOT covered; CMS Hospital Compare and CDC WONDER are manual downloads."""
import json
import urllib.request
import pandas as pd


def _get(url, timeout=20):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "healthrisk-ai research"}), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print("download failed:", url[:60], type(e).__name__)
        return None


def fetch_clinicaltrials(condition="lung cancer", page_size=50):
    d = _get(f"https://clinicaltrials.gov/api/v2/studies?query.cond={condition.replace(' ', '+')}&pageSize={page_size}")
    try:
        return pd.json_normalize(d["studies"])
    except Exception:
        return None


def fetch_openfda_events(drug="oxycodone", limit=100):
    d = _get(f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{drug}&limit={limit}")
    try:
        return pd.json_normalize(d["results"])
    except Exception:
        return None


def fetch_who_gho(indicator="WHOSIS_000001"):
    d = _get(f"https://ghoapi.azureedge.net/api/{indicator}")
    try:
        return pd.DataFrame(d["value"])
    except Exception:
        return None
