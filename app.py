"""
Depot Infrastructure Tool
==============================================================
Primary outputs:
    1. Optimal charger selection per vehicle (smallest standard charger that meets need)
    2. Charger fleet count (shared, not 1:1) based on scheduling overlaps
    3. Peak demand (managed vs unmanaged)
    4. Aggregated load profile (15-min resolution)
    5. Site electrical upgrade path: MSB, transformer, HV connection

Key design:
    - Fleet sheet has max_ac_kw / max_dc_kw (vehicle capability), NOT charger assignment
    - Consumption (kWh/km) set per vehicle type as sidebar assumption
    - Charger selected from standard commercial sizes
    - Charger sharing determined by scheduling engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from copy import deepcopy

# ============================================================================
# STANDARD CHARGER CATALOGUE
# ============================================================================

AC_CHARGER_SIZES_KW = [7, 11, 22]       # single-phase 7kW, 3-phase 11/22kW
DC_CHARGER_SIZES_KW = [30, 50, 60, 120, 150, 300]
ALL_CHARGER_SIZES = (
    [("AC", kw) for kw in AC_CHARGER_SIZES_KW] +
    [("DC", kw) for kw in DC_CHARGER_SIZES_KW]
)

# ============================================================================
# DEFAULT CONSUMPTION RATES (kWh/km) — per vehicle type
# ============================================================================

DEFAULT_CONSUMPTION = {
    "bus":   {"value": 1.20, "range": "0.9–1.5", "note": "12m urban bus, HVAC dependent"},
    "van":   {"value": 0.22, "range": "0.18–0.30", "note": "LCV 2–3.5t (e.g., eVito, eTransit)"},
    "truck": {"value": 0.65, "range": "0.45–1.0", "note": "Medium rigid 4.5–12t GVM"},
    "car":   {"value": 0.16, "range": "0.13–0.22", "note": "Passenger vehicle / sedan"},
}

# ============================================================================
# ASSUMPTIONS
# ============================================================================

ASSUMPTIONS = {
    "charger_efficiency_ac": {
        "value": 0.92, "unit": "",
        "justification": "AC onboard charger wall-to-battery efficiency (IEC 61851). "
                         "Range 0.88–0.95."
    },
    "charger_efficiency_dc": {
        "value": 0.95, "unit": "",
        "justification": "DC EVSE wall-to-battery efficiency. Range 0.92–0.97. "
                         "ABB/Kempower typical ~0.95."
    },
    "max_soc_limit": {
        "value": 0.90, "unit": "fraction",
        "info": "Maximum state-of-charge target when charging.",
        "impact": "Lower values reduce battery degradation but require more frequent "
                  "charging or higher-power chargers. At 90%, ~10% of usable capacity "
                  "is unused — this is the industry standard for fleet longevity.",
        "justification": "Charging above 90% increases cell stress disproportionately."
    },
    "min_soc_floor": {
        "value": 0.10, "unit": "fraction",
        "justification": "Never discharge below 10%. Most OEMs void warranty below this."
    },
    "low_km_threshold_fraction": {
        "value": 0.30, "unit": "fraction",
        "info": "If a vehicle's daily energy need is below this fraction of its usable "
                "battery capacity, it doesn't need to charge every day.",
        "impact": "Lower values → fewer vehicles skip charge days → more charger demand "
                  "but higher operational safety margin. Higher values → more vehicles "
                  "skip days → lower charger count needed but tighter SoC margins.",
        "justification": "30% threshold means a vehicle using <30% of its battery daily "
                         "can safely charge every 2–3 days."
    },
    "low_km_charge_interval_days": {
        "value": 3, "unit": "days",
        "info": "How often low-utilisation vehicles charge (if below low-km threshold).",
        "impact": "Longer intervals → fewer chargers needed, but vehicles spend more "
                  "time at lower SoC. Safety override forces charging if SoC would "
                  "breach the floor before next scheduled charge.",
        "justification": "Every 3 days balances charger utilisation with operational readiness."
    },
    "time_resolution_minutes": {
        "value": 15, "unit": "minutes",
        "justification": "Aligns with NEM metering and demand measurement periods."
    },
    "ambient_hvac_factor": {
        "value": 1.10, "unit": "multiplier",
        "info": "Multiplier on base consumption to account for cabin HVAC load.",
        "impact": "Higher values → vehicles need more energy per trip → larger chargers "
                  "or longer charge windows. 1.10 = 10% overhead (mild climate). "
                  "Use 1.15–1.20 for QLD/NT summer conditions.",
        "justification": "10% conservative for temperate Australian conditions."
    },
    "diversity_factor": {
        "value": 0.80, "unit": "fraction",
        "info": "What fraction of total installed charger capacity draws full power "
                "simultaneously at peak.",
        "impact": "Directly scales the EV load used for MSB and transformer sizing. "
                  "0.80 = assume 80% of chargers at full power simultaneously. "
                  "Lower diversity → smaller MSB/transformer needed. "
                  "AS/NZS 3000 suggests 0.7–0.9 for EV installations.",
        "justification": "0.80 is a moderate assumption per AS/NZS 3000 guidance."
    },
    "charger_utilisation_target": {
        "value": 0.70, "unit": "fraction",
        "info": "Target utilisation when sizing the charger fleet. A charger operating "
                "at 70% of its theoretical maximum throughput over the parking window.",
        "impact": "Lower targets → more chargers purchased (more headroom for late "
                  "arrivals, maintenance, variability). Higher targets → fewer chargers "
                  "but tighter operations. Below 60% is wasteful; above 85% is risky.",
        "justification": "70% is standard fleet practice — allows ~30% margin."
    },
    "charger_sizing_margin": {
        "value": 1.15, "unit": "multiplier",
        "info": "Safety margin applied when selecting charger size. The minimum required "
                "power is multiplied by this factor before selecting the next standard size.",
        "impact": "1.15 = 15% headroom. Covers real-world losses not in the model "
                  "(cable voltage drop, thermal derating, connection overhead).",
        "justification": "15% margin is conservative industry practice."
    },
    "voltage_three_phase": {"value": 415, "unit": "V", "justification": "AS/NZS 3000:2018."},
    "power_factor": {"value": 0.95, "unit": "", "justification": "Modern chargers with active PFC."},
    "sqrt3": {"value": 1.732, "unit": "", "justification": "√3."},
    "msb_ratings_amps": {
        "value": [100, 200, 315, 400, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150],
        "unit": "A @ 415V 3φ",
        "justification": "Standard MSB frame sizes per AS/NZS 3000."
    },
    "transformer_sizes_kva": {
        "value": [100, 200, 315, 500, 750, 1000, 1500, 2000, 2500],
        "unit": "kVA",
        "justification": "Standard AU distribution transformer ratings. "
                         "≤500=pole-mount, 500–2500=pad-mount/kiosk. "
                         ">1000 usually requires HV connection."
    },
}


def amps_to_kw(a):
    return (ASSUMPTIONS["sqrt3"]["value"] * ASSUMPTIONS["voltage_three_phase"]["value"] *
            a * ASSUMPTIONS["power_factor"]["value"]) / 1000

def kw_to_amps(kw):
    return (kw * 1000) / (ASSUMPTIONS["sqrt3"]["value"] *
            ASSUMPTIONS["voltage_three_phase"]["value"] * ASSUMPTIONS["power_factor"]["value"])

def kw_to_kva(kw):
    return kw / ASSUMPTIONS["power_factor"]["value"]

MSB_TIERS = [(a, round(amps_to_kw(a), 1)) for a in ASSUMPTIONS["msb_ratings_amps"]["value"]]
TRAFO_TIERS = [(kva, round(kva * ASSUMPTIONS["power_factor"]["value"], 1))
               for kva in ASSUMPTIONS["transformer_sizes_kva"]["value"]]


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Vehicle:
    id: str
    vehicle_type: str
    battery_capacity_kwh: float
    usable_capacity_fraction: float
    daily_km: float
    arrival_time: str
    departure_time: str
    max_ac_kw: float
    max_dc_kw: float
    consumption_kwh_per_km: float = 0.0
    assigned_charger_type: str = ""
    assigned_charger_kw: float = 0.0
    charge_today: bool = True
    priority: float = 0.0

    @property
    def usable_capacity_kwh(self):
        return self.battery_capacity_kwh * self.usable_capacity_fraction

    @property
    def charger_efficiency(self):
        if self.assigned_charger_type == "DC":
            return ASSUMPTIONS["charger_efficiency_dc"]["value"]
        return ASSUMPTIONS["charger_efficiency_ac"]["value"]

    @property
    def daily_energy_need_kwh(self):
        raw = self.daily_km * self.consumption_kwh_per_km * ASSUMPTIONS["ambient_hvac_factor"]["value"]
        return raw / self.charger_efficiency

    @property
    def daily_battery_energy_kwh(self):
        return self.daily_km * self.consumption_kwh_per_km * ASSUMPTIONS["ambient_hvac_factor"]["value"]

    @property
    def needs_daily_charge(self):
        frac = self.daily_battery_energy_kwh / self.usable_capacity_kwh
        return frac >= ASSUMPTIONS["low_km_threshold_fraction"]["value"]

    @property
    def parking_hours(self):
        a = _t2m(self.arrival_time); d = _t2m(self.departure_time)
        return ((1440 - a + d) if d <= a else (d - a)) / 60.0

    @property
    def min_charger_power_kw(self):
        if self.parking_hours <= 0: return 999
        margin = ASSUMPTIONS["charger_sizing_margin"]["value"]
        return (self.daily_energy_need_kwh / self.parking_hours) * margin


@dataclass
class TariffPeriod:
    name: str; start_time: str; end_time: str
    rate_dollars_per_kwh: float; color: str = "#000000"

@dataclass
class SiteConfig:
    existing_capacity_kw: float
    existing_base_load_kw: float
    demand_charge_per_kw: float = 0.0


# ============================================================================
# HELPERS
# ============================================================================

def _t2m(t): h, m = map(int, t.split(":")); return h * 60 + m
def t2s(t, r=15): return _t2m(t) // r
def s2t(s, r=15): return f"{(s*r//60):02d}:{(s*r%60):02d}"

def build_tariff_array(tps, r=15):
    n = (24*60)//r; rates = np.zeros(n)
    for tp in tps:
        s, e = t2s(tp.start_time, r), t2s(tp.end_time, r)
        if e <= s: rates[s:] = tp.rate_dollars_per_kwh; rates[:e] = tp.rate_dollars_per_kwh
        else: rates[s:e] = tp.rate_dollars_per_kwh
    return rates

def build_tariff_colors(tps, r=15):
    n = (24*60)//r; c = ["#cccccc"]*n
    for tp in tps:
        s, e = t2s(tp.start_time, r), t2s(tp.end_time, r)
        if e <= s:
            for i in range(s, n): c[i] = tp.color
            for i in range(0, e): c[i] = tp.color
        else:
            for i in range(s, e): c[i] = tp.color
    return c

def hex_rgba(h, a=1.0):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


# ============================================================================
# CHARGER SELECTION ENGINE
# ============================================================================

def select_optimal_charger(vehicle):
    min_power = vehicle.min_charger_power_kw
    for kw in AC_CHARGER_SIZES_KW:
        if kw >= min_power and kw <= vehicle.max_ac_kw:
            return ("AC", kw, f"AC {kw}kW meets {min_power:.1f}kW need within {vehicle.parking_hours:.1f}h window")
    for kw in DC_CHARGER_SIZES_KW:
        if kw >= min_power and kw <= vehicle.max_dc_kw:
            return ("DC", kw, f"DC {kw}kW needed — AC insufficient ({min_power:.1f}kW required, "
                    f"max AC {vehicle.max_ac_kw}kW)")
    return ("DC", vehicle.max_dc_kw,
            f"⚠️ No standard charger sufficient. Need {min_power:.1f}kW but max DC={vehicle.max_dc_kw}kW. "
            f"Vehicle may not fully charge in {vehicle.parking_hours:.1f}h window.")

def assign_chargers_to_fleet(vehicles):
    assignments = []; warnings = []
    for v in vehicles:
        ctype, ckw, rationale = select_optimal_charger(v)
        v.assigned_charger_type = ctype; v.assigned_charger_kw = ckw
        assignments.append({
            "Vehicle": v.id, "Type": v.vehicle_type, "Daily km": v.daily_km,
            "Energy Need (kWh)": round(v.daily_energy_need_kwh, 1),
            "Park (hrs)": round(v.parking_hours, 1),
            "Min Power (kW)": round(v.min_charger_power_kw, 1),
            "Max AC (kW)": v.max_ac_kw, "Max DC (kW)": v.max_dc_kw,
            "Selected": f"{ctype} {ckw}kW", "Rationale": rationale,
        })
        if "⚠️" in rationale:
            warnings.append({"Vehicle": v.id, "Issue": rationale})
    return {"assignments": assignments, "warnings": warnings}


# ============================================================================
# CHARGER FLEET SIZING (shared chargers)
# ============================================================================

def size_charger_fleet(vehicles):
    util = ASSUMPTIONS["charger_utilisation_target"]["value"]
    div = ASSUMPTIONS["diversity_factor"]["value"]
    res = ASSUMPTIONS["time_resolution_minutes"]["value"]
    n_slots = (24*60)//res
    groups = {}
    for v in vehicles:
        key = f"{v.assigned_charger_type}_{int(v.assigned_charger_kw)}"
        groups.setdefault(key, []).append(v)
    recs = {}; total_installed = 0
    for key, vlist in groups.items():
        ctype, ckw_str = key.split("_", 1); ckw = float(ckw_str)
        total_energy = sum(v.daily_energy_need_kwh for v in vlist)
        avg_park = float(np.mean([v.parking_hours for v in vlist]))
        min_park = float(min(v.parking_hours for v in vlist))
        if avg_park > 0 and ckw > 0:
            n_energy = math.ceil((total_energy / ckw) / (avg_park * util))
        else:
            n_energy = len(vlist)
        concurrent = np.zeros(n_slots, dtype=int)
        for v in vlist:
            a, d = t2s(v.arrival_time, res), t2s(v.departure_time, res)
            if d <= a: concurrent[a:] += 1; concurrent[:d] += 1
            else: concurrent[a:d] += 1
        peak_conc = int(np.max(concurrent))
        n_rec = min(max(n_energy, 1), len(vlist)); gpower = n_rec * ckw
        recs[key] = {
            "label": f"{ctype} {int(ckw)}kW", "charger_type": ctype, "charger_kw": ckw,
            "vehicle_count": len(vlist), "vehicle_ids": [v.id for v in vlist],
            "total_daily_energy_kwh": round(total_energy, 1),
            "avg_parking_hours": round(avg_park, 1), "min_parking_hours": round(min_park, 1),
            "peak_concurrent": peak_conc, "chargers_recommended": n_rec,
            "total_power_kw": round(gpower, 1), "diversified_power_kw": round(gpower * div, 1),
            "sizing_method": "energy" if n_energy > peak_conc else "concurrency",
            "ratio": f"{len(vlist)}:{n_rec}",
        }
        total_installed += gpower
    return {
        "by_group": recs, "total_installed_kw": round(total_installed, 1),
        "total_diversified_kw": round(total_installed * div, 1),
        "diversity_factor": div,
        "total_chargers": sum(r["chargers_recommended"] for r in recs.values()),
    }


# ============================================================================
# SITE ELECTRICAL SIZING
# ============================================================================

def size_site_electrical(charger_sizing, base_load_kw, existing_capacity_kw):
    ev_kw = charger_sizing["total_diversified_kw"]
    total_kw = base_load_kw + ev_kw; total_kva = kw_to_kva(total_kw); total_amps = kw_to_amps(total_kw)
    msb_a = msb_kw = None
    for a, kw in MSB_TIERS:
        if kw >= total_kw: msb_a = a; msb_kw = kw; break
    if msb_a is None: msb_a = f">{MSB_TIERS[-1][0]}A (bus-bar)"; msb_kw = total_kw
    trafo_kva = trafo_kw = None
    for kva, kw in TRAFO_TIERS:
        if kva >= total_kva: trafo_kva = kva; trafo_kw = kw; break
    if trafo_kva is None: trafo_kva = f">{TRAFO_TIERS[-1][0]}kVA (multiple/HV)"; trafo_kw = total_kw
    hv = isinstance(trafo_kva, (int, float)) and trafo_kva > 1000
    hv_note = ""
    if hv:
        hv_note = ("Demand exceeds 1000 kVA — typically requires dedicated 11kV/22kV HV "
                   "connection and customer-owned substation. Budget $150–400k, lead time 6–18 months.")
    elif isinstance(trafo_kva, (int, float)) and trafo_kva > 500:
        hv_note = ("Demand 500–1000 kVA — may need pad-mount kiosk transformer. Lead time 3–6 months.")
    upgrade = total_kw > existing_capacity_kw; path = []
    if upgrade:
        ex_a = round(kw_to_amps(existing_capacity_kw))
        path = [
            {"Component": "Main Switchboard (MSB)", "Current": f"{ex_a}A ({existing_capacity_kw:.0f}kW)",
             "Required": f"{msb_a}A ({msb_kw:.0f}kW)" if isinstance(msb_a, int) else str(msb_a),
             "Action": "Replace MSB frame", "Lead Time": "4–8 wks", "Est. Cost": "$15–40k"},
            {"Component": "Transformer", "Current": "Existing (verify w/ DNSP)",
             "Required": f"{trafo_kva}kVA ({trafo_kw:.0f}kW)" if isinstance(trafo_kva, int) else str(trafo_kva),
             "Action": "DNSP application" + (" + HV substation" if hv else ""),
             "Lead Time": "3–18 months", "Est. Cost": "$30–400k"},
            {"Component": "EV Sub-board", "Current": "N/A",
             "Required": f"Dedicated board for {ev_kw:.0f}kW EV",
             "Action": "Isolates EV load", "Lead Time": "2–4 wks", "Est. Cost": "$5–15k"},
            {"Component": "Cabling", "Current": "Existing",
             "Required": f"Mains for {total_amps:.0f}A",
             "Action": "Upgrade if undersized", "Lead Time": "2–6 wks", "Est. Cost": "$10–50k"},
            {"Component": "Protection", "Current": "Existing",
             "Required": "Type B RCDs for DC (AS/NZS 3000 cl 2.6.3.2)",
             "Action": "Type B RCDs per DC circuit", "Lead Time": "1–2 wks", "Est. Cost": "$2–5k"},
        ]
    return {
        "total_demand_kw": round(total_kw, 1), "total_demand_kva": round(total_kva, 1),
        "total_demand_amps": round(total_amps, 1),
        "base_load_kw": round(base_load_kw, 1), "ev_load_kw": round(ev_kw, 1),
        "ev_undiversified_kw": charger_sizing["total_installed_kw"],
        "msb_amps": msb_a, "msb_kw": msb_kw if isinstance(msb_kw, (int, float)) else total_kw,
        "trafo_kva": trafo_kva, "trafo_kw": trafo_kw if isinstance(trafo_kw, (int, float)) else total_kw,
        "hv_required": hv, "hv_note": hv_note,
        "existing_kw": existing_capacity_kw, "upgrade": upgrade,
        "headroom_kw": round(existing_capacity_kw - total_kw, 1), "path": path,
    }


# ============================================================================
# SCHEDULING ENGINE
# ============================================================================

def schedule_fleet(vehicles, tariff_periods, site, charger_fleet,
                   day_of_week=0, res=15, energy_overrides=None):
    n = (24*60)//res; rates = build_tariff_array(tariff_periods, res)
    agg = np.full(n, site.existing_base_load_kw)
    cu = {k: np.zeros(n, dtype=int) for k in charger_fleet}
    interval = ASSUMPTIONS["low_km_charge_interval_days"]["value"]
    for v in vehicles:
        v.charge_today = v.needs_daily_charge or (day_of_week % interval == 0)
    charging = [v for v in vehicles if v.charge_today]
    for v in charging: v.priority = _priority(v, rates, res)
    charging.sort(key=lambda v: v.priority, reverse=True)
    sched = {}; costs = {}; unsched = []
    for v in charging:
        eneed = (energy_overrides or {}).get(v.id, v.daily_energy_need_kwh)
        if eneed <= 0: sched[v.id] = np.zeros(n); costs[v.id] = 0.0; continue
        a_s, d_s = t2s(v.arrival_time, res), t2s(v.departure_time, res)
        valid = list(range(a_s, n)) + list(range(0, d_s)) if d_s <= a_s else list(range(a_s, d_s))
        if not valid:
            unsched.append({"Vehicle": v.id, "Issue": "No parking window"})
            sched[v.id] = np.zeros(n); costs[v.id] = 0.0; continue
        gkey = f"{v.assigned_charger_type}_{int(v.assigned_charger_kw)}"
        max_c = charger_fleet.get(gkey, {}).get("chargers_recommended", 0)
        ct_arr = cu.get(gkey, np.zeros(n, dtype=int))
        scores = [(s, rates[s], agg[s], site.existing_capacity_kw - agg[s], ct_arr[s] < max_c) for s in valid]
        scores.sort(key=lambda x: (x[1], x[2]))
        erem = eneed; vs = np.zeros(n); vc = 0.0
        for s, rate, _, head, cavail in scores:
            if erem <= 0: break
            if not cavail or head <= 0: continue
            maxp = min(v.assigned_charger_kw, head)
            e = min(maxp * (res/60.0), erem); p = e / (res/60.0)
            vs[s] = p; agg[s] += p; ct_arr[s] += 1; vc += e * rate; erem -= e
        if erem > 0.5:
            cl = sum(1 for s in valid if ct_arr[s] >= max_c); tv = len(valid)
            if max_c == 0: cause = f"No {gkey} chargers"
            elif cl > tv * 0.5: cause = f"Charger contention ({cl}/{tv} slots)"
            else: cause = "Site capacity / time"
            unsched.append({"Vehicle": v.id, "Issue": f"Unmet: {erem:.1f} kWh — {cause}",
                           "Shortfall (kWh)": round(float(erem), 1)})
        sched[v.id] = vs; costs[v.id] = vc
    ev_only = sum(sched.values(), np.zeros(n))
    peak_t = float(np.max(agg)); peak_ev = float(np.max(ev_only))
    total_e = float(np.sum(ev_only) * (res/60.0)); ecost = sum(costs.values())
    dcd = (site.demand_charge_per_kw * peak_t) / 30.0; tc = ecost + dcd
    nc, np_, naive_load = _naive(charging, rates, site.existing_base_load_kw, res, energy_overrides)
    ndd = (site.demand_charge_per_kw * np_) / 30.0; nt = nc + ndd
    cutil = {}; active = max(int(np.sum(ev_only > 0)), 1)
    for k, arr in cu.items():
        mc = max(charger_fleet.get(k, {}).get("chargers_recommended", 1), 1)
        cutil[k] = {"peak": int(np.max(arr)), "avail": charger_fleet.get(k, {}).get("chargers_recommended", 0),
                     "peak_pct": round(float(np.max(arr))/mc*100, 1),
                     "avg_pct": round(float(np.sum(arr))/(active*mc)*100, 1)}
    return {
        "schedule": sched, "aggregate_load": agg, "ev_only_load": ev_only,
        "tariff_rates": rates, "vehicle_costs": costs,
        "total_cost": tc, "total_energy_cost": ecost, "demand_charge_daily": dcd,
        "naive_cost": nt, "naive_peak_kw": np_, "naive_load": naive_load,
        "cost_saving_pct": ((nt-tc)/nt*100) if nt > 0 else 0,
        "total_ev_energy_kwh": total_e, "peak_total_load_kw": peak_t, "peak_ev_load_kw": peak_ev,
        "cap_util": peak_t / site.existing_capacity_kw if site.existing_capacity_kw > 0 else 999,
        "charger_util": cutil, "charger_use": cu, "unscheduled": unsched,
        "n_charging": len(charging), "n_skipped": len(vehicles) - len(charging),
        "n_slots": n, "resolution_min": res,
    }

def _priority(v, rates, res):
    a, d = t2s(v.arrival_time, res), t2s(v.departure_time, res); n = (24*60)//res
    avail = (n-a+d) if d <= a else (d-a)
    if avail == 0: return 999.0
    eps = v.assigned_charger_kw * (res/60.0)
    sn = v.daily_energy_need_kwh / eps if eps > 0 else 999
    return (sn/avail)*2.0 + 1.0/(1.0+d/n)

def _naive(vehicles, rates, base, res, eo=None):
    n = len(rates); load = np.full(n, base); cost = 0.0
    for v in vehicles:
        en = (eo or {}).get(v.id, v.daily_energy_need_kwh)
        a = t2s(v.arrival_time, res); erem = en; eps = v.assigned_charger_kw*(res/60.0); s = a
        while erem > 0 and s < a+n:
            e = min(eps, erem); cost += e*rates[s%n]
            load[s%n] += v.assigned_charger_kw*(e/eps) if eps > 0 else 0; erem -= e; s += 1
    return cost, float(np.max(load)), load


# ============================================================================
# WEEKLY SIMULATION
# ============================================================================

def simulate_week(vehicles, tariff_periods, site, charger_fleet, res=15):
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    max_soc = ASSUMPTIONS["max_soc_limit"]["value"]
    min_floor = ASSUMPTIONS["min_soc_floor"]["value"]
    interval = ASSUMPTIONS["low_km_charge_interval_days"]["value"]
    soc = {v.id: max_soc for v in vehicles}; results = []
    for di in range(7):
        for v in vehicles:
            soc[v.id] -= v.daily_battery_energy_kwh / v.usable_capacity_kwh
        eo = {}
        for v in vehicles:
            deficit = max(max_soc - soc[v.id], 0)
            eo[v.id] = (deficit * v.usable_capacity_kwh) / v.charger_efficiency
        for v in vehicles:
            if not v.needs_daily_charge:
                scheduled = (di % interval) == 0
                dtu = interval - (di % interval); dtu = interval if dtu == 0 else dtu
                cpd = v.daily_battery_energy_kwh / v.usable_capacity_kwh
                v.charge_today = scheduled or (soc[v.id] - cpd * dtu < min_floor)
            else: v.charge_today = True
        r = schedule_fleet(vehicles, tariff_periods, site, charger_fleet, di, res, eo)
        for v in vehicles:
            if v.id in r["schedule"]:
                ge = float(np.sum(r["schedule"][v.id]) * (res/60.0))
                soc[v.id] = min(soc[v.id] + ge * v.charger_efficiency / v.usable_capacity_kwh, max_soc)
        r["day_name"] = days[di]; r["day_index"] = di
        r["soc_end"] = {vid: round(s, 3) for vid, s in soc.items()}; results.append(r)
    return {
        "daily": results, "weekly_peak_kw": max(r["peak_total_load_kw"] for r in results),
        "weekly_energy_kwh": round(sum(r["total_ev_energy_kwh"] for r in results), 1),
        "weekly_cost": round(sum(r["total_cost"] for r in results), 2),
        "soc_history": {v.id: [dr["soc_end"][v.id] for dr in results] for v in vehicles},
    }


# ============================================================================
# SAMPLE DATA
# ============================================================================

def get_sample_fleet():
    return [
        {"id":"BUS-001","vehicle_type":"bus","battery_capacity_kwh":350,"usable_capacity_fraction":0.90,
         "daily_km":180,"arrival_time":"18:00","departure_time":"05:30","max_ac_kw":22,"max_dc_kw":150},
        {"id":"BUS-002","vehicle_type":"bus","battery_capacity_kwh":350,"usable_capacity_fraction":0.90,
         "daily_km":200,"arrival_time":"19:00","departure_time":"05:00","max_ac_kw":22,"max_dc_kw":150},
        {"id":"BUS-003","vehicle_type":"bus","battery_capacity_kwh":280,"usable_capacity_fraction":0.90,
         "daily_km":150,"arrival_time":"17:30","departure_time":"06:00","max_ac_kw":22,"max_dc_kw":120},
        {"id":"VAN-001","vehicle_type":"van","battery_capacity_kwh":75,"usable_capacity_fraction":0.92,
         "daily_km":120,"arrival_time":"16:00","departure_time":"07:00","max_ac_kw":11,"max_dc_kw":50},
        {"id":"VAN-002","vehicle_type":"van","battery_capacity_kwh":75,"usable_capacity_fraction":0.92,
         "daily_km":80,"arrival_time":"15:30","departure_time":"07:30","max_ac_kw":11,"max_dc_kw":50},
        {"id":"VAN-003","vehicle_type":"van","battery_capacity_kwh":75,"usable_capacity_fraction":0.92,
         "daily_km":40,"arrival_time":"14:00","departure_time":"08:00","max_ac_kw":11,"max_dc_kw":50},
        {"id":"VAN-004","vehicle_type":"van","battery_capacity_kwh":60,"usable_capacity_fraction":0.90,
         "daily_km":35,"arrival_time":"16:30","departure_time":"06:30","max_ac_kw":7,"max_dc_kw":50},
        {"id":"TRUCK-001","vehicle_type":"truck","battery_capacity_kwh":200,"usable_capacity_fraction":0.88,
         "daily_km":100,"arrival_time":"17:00","departure_time":"06:00","max_ac_kw":22,"max_dc_kw":120},
    ]

def get_default_tariff():
    return [
        {"name":"off_peak","start_time":"22:00","end_time":"07:00","rate_dollars_per_kwh":0.08,"color":"#02C39A"},
        {"name":"shoulder","start_time":"07:00","end_time":"14:00","rate_dollars_per_kwh":0.18,"color":"#1C7293"},
        {"name":"peak","start_time":"14:00","end_time":"20:00","rate_dollars_per_kwh":0.35,"color":"#9B1C1C"},
        {"name":"shoulder_eve","start_time":"20:00","end_time":"22:00","rate_dollars_per_kwh":0.18,"color":"#1C7293"},
    ]


# ============================================================================
# PLOTTING & DASHBOARD COMPONENTS
# ============================================================================

BRAND = {
    "navy": "#021526", "navy_mid": "#0A2A3E", "blue": "#065A82",
    "teal": "#1C7293", "mint": "#02C39A", "seafoam": "#00A896",
    "text": "#0D1F2D", "muted": "#6B8A99", "error": "#9B1C1C",
    "success": "#027A48", "bg": "#F4F8FB", "card": "#FFFFFF",
}

def plot_gauge(value, max_val, title, unit="%", thresholds=None):
    if thresholds is None:
        thresholds = [(0, 60, BRAND["mint"]), (60, 85, BRAND["teal"]), (85, 100, BRAND["error"])]
    steps = [dict(range=[lo, hi], color=hex_rgba(c, 0.15)) for lo, hi, c in thresholds]
    bar_color = BRAND["mint"]
    for lo, hi, c in thresholds:
        if lo <= value <= hi: bar_color = c; break
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": unit, "font": {"size": 36, "color": BRAND["navy"]}},
        title={"text": title, "font": {"size": 14, "color": BRAND["muted"]}},
        gauge={"axis": {"range": [0, max_val], "tickcolor": BRAND["muted"]},
               "bar": {"color": bar_color, "thickness": 0.7}, "bgcolor": "#F4F8FB",
               "steps": steps,
               "threshold": {"line": {"color": BRAND["error"], "width": 2}, "thickness": 0.8, "value": max_val * 0.95}}
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=30, r=30),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter, Calibri"})
    return fig

def plot_donut(values, labels, colors, title="", hole=0.65):
    fig = go.Figure(go.Pie(values=values, labels=labels, hole=hole,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent', textfont=dict(size=12, color=BRAND["text"])))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10),
                      title=dict(text=title, font=dict(size=14, color=BRAND["blue"])),
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig

def plot_waterfall(categories, values, title=""):
    fig = go.Figure(go.Waterfall(x=categories, y=values,
        connector={"line": {"color": BRAND["muted"], "width": 1, "dash": "dot"}},
        increasing={"marker": {"color": BRAND["blue"]}},
        decreasing={"marker": {"color": BRAND["mint"]}},
        totals={"marker": {"color": BRAND["navy"]}},
        textposition="outside",
        text=[f"{v:+.0f} kW" if i < len(values)-1 else f"{v:.0f} kW" for i, v in enumerate(values)],
        textfont=dict(size=11, color=BRAND["text"])))
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=BRAND["blue"])),
                      height=350, margin=dict(t=60, b=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="kW")
    return fig

def dash_card(label, value, sub="", style="accent"):
    return f"""<div class="dash-card {style}">
        <div class="dash-card-header">{label}</div>
        <div class="dash-card-value">{value}</div>
        <div class="dash-card-sub">{sub}</div>
    </div>"""

def plot_load(result, tps, cap_kw, title="Aggregated Load Profile"):
    n = result["n_slots"]; r = result["resolution_min"]
    times = [s2t(s, r) for s in range(n)]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    rates = result["tariff_rates"]; tc = build_tariff_colors(tps, r)
    fig.add_trace(go.Bar(x=times, y=rates, marker_color=[hex_rgba(c, 0.2) for c in tc],
        name="Tariff $/kWh"), secondary_y=True)
    ev = result.get("ev_only_load", np.zeros(n)); base = result["aggregate_load"] - ev
    fig.add_trace(go.Scatter(x=times, y=base, fill='tozeroy', name="Base Load",
        fillcolor='rgba(107,138,153,0.3)', line=dict(color='#6B8A99', width=1)), secondary_y=False)
    fig.add_trace(go.Scatter(x=times, y=result["aggregate_load"], fill='tonexty',
        name="Managed Load (Base+EV)", fillcolor='rgba(6,90,130,0.4)',
        line=dict(color='#065A82', width=2)), secondary_y=False)
    # Unmanaged/naive load profile as dashed overlay
    if "naive_load" in result:
        fig.add_trace(go.Scatter(x=times, y=result["naive_load"],
            name="Unmanaged Load (charge on arrival)",
            line=dict(color='#9B1C1C', width=2, dash='dash'),
            opacity=0.7), secondary_y=False)
    fig.add_hline(y=cap_kw, line_dash="dash", line_color="#9B1C1C",
                  annotation_text=f"Site Capacity: {cap_kw:.0f} kW", secondary_y=False)
    fig.update_layout(title=title, height=500, barmode='overlay', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    fig.update_xaxes(title_text="Time of Day", dtick=4, tickangle=45)
    fig.update_yaxes(title_text="Power (kW)", secondary_y=False)
    fig.update_yaxes(title_text="$/kWh", secondary_y=True, range=[0, max(rates)*2])
    return fig

def plot_gantt(result):
    n = result["n_slots"]; r = result["resolution_min"]; times = [s2t(s, r) for s in range(n)]
    fig = go.Figure()
    colors = ['#065A82','#02C39A','#1C7293','#00A896','#0A2A3E','#6B8A99','#027A48','#2C4A5A','#6C3483','#9B1C1C']
    for i, vid in enumerate(sorted(result["schedule"].keys())):
        s = result["schedule"][vid]
        if np.sum(s) > 0:
            fig.add_trace(go.Bar(x=times, y=s, name=vid, marker_color=colors[i%len(colors)], opacity=0.8))
    fig.update_layout(title="Vehicle Charging Schedule", barmode='stack', height=500, hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    fig.update_xaxes(title_text="Time of Day", dtick=4, tickangle=45); fig.update_yaxes(title_text="kW")
    return fig

def plot_charger_occ(result, charger_fleet):
    n = result["n_slots"]; r = result["resolution_min"]; times = [s2t(s, r) for s in range(n)]
    fig = go.Figure(); cm = {"AC":"#02C39A","DC":"#065A82"}
    for k, arr in result.get("charger_use", {}).items():
        t = "DC" if k.startswith("DC") else "AC"
        mc = charger_fleet.get(k, {}).get("chargers_recommended", 0)
        fig.add_trace(go.Scatter(x=times, y=arr, name=f"{k.replace('_',' ')}kW ({mc} avail)",
            fill='tozeroy', fillcolor=hex_rgba(cm[t], 0.2), line=dict(color=cm[t], width=2)))
        fig.add_hline(y=mc, line_dash="dot", line_color=cm[t], annotation_text=f"{k} limit")
    fig.update_layout(title="Charger Occupancy", height=400, yaxis_title="In Use")
    fig.update_xaxes(dtick=4, tickangle=45); return fig

def plot_soc(soc_hist):
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    colors = ['#065A82','#02C39A','#1C7293','#00A896','#0A2A3E','#6B8A99','#027A48','#2C4A5A']
    fig = go.Figure()
    for i, (vid, socs) in enumerate(soc_hist.items()):
        fig.add_trace(go.Scatter(x=days, y=[s*100 for s in socs], name=vid, mode='lines+markers',
            line=dict(color=colors[i%len(colors)], width=2), marker=dict(size=8)))
    fig.add_hline(y=ASSUMPTIONS["min_soc_floor"]["value"]*100, line_dash="dash", line_color="#9B1C1C", annotation_text="Min SoC")
    fig.add_hline(y=ASSUMPTIONS["max_soc_limit"]["value"]*100, line_dash="dash", line_color="#02C39A", annotation_text="Max SoC")
    fig.update_layout(title="End-of-Day SoC (after charging)", height=450, yaxis_title="SoC (%)", yaxis_range=[0, 105])
    return fig


# ============================================================================
# CSS — FULLY REWRITTEN TO FIX ALL VISIBILITY ISSUES
# ============================================================================

BRAND_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Calibri', sans-serif !important; }
.stApp { background-color: #F4F8FB; }

/* =====================================================
   SIDEBAR — dark navy background, all text light
   ===================================================== */
section[data-testid="stSidebar"] { background-color: #021526 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5 { color: #FFFFFF !important; }
section[data-testid="stSidebar"] label { color: #A8D4E6 !important; }
section[data-testid="stSidebar"] p { color: #C8E6F5 !important; }
section[data-testid="stSidebar"] small { color: #6B99B5 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(200,230,245,0.15) !important; }

/* Sidebar EXPANDERS — the tariff sections */
section[data-testid="stSidebar"] details {
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(200,230,245,0.15) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] details > summary,
section[data-testid="stSidebar"] details > summary *,
section[data-testid="stSidebar"] details > summary p,
section[data-testid="stSidebar"] details > summary span,
section[data-testid="stSidebar"] details > summary div {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
section[data-testid="stSidebar"] details > summary svg {
    color: #A8D4E6 !important;
    fill: #A8D4E6 !important;
}
/* When expander is OPEN — the [open] attribute is set */
section[data-testid="stSidebar"] details[open] > summary,
section[data-testid="stSidebar"] details[open] > summary *,
section[data-testid="stSidebar"] details[open] > summary p,
section[data-testid="stSidebar"] details[open] > summary span {
    color: #02C39A !important;
}

/* Sidebar INPUT FIELDS — dark bg with visible light text */
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] input[type="number"],
section[data-testid="stSidebar"] [data-baseweb="input"] input {
    color: #FFFFFF !important;
    background-color: #0A2A3E !important;
    border: 1px solid rgba(168,212,230,0.25) !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"] {
    background-color: #0A2A3E !important;
    border-color: rgba(168,212,230,0.25) !important;
}

/* Sidebar slider text */
section[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"],
section[data-testid="stSidebar"] .stSlider div[data-testid="stThumbValue"] {
    color: #C8E6F5 !important;
}

/* =====================================================
   MAIN CONTENT — dark text on light background
   ===================================================== */
.stMainBlockContainer h1 { color: #021526 !important; font-weight: 700 !important; }
.stMainBlockContainer h2, .stMainBlockContainer h3 { color: #065A82 !important; font-weight: 600 !important; }
.stMainBlockContainer p, .stMainBlockContainer li { color: #0D1F2D !important; }
.stMainBlockContainer span { color: #0D1F2D; }
.stMainBlockContainer label { color: #2C4A5A !important; }

/* =====================================================
   TABS — readable at all states
   ===================================================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background-color: #FFFFFF !important;
    border-radius: 10px !important;
    padding: 6px !important;
    border: 1px solid #E2EBF0 !important;
}
/* Unselected tabs: dark text on white bg */
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 14px 32px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    color: #2C4A5A !important;
    background-color: transparent !important;
}
/* Hover: slightly tinted background, still dark text */
.stTabs [data-baseweb="tab"]:hover {
    background-color: #EDF4F8 !important;
    color: #065A82 !important;
}
/* Selected: brand blue bg, white text */
.stTabs [aria-selected="true"] {
    background-color: #065A82 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-size: 1.1rem !important;
    padding: 14px 32px !important;
}
/* Override the underline indicator that Streamlit adds */
.stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* =====================================================
   METRIC CARDS
   ===================================================== */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2EBF0;
    border-radius: 10px;
    padding: 18px 22px;
    box-shadow: 0 2px 6px rgba(2,21,38,0.05);
}
div[data-testid="stMetric"] label {
    color: #6B8A99 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #021526 !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
}

/* =====================================================
   BUTTONS
   ===================================================== */
.stButton > button[kind="primary"],
.stButton > button:first-child {
    background-color: #02C39A !important;
    color: #021526 !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
}
.stButton > button:hover {
    background-color: #00A896 !important;
    box-shadow: 0 3px 12px rgba(2,195,154,0.25);
}
.stDownloadButton > button {
    background-color: #0A2A3E !important;
    color: white !important;
    border: none !important;
    border-radius: 8px;
}

/* =====================================================
   BRAND HEADER
   ===================================================== */
.brand-header {
    background: linear-gradient(135deg, #021526 0%, #0A2A3E 50%, #065A82 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.brand-header .brand-title {
    color: #FFFFFF;
    font-size: 4rem;
    font-weight: 700;
    line-height: 1.2;
    margin: 0;
}
.brand-header .brand-subtitle {
    color: #02C39A;
    font-size: 1rem;
    font-weight: 400;
    margin-top: 6px;
}
.brand-header .brand-tag {
    background: rgba(2,195,154,0.15);
    color: #02C39A;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    white-space: nowrap;
}
.brand-header .brand-logo {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-header .brand-logo-text {
    font-size: 0.85rem;
    font-weight: 600;
    color: #A8D4E6;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* =====================================================
   DASHBOARD CARDS (custom HTML)
   ===================================================== */
.dash-card {
    background: white; border: 1px solid #E2EBF0; border-radius: 12px;
    padding: 24px; box-shadow: 0 2px 8px rgba(2,21,38,0.04); margin-bottom: 16px;
}
.dash-card-header {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: #6B8A99; margin-bottom: 8px;
}
.dash-card-value { font-size: 2rem; font-weight: 700; color: #021526; line-height: 1.1; }
.dash-card-sub { font-size: 0.85rem; color: #6B8A99; margin-top: 4px; }
.dash-card.accent { border-left: 4px solid #02C39A; }
.dash-card.warn { border-left: 4px solid #9B1C1C; }
.dash-card.blue { border-left: 4px solid #065A82; }

.charger-badge {
    display: inline-block; background: linear-gradient(135deg, #065A82 0%, #1C7293 100%);
    color: white; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem;
    font-weight: 600; margin: 3px 4px 3px 0;
}
.charger-badge.ac { background: linear-gradient(135deg, #02C39A 0%, #00A896 100%); color: #021526; }
.section-divider { border: none; border-top: 2px solid #E2EBF0; margin: 2rem 0; }

/* Expanders in main content */
.streamlit-expanderHeader { font-weight: 600 !important; color: #0D1F2D !important; }
div[data-testid="stAlert"] { border-radius: 8px; }
</style>
"""


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="Depot Infrastructure Tool", page_icon="⚡",
                       layout="wide", initial_sidebar_state="expanded")

    st.markdown(BRAND_CSS, unsafe_allow_html=True)

    # ================================================================
    # AUTHENTICATION
    # ================================================================
    # Add/remove users here. Passwords are SHA-256 hashed.
    # To generate a hash: python -c "import hashlib; print(hashlib.sha256(b'yourpassword').hexdigest())"
    import hashlib

    AUTHORISED_USERS = {
        "admin": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918",  # admin
        "beyondev": "d491962f0aa55e7eb92253069bda864cdbdfc7e3701059b52c6c28f09b25e93d",  # beyondev2026
        "demo": "43c27b4e263fa191a6a7ec198cd4d5b47d17413c49d77dc533a01720707e3202",  # demo2026
    }

    def check_password(username, password):
        if username in AUTHORISED_USERS:
            return hashlib.sha256(password.encode()).hexdigest() == AUTHORISED_USERS[username]
        return False

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = ""

    if not st.session_state.authenticated:
        # Login page — branded
        st.markdown("""
        <div style="max-width: 420px; margin: 4rem auto; text-align: center;">
            <div style="background: linear-gradient(135deg, #021526 0%, #0A2A3E 50%, #065A82 100%);
                        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
                <div style="color: #FFFFFF; font-size: 1.6rem; font-weight: 700;">
                    ⚡ Depot Infrastructure Tool
                </div>
                <div style="color: #02C39A; font-size: 0.9rem; margin-top: 6px;">
                    Beyond EV
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_form, col_right = st.columns([1, 1.2, 1])
        with col_form:
            st.markdown("#### Sign In")
            username = st.text_input("Username", key="login_user", placeholder="Enter username")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")

            if st.button("Sign In", type="primary", use_container_width=True):
                if check_password(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            st.caption("Contact your administrator for access credentials.")

        st.stop()  # Nothing below renders until authenticated

    # ---- Logout button in sidebar ----
    # (placed here so it appears at the top of sidebar on every page)

    # Brand header with logo
    st.markdown("""
    <div class="brand-header">
        <div>
            <div class="brand-title">Depot Infrastructure Tool</div>
        </div>
        <div class="brand-logo">
            <img src="data:image/jpeg;base64,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" style="width:120px;height:120px;border-radius:8px;" />
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- SIDEBAR ----
    with st.sidebar:
        # User info and logout
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding: 8px 0 12px 0; margin-bottom: 8px;
                    border-bottom: 1px solid rgba(200,230,245,0.15);">
            <span style="color: #A8D4E6; font-size: 0.82rem;">
                Signed in as <strong style="color:#02C39A">{st.session_state.username}</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Sign Out", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()

        st.markdown("### ⚙️ Configuration")

        st.subheader("Energy Tariff")
        dt = get_default_tariff(); tps = []
        for i, t in enumerate(dt):
            with st.expander(t['name'].replace('_',' ').title(), expanded=False):
                rate = st.number_input("$/kWh", value=t["rate_dollars_per_kwh"],
                    min_value=0.0, max_value=2.0, step=0.01, key=f"r{i}")
                s_ = st.text_input("Start", value=t["start_time"], key=f"s{i}")
                e_ = st.text_input("End", value=t["end_time"], key=f"e{i}")
                tps.append(TariffPeriod(name=t["name"], start_time=s_, end_time=e_,
                    rate_dollars_per_kwh=rate, color=t["color"]))

        st.divider()
        st.subheader("Site Electrical")
        scap = st.number_input("Existing Capacity (kW)", value=200.0, step=10.0,
                                help="Current MSB rating of the site")
        bload = st.number_input("Base Load (kW)", value=50.0, step=5.0,
                                 help="Non-EV load (lighting, HVAC, offices)")
        dcharge = st.number_input("Demand Charge ($/kW/month)", value=0.0, step=0.5,
                                   help="0 to disable. Ausgrid typical: $8–15/kW/month.")
        site = SiteConfig(existing_capacity_kw=scap, existing_base_load_kw=bload,
                          demand_charge_per_kw=dcharge)

        st.divider()
        st.subheader("Consumption (kWh/km)")
        consumption = {}
        for vtype, info in DEFAULT_CONSUMPTION.items():
            consumption[vtype] = st.number_input(
                f"{vtype.title()}", value=info["value"], step=0.01, format="%.3f",
                key=f"cons_{vtype}", help=f"Range: {info['range']}. {info['note']}")

        st.divider()
        st.subheader("Model Assumptions")
        ASSUMPTIONS["max_soc_limit"]["value"] = st.slider("Max SoC (%)", 80, 100, 90, key="a_soc",
            help="ℹ️ Maximum charge target. 90% is industry standard.") / 100.0
        ASSUMPTIONS["low_km_threshold_fraction"]["value"] = st.slider("Low-km Threshold (%)", 10, 50, 30, key="a_lkt",
            help="ℹ️ Below this % of capacity → vehicle skips some charge days.") / 100.0
        ASSUMPTIONS["low_km_charge_interval_days"]["value"] = st.number_input("Low-km Interval (days)",
            value=3, min_value=2, max_value=7, key="a_lki",
            help="ℹ️ How often low-km vehicles charge.")
        ASSUMPTIONS["ambient_hvac_factor"]["value"] = st.slider("HVAC Factor (%)", 100, 130, 110, key="a_hvac",
            help="ℹ️ Energy overhead for HVAC. 110% = mild, 120% = hot climate.") / 100.0
        ASSUMPTIONS["diversity_factor"]["value"] = st.slider("Diversity Factor (%)", 50, 100, 80, key="a_df",
            help="ℹ️ % of chargers at full power simultaneously. Used for MSB/trafo sizing.") / 100.0
        ASSUMPTIONS["charger_utilisation_target"]["value"] = st.slider("Charger Util Target (%)", 50, 95, 70, key="a_ut",
            help="ℹ️ Target throughput for charger fleet sizing. 70% = 30% headroom.") / 100.0
        ASSUMPTIONS["charger_sizing_margin"]["value"] = st.slider("Charger Sizing Margin (%)", 100, 130, 115, key="a_csm",
            help="ℹ️ Safety margin on min charger power. 115% = 15% headroom.") / 100.0

    # ---- TABS ----
    t0, t1, t2, t3, t4, t5 = st.tabs([
        "📖 Overview", "🚌 Fleet", "🏗️ Infrastructure", "📊 Schedule & Load", "📅 Weekly", "📋 Methodology"])

    # ==== OVERVIEW ====
    with t0:
        st.markdown("""
        <div style="max-width: 820px;">

        <h3 style="color: #065A82; margin-bottom: 0.3em;">What this tool does</h3>
        <p style="color: #0D1F2D; font-size: 1.02rem; line-height: 1.7;">
        Fleet operators thinking about going electric face a practical problem before they even order
        a vehicle: what infrastructure do they actually need at the depot? How many chargers, what size,
        will the existing switchboard cope, and does the transformer need upgrading? Getting this wrong
        is expensive. Undersize it and vehicles don't charge in time for their morning runs. Oversize
        it and you've spent six figures on electrical capacity you don't use.
        </p>

        <p style="color: #0D1F2D; font-size: 1.02rem; line-height: 1.7;">
        This tool works through that problem. You feed in your fleet data (vehicle count, daily kilometres,
        depot arrival and departure times, charging rates the vehicles can accept) and it calculates the
        rest. It picks the smallest standard charger that can deliver each vehicle's energy within its
        parking window, favouring cheaper AC chargers where the maths allows. It then figures out how many
        physical chargers the depot needs, accounting for the fact that chargers can be shared between
        vehicles with staggered schedules. From there it sizes the MSB, transformer, and cabling against
        AS/NZS 3000 standards, and flags if you need a DNSP connection upgrade or a dedicated HV supply.
        </p>

        <h3 style="color: #065A82; margin-bottom: 0.3em;">How it validates the result</h3>
        <p style="color: #0D1F2D; font-size: 1.02rem; line-height: 1.7;">
        The scheduling engine runs underneath to validate that the charger fleet actually works. It builds
        a 15-minute resolution load profile across a full week, tracks each vehicle's state-of-charge day
        by day, and confirms every vehicle gets enough energy for its next service run. It also shows what
        the load profile would look like without any charge management, which is useful for demonstrating
        to DNSPs and site owners why managed charging matters.
        </p>

        <h3 style="color: #065A82; margin-bottom: 0.3em;">Where it fits in the electrification journey</h3>
        <p style="color: #0D1F2D; font-size: 1.02rem; line-height: 1.7;">
        This is a planning-phase tool. Before the business case is finalised, before chargers are procured,
        before the DNSP application goes in. It answers the fleet manager who asks "what's this going to
        cost us on the electrical side?" with numbers rather than a rough estimate. It's also useful for
        scenario testing: what happens if we add 10 more vehicles next year, or if parking windows shrink
        because routes get longer?
        </p>

        <p style="color: #0D1F2D; font-size: 1.02rem; line-height: 1.7;">
        It is not a real-time charge management system. It doesn't dispatch chargers or talk to OCPP.
        That's what NexusCharge does once the infrastructure is built and the vehicles are running.
        This tool tells you what to build.
        </p>

        <h3 style="color: #065A82; margin-bottom: 0.3em;">How to use</h3>
        </div>
        """, unsafe_allow_html=True)

        col_step1, col_step2, col_step3 = st.columns(3)
        with col_step1:
            st.markdown(dash_card("Step 1", "Configure",
                "Set your energy tariff, site capacity, base load, and consumption rates in the sidebar.",
                "blue"), unsafe_allow_html=True)
        with col_step2:
            st.markdown(dash_card("Step 2", "Enter Fleet",
                "Go to the Fleet tab. Upload a CSV or edit the sample data with your vehicle details.",
                "accent"), unsafe_allow_html=True)
        with col_step3:
            st.markdown(dash_card("Step 3", "Run Analysis",
                "Click Run Full Analysis. Review results across Infrastructure, Schedule, and Weekly tabs.",
                "blue"), unsafe_allow_html=True)

        st.markdown("""
        <div style="max-width: 820px; margin-top: 1.5rem;">
        <h3 style="color: #065A82; margin-bottom: 0.3em;">Input data required</h3>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Fleet CSV format", expanded=False):
            st.markdown("""
            | Column | Description | Example |
            |--------|-------------|---------|
            | `id` | Unique vehicle identifier | BUS-001 |
            | `vehicle_type` | bus, van, truck, or car | bus |
            | `battery_capacity_kwh` | Total battery capacity | 350 |
            | `usable_capacity_fraction` | Usable fraction (0.85–0.95) | 0.90 |
            | `daily_km` | Average daily distance | 180 |
            | `arrival_time` | Depot arrival (HH:MM) | 18:00 |
            | `departure_time` | Depot departure (HH:MM) | 05:30 |
            | `max_ac_kw` | Vehicle's max AC charge rate | 22 |
            | `max_dc_kw` | Vehicle's max DC charge rate | 150 |
            """)

        st.markdown("""
        <div style="max-width: 820px; margin-top: 1rem;">
        <p style="color: #6B8A99; font-size: 0.88rem;">
        Built by Beyond EV. Assumptions and methodology are documented in the Methodology tab.
        All parameters are adjustable in the sidebar.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with t1:
        st.subheader("Fleet Vehicle Data")
        st.caption("Enter daily km, battery specs, and max AC/DC charging capability per vehicle.")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up: fdf = pd.read_csv(up); st.success(f"Loaded {len(fdf)} vehicles")
        else: fdf = pd.DataFrame(get_sample_fleet()); st.info("Sample fleet (8 vehicles)")

        with st.expander(f"📋 Fleet Data ({len(fdf)} vehicles)", expanded=False):
            edf = st.data_editor(fdf, num_rows="dynamic", use_container_width=True,
                column_config={
                    "vehicle_type": st.column_config.SelectboxColumn("Type", options=["bus","van","truck","car"]),
                    "battery_capacity_kwh": st.column_config.NumberColumn("Battery (kWh)", min_value=10),
                    "daily_km": st.column_config.NumberColumn("Daily km", min_value=0),
                    "max_ac_kw": st.column_config.SelectboxColumn("Max AC (kW)", options=[0,7,11,22]),
                    "max_dc_kw": st.column_config.SelectboxColumn("Max DC (kW)", options=[0,30,50,60,120,150,300]),
                })

        vehicles = []
        for _, row in edf.iterrows():
            try:
                vtype = str(row["vehicle_type"])
                vehicles.append(Vehicle(
                    id=str(row["id"]), vehicle_type=vtype,
                    battery_capacity_kwh=float(row["battery_capacity_kwh"]),
                    usable_capacity_fraction=float(row.get("usable_capacity_fraction", 0.90)),
                    daily_km=float(row["daily_km"]), arrival_time=str(row["arrival_time"]),
                    departure_time=str(row["departure_time"]),
                    max_ac_kw=float(row.get("max_ac_kw", 11)), max_dc_kw=float(row.get("max_dc_kw", 50)),
                    consumption_kwh_per_km=consumption.get(vtype, 0.22)))
            except Exception as e: st.warning(f"Skip: {e}")

        if vehicles:
            assign_result = assign_chargers_to_fleet(vehicles)
            with st.expander(f"🔌 Charger Assignment — {len(vehicles)} vehicles analysed", expanded=False):
                st.dataframe(pd.DataFrame(assign_result["assignments"]), use_container_width=True, hide_index=True)
            if assign_result["warnings"]:
                for w in assign_result["warnings"]: st.warning(f"{w['Vehicle']}: {w['Issue']}")
        if st.button("⚡ Run Full Analysis", type="primary", use_container_width=True):
            if not vehicles: st.error("No vehicles."); st.stop()
            cs = size_charger_fleet(vehicles); se = size_site_electrical(cs, bload, scap)
            cf = cs["by_group"]; r = schedule_fleet(vehicles, tps, site, cf, 0)
            wk = simulate_week(vehicles, tps, site, cf)
            st.session_state.update({"vehicles": vehicles, "cs": cs, "se": se, "cf": cf,
                "result": r, "weekly": wk, "tps": tps, "site": site, "assign": assign_result})
            st.success("Analysis complete — see Infrastructure tab.")

    with t2:
        if "cs" not in st.session_state: st.info("Run analysis from Fleet tab."); st.stop()
        cs = st.session_state["cs"]; se = st.session_state["se"]
        r = st.session_state["result"]; cf = st.session_state["cf"]
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Total Chargers", cs["total_chargers"])
        k2.metric("Installed Capacity", f"{cs['total_installed_kw']} kW")
        k3.metric("Managed Peak", f"{r['peak_total_load_kw']:.0f} kW")
        pr = (1-r['peak_total_load_kw']/r['naive_peak_kw'])*100 if r['naive_peak_kw']>0 else 0
        k4.metric("Peak Reduction", f"{pr:.0f}%", delta=f"{r['peak_total_load_kw']-r['naive_peak_kw']:.0f} kW")
        k5.metric("Site Headroom", f"{se['headroom_kw']} kW",
                   delta="Upgrade needed" if se['upgrade'] else "OK",
                   delta_color="inverse" if se['upgrade'] else "normal")
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("Charger Fleet Recommendation")
        badges = ""
        for k, info in cs["by_group"].items():
            bc = "ac" if info["charger_type"]=="AC" else ""
            badges += f'<span class="charger-badge {bc}">{info["chargers_recommended"]}× {info["label"]}</span>'
        st.markdown(f'<div style="margin-bottom:16px">{badges}</div>', unsafe_allow_html=True)
        cols = st.columns(len(cs["by_group"]))
        for col, (k, info) in zip(cols, cs["by_group"].items()):
            with col:
                st.markdown(dash_card(info["label"],
                    f"{info['chargers_recommended']} charger{'s' if info['chargers_recommended']>1 else ''}",
                    f"{info['vehicle_count']} vehicles · {info['ratio']}<br>{info['total_daily_energy_kwh']} kWh/day",
                    "blue" if info["charger_type"]=="DC" else "accent"), unsafe_allow_html=True)
        cd, cw = st.columns(2)
        with cd:
            st.plotly_chart(plot_donut(
                [info["total_power_kw"] for info in cs["by_group"].values()],
                [info["label"] for info in cs["by_group"].values()],
                [BRAND["blue"] if info["charger_type"]=="DC" else BRAND["mint"] for info in cs["by_group"].values()],
                f"Installed: {cs['total_installed_kw']} kW"), use_container_width=True)
        with cw:
            wfc = ["Base"] + [info["label"] for info in cs["by_group"].values()] + ["Total"]
            wfv = [se["base_load_kw"]] + [info["diversified_power_kw"] for info in cs["by_group"].values()] + [se["total_demand_kw"]]
            st.plotly_chart(plot_waterfall(wfc, wfv, "Demand Buildup (Diversified)"), use_container_width=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("Peak Demand & Site Utilisation")
        g1,g2,g3 = st.columns(3)
        with g1: st.plotly_chart(plot_gauge(min(r['cap_util']*100, 120), 120, "Site Utilisation"), use_container_width=True)
        with g2:
            st.markdown(dash_card("Managed Peak", f"{r['peak_total_load_kw']:.0f} kW", f"vs {r['naive_peak_kw']:.0f} kW unmanaged", "accent"), unsafe_allow_html=True)
            st.markdown(dash_card("Unmanaged Peak", f"{r['naive_peak_kw']:.0f} kW", "All charge on arrival", "warn"), unsafe_allow_html=True)
        with g3:
            st.markdown(dash_card("Site Capacity", f"{se['existing_kw']:.0f} kW", f"Headroom: {se['headroom_kw']} kW", "blue"), unsafe_allow_html=True)
            st.markdown(dash_card("Diversified EV", f"{se['ev_load_kw']:.0f} kW", f"Undiversified: {se['ev_undiversified_kw']} kW", "accent"), unsafe_allow_html=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("Aggregated Load Profile")
        st.plotly_chart(plot_load(r, st.session_state["tps"], st.session_state["site"].existing_capacity_kw), use_container_width=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("Site Electrical Upgrade")
        u1,u2,u3 = st.columns(3)
        u1.markdown(dash_card("Total Demand", f"{se['total_demand_kw']} kW", f"{se['total_demand_kva']:.0f} kVA · {se['total_demand_amps']:.0f}A", "blue"), unsafe_allow_html=True)
        u2.markdown(dash_card("MSB Required", f"{se['msb_amps']}A" if isinstance(se['msb_amps'], int) else str(se['msb_amps']), "", "blue"), unsafe_allow_html=True)
        u3.markdown(dash_card("Transformer", f"{se['trafo_kva']} kVA" if isinstance(se['trafo_kva'], int) else str(se['trafo_kva']), "", "blue"), unsafe_allow_html=True)
        if se["upgrade"]:
            st.error(f"⚠️ Upgrade required. {se['existing_kw']}kW → {se['total_demand_kw']}kW (shortfall: {abs(se['headroom_kw'])}kW)")
            st.dataframe(pd.DataFrame(se["path"]), use_container_width=True, hide_index=True)
            if se["hv_required"]: st.error(f"🔴 HV Connection Required. {se['hv_note']}")
            elif se["hv_note"]: st.warning(se["hv_note"])
        else: st.success(f"✅ Infrastructure sufficient. Headroom: {se['headroom_kw']}kW")
        with st.expander("📋 Reference: MSB & Transformer"):
            c1,c2 = st.columns(2)
            with c1: st.dataframe(pd.DataFrame([{"A":a,"kW":kw} for a,kw in MSB_TIERS]), hide_index=True)
            with c2: st.dataframe(pd.DataFrame([{"kVA":kva,"kW":kw} for kva,kw in TRAFO_TIERS]), hide_index=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("Charger Utilisation")
        gc = st.columns(len(r["charger_util"]))
        for col, (k, info) in zip(gc, r["charger_util"].items()):
            with col:
                st.plotly_chart(plot_gauge(info["avg_pct"], 100, f"{k.replace('_',' ')}kW"), use_container_width=True)
                st.caption(f"Avg: {info['avg_pct']}% · Peak: {info['peak']}/{info['avail']}")
        st.plotly_chart(plot_charger_occ(r, cf), use_container_width=True)
        if r["unscheduled"]:
            st.error(f"⚠️ {len(r['unscheduled'])} vehicle(s) with issues")
            st.dataframe(pd.DataFrame(r["unscheduled"]), use_container_width=True, hide_index=True)
        st.download_button("📥 Infrastructure Report (JSON)", json.dumps({
            "chargers": {k: {kk:vv for kk,vv in v.items() if kk!="vehicle_ids"} for k,v in cs["by_group"].items()},
            "totals": {"n": cs["total_chargers"], "kw": cs["total_installed_kw"], "div_kw": cs["total_diversified_kw"]},
            "site": se, "peak": {"managed": round(r["peak_total_load_kw"],1), "unmanaged": round(r["naive_peak_kw"],1)},
        }, indent=2, default=str), "infra_report.json", "application/json")

    with t3:
        if "result" not in st.session_state: st.info("Run analysis."); st.stop()
        r = st.session_state["result"]
        st.subheader("Charging Schedule")
        st.plotly_chart(plot_gantt(r), use_container_width=True)
        st.subheader("Cost Summary")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Daily Cost", f"${r['total_cost']:.2f}")
        c2.metric("vs Unmanaged", f"{r['cost_saving_pct']:.1f}% less")
        c3.metric("Avg $/kWh", f"${r['total_energy_cost']/r['total_ev_energy_kwh']:.3f}" if r['total_ev_energy_kwh']>0 else "N/A")
        c4.metric("Energy", f"{r['total_ev_energy_kwh']:.0f} kWh")
        with st.expander("📋 Detail"):
            n_ = r["n_slots"]; rs = r["resolution_min"]
            sd = {"Time": [s2t(s, rs) for s in range(n_)]}
            for vid in sorted(r["schedule"].keys()): sd[vid] = np.round(r["schedule"][vid], 1)
            sd["Total kW"] = np.round(r["aggregate_load"], 1)
            sdf = pd.DataFrame(sd); mask = sdf.drop(columns=["Time","Total kW"]).sum(axis=1)>0
            st.dataframe(sdf[mask], use_container_width=True, hide_index=True)
            st.download_button("CSV", sdf.to_csv(index=False), "schedule.csv", "text/csv")

    with t4:
        if "weekly" not in st.session_state: st.info("Run analysis."); st.stop()
        w = st.session_state["weekly"]
        st.subheader("Weekly Summary")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Energy", f"{w['weekly_energy_kwh']:.0f} kWh")
        c2.metric("Cost", f"${w['weekly_cost']:.2f}")
        c3.metric("Peak", f"{w['weekly_peak_kw']:.0f} kW")
        c4.metric("Annual Est.", f"${w['weekly_cost']*52:,.0f}")
        dt_ = [{"Day":dr["day_name"],"Charging":dr["n_charging"],"Skipped":dr["n_skipped"],
                "kWh":round(dr["total_ev_energy_kwh"],1),"Cost $":round(dr["total_cost"],2),
                "Peak kW":round(dr["peak_total_load_kw"],1)} for dr in w["daily"]]
        st.dataframe(pd.DataFrame(dt_), use_container_width=True, hide_index=True)
        st.subheader("SoC Across Week")
        st.plotly_chart(plot_soc(w["soc_history"]), use_container_width=True)
        st.subheader("Daily Load Overlay")
        fig = go.Figure()
        dc_ = ['#065A82','#02C39A','#1C7293','#00A896','#0A2A3E','#6B8A99','#027A48']
        for i, dr in enumerate(w["daily"]):
            ts = [s2t(s, dr["resolution_min"]) for s in range(dr["n_slots"])]
            fig.add_trace(go.Scatter(x=ts, y=dr["aggregate_load"], name=dr["day_name"],
                line=dict(color=dc_[i], width=2), opacity=0.8))
        fig.add_hline(y=st.session_state["site"].existing_capacity_kw, line_dash="dash",
                      line_color="#9B1C1C", annotation_text="Site Capacity")
        fig.update_layout(title="Load by Day", height=450, yaxis_title="kW", hovermode='x unified')
        fig.update_xaxes(dtick=4, tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with t5:
        st.subheader("Methodology & Assumptions")
        st.caption("All parameters used in the model, with justification. Values are adjustable in the sidebar.")

        st.markdown("#### Configurable Parameters")
        for k, info in ASSUMPTIONS.items():
            with st.expander(f"**{k.replace('_',' ').title()}** = {info['value']} {info.get('unit','')}"):
                st.write(info["justification"])
                if "info" in info: st.info(info["info"])
                if "impact" in info: st.caption(f"**Impact:** {info['impact']}")

        st.markdown("#### Charger Selection Logic")
        st.markdown("""
The tool selects the smallest standard commercial charger that can deliver each vehicle's
daily energy requirement within its available parking window.

1. Calculate minimum power needed: `daily_energy_kWh / parking_hours × sizing_margin`
2. Try AC chargers first (7, 11, 22 kW) as they are cheaper to install and require simpler protection
3. Pick the smallest AC size that meets minimum power AND does not exceed the vehicle's max AC capability
4. If no AC charger fits (vehicle needs more power than AC can deliver, or vehicle's AC limit is too low), try DC chargers (30, 50, 60, 120, 150, 300 kW) using the same logic
5. Flag any vehicle where no standard charger can deliver the required energy in the available window
        """)

        st.markdown("#### Charger Fleet Sizing")
        st.markdown("""
Chargers are shared across vehicles, not allocated 1:1. The tool groups vehicles by their
assigned charger type and size, then determines how many physical chargers each group needs
using the binding constraint of two methods:

**Energy method:** total daily energy for the group ÷ (average parking hours × charger power × utilisation target). This answers: "how many charger-hours do we need, and how many chargers running at target utilisation does that require?"

**Concurrency method:** the peak number of vehicles in the group parked at the depot simultaneously. This answers: "at worst case, how many vehicles could need a charger at the same time?"

The tool takes the higher of the two (capped at the vehicle count), then validates the result by running the scheduling engine to confirm all vehicles receive their required energy.
        """)

        st.markdown("#### Site Electrical Sizing")
        st.markdown("""
Site infrastructure is sized following the standard DNSP connection process:

1. **EV load** = total installed charger power × diversity factor (accounts for not all chargers drawing full power simultaneously)
2. **Total site demand** = existing base load + diversified EV load
3. **MSB** = next standard frame size above total demand per AS/NZS 3000
4. **Transformer** = next standard kVA rating above total demand
5. If transformer exceeds 1000 kVA, a dedicated 11kV/22kV HV connection is flagged
6. Upgrade path includes: MSB, transformer, dedicated EV sub-distribution board, cabling, and Type B RCD protection for DC charger circuits (per AS/NZS 3000 clause 2.6.3.2)
        """)

        st.markdown("#### Scheduling Heuristic")
        st.markdown("""
The scheduling engine uses a greedy priority-queue approach to assign charging slots:

1. Vehicles below the low-km threshold skip some charge days (configurable interval), with a safety override that forces charging if state-of-charge would breach the minimum floor
2. Each vehicle is scored by urgency (energy needed relative to parking time) and departure earliness
3. Vehicles are scheduled in priority order. Each vehicle's available parking slots are sorted by tariff rate (cheapest first), then by current aggregate load (least loaded first)
4. Charging is assigned to slots that have both site capacity headroom and a free charger of the correct type
5. The weekly simulation carries state-of-charge forward across days, adjusting each day's energy requirement based on actual SoC rather than assuming a fixed daily need

This is a planning heuristic, not a globally optimal solution. For fleets under 50 vehicles, the cost gap versus a full MILP optimisation is typically under 5%.
        """)

if __name__ == "__main__":
    main()
