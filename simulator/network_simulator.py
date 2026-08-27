"""
Telemetry Simulator for Predictive AI Framework for Dynamic Resource
Allocation in Next Generation Communication Networks.

This module implements:
  - A BaseStation class representing one cell/base station
  - Realistic, non-linear latency and packet-loss models driven by utilization
  - Four scenario generators: normal_traffic, rush_hour, sudden_event, recovery
  - A combined multi-scenario "full day" dataset for downstream ML training

Design notes
------------
Instead of hand-writing thousands of JSON records, we simulate them:
  traffic_demand(t) -> utilization(t) -> latency_ms(t), packet_loss_percent(t)

Latency and packet loss are modeled as non-linear (convex) functions of
utilization, because real radio networks degrade gently at low load and much
more sharply as they approach capacity (queuing-theory-like behavior). Small
Gaussian noise is added to every signal so the data looks like real telemetry
rather than a perfect formula, which is important for training a predictive
model that has to learn from noisy, realistic patterns.
"""

import json
import math
import random
from pathlib import Path

random.seed(42)  # reproducible datasets

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core physical / QoS model
# ---------------------------------------------------------------------------
class BaseStation:
    """Represents one base station and how it converts load into QoS metrics."""

    def __init__(self, station_id, resource_capacity=100, base_latency_ms=15):
        self.station_id = station_id
        self.resource_capacity = resource_capacity
        self.base_latency_ms = base_latency_ms

    def utilization(self, traffic_demand):
        return round(min(traffic_demand / self.resource_capacity, 1.5) * 100, 2)

    def latency_ms(self, utilization_pct, noise=True):
        """Non-linear (sigmoid) latency growth. Stays flat and low until
        roughly 75-85% load, then climbs steeply through a congestion knee,
        and saturates instead of blowing up -- matching how real networks
        behave once admission control / scheduling kicks in under overload."""
        u = utilization_pct / 100.0
        congestion_term = 220 / (1 + math.exp(-11 * (u - 0.85)))
        latency = self.base_latency_ms + congestion_term
        if noise:
            latency += random.gauss(0, max(1.0, latency * 0.03))
        return round(max(5.0, latency), 1)

    def packet_loss_percent(self, utilization_pct, noise=True):
        """Packet loss (sigmoid). Near zero at low/medium load, rises sharply
        once the station is pushed past ~85-90% utilization, saturating at a
        bounded worst-case loss rate rather than diverging."""
        u = utilization_pct / 100.0
        loss = 0.05 + 18 / (1 + math.exp(-13 * (u - 0.90)))
        if noise:
            loss += max(0, random.gauss(0, 0.08))
        return round(max(0.0, min(loss, 20.0)), 2)

    def record(self, timestamp, active_users, traffic_demand):
        util = self.utilization(traffic_demand)
        return {
            "timestamp": timestamp,
            "base_station_id": self.station_id,
            "active_users": int(active_users),
            "traffic_demand": round(traffic_demand, 1),
            "resource_capacity": self.resource_capacity,
            "utilization": util,
            "latency_ms": self.latency_ms(util),
            "packet_loss_percent": self.packet_loss_percent(util),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def users_to_demand(active_users, users_per_capacity_unit=9.5):
    """Rough conversion from concurrent active users to traffic demand units,
    with a bit of per-timestep noise to mimic varying per-user data needs."""
    demand = active_users / users_per_capacity_unit
    demand *= random.uniform(0.95, 1.05)
    return demand


def clamp(value, low, high):
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------
def generate_normal_traffic(stations, num_timestamps=288):
    """Stable, low/medium load across the day with small random fluctuation
    (288 steps = 24h at 5-minute resolution)."""
    records = []
    baselines = {s.station_id: random.uniform(250, 550) for s in stations}
    for t in range(1, num_timestamps + 1):
        for s in stations:
            base = baselines[s.station_id]
            # gentle day/night wave + small noise
            wave = 40 * math.sin(2 * math.pi * t / num_timestamps)
            users = clamp(base + wave + random.gauss(0, 15), 50, 950)
            demand = users_to_demand(users)
            records.append(s.record(t, users, demand))
    return records


def generate_rush_hour(stations, num_timestamps=120):
    """Traffic ramps up to a sharp peak (morning/evening rush) then eases."""
    records = []
    peak_t = num_timestamps * 0.6
    width = num_timestamps / 5.0
    for t in range(1, num_timestamps + 1):
        for s in stations:
            # Gaussian-shaped rush hour bump on top of a moderate baseline
            bump = 700 * math.exp(-((t - peak_t) ** 2) / (2 * width ** 2))
            users = clamp(200 + bump + random.gauss(0, 20), 100, 980)
            demand = users_to_demand(users)
            records.append(s.record(t, users, demand))
    return records


def generate_sudden_event(stations, num_timestamps=100, event_start=40):
    """Sharp, sudden spike (e.g. concert/stadium crowd) rather than a gradual
    ramp -- useful for testing whether a predictive model can catch
    congestion *before* it fully materializes."""
    records = []
    for t in range(1, num_timestamps + 1):
        for s in stations:
            if t < event_start:
                users = clamp(250 + random.gauss(0, 15), 100, 400)
            else:
                # fast exponential-ish rise right after the event trigger
                steps_in = t - event_start
                surge = 900 * (1 - math.exp(-steps_in / 6.0))
                users = clamp(250 + surge + random.gauss(0, 25), 100, 1000)
            demand = users_to_demand(users)
            records.append(s.record(t, users, demand))
    return records


def generate_recovery(stations, num_timestamps=90):
    """Traffic decays from a high-demand peak back down to baseline."""
    records = []
    for t in range(1, num_timestamps + 1):
        for s in stations:
            decay = math.exp(-t / 25.0)
            users = clamp(300 + 700 * decay + random.gauss(0, 15), 150, 1000)
            demand = users_to_demand(users)
            records.append(s.record(t, users, demand))
    return records


def generate_full_simulation(stations, num_timestamps=1440):
    """One long, continuous multi-day-like stream combining normal load,
    two rush-hour peaks, one sudden event, and a recovery tail -- intended as
    the larger dataset for training/testing the predictive model."""
    records = []
    for t in range(1, num_timestamps + 1):
        phase = t / num_timestamps
        for s in stations:
            baseline = 300 + 80 * math.sin(2 * math.pi * t / 288)  # daily wave
            morning_rush = 500 * math.exp(-((t - num_timestamps * 0.25) ** 2) / (2 * 60 ** 2))
            evening_rush = 550 * math.exp(-((t - num_timestamps * 0.7) ** 2) / (2 * 60 ** 2))
            event_spike = 0
            if 0.45 < phase < 0.55:
                steps_in = t - int(num_timestamps * 0.45)
                event_spike = 600 * (1 - math.exp(-max(steps_in, 0) / 10.0))
            users = clamp(
                baseline + morning_rush + evening_rush + event_spike + random.gauss(0, 20),
                50, 1100,
            )
            demand = users_to_demand(users)
            records.append(s.record(t, users, demand))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def save_json(records, filename):
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(records, f, indent=4)
    print(f"Saved {len(records):>6} records -> {path}")


def main():
    stations = [BaseStation(f"BS{i}") for i in range(1, 7)]  # BS1..BS6

    save_json(generate_normal_traffic(stations), "normal_traffic.json")
    save_json(generate_rush_hour(stations), "rush_hour.json")
    save_json(generate_sudden_event(stations), "sudden_event.json")
    save_json(generate_recovery(stations), "recovery.json")
    save_json(generate_full_simulation(stations), "full_simulation.json")


if __name__ == "__main__":
    main()
