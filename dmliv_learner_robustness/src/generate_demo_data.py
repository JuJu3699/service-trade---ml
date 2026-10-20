"""Generate clearly ARTIFICIAL panel data for validating code (not paper data)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate(path: Path, n_partners: int = 24, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_partners):
        c_effect = rng.normal(0, .5)
        tel2005 = rng.uniform(.05, .6)
        for year in range(2005, 2025):
            trend = (year - 2005) / 19.0
            rta = int(rng.uniform() < .2)
            lsbci = float(rng.uniform(.1, 0.6))
            ln_exchange_rate = rng.normal(1, .4)
            ln_gdp_per_capita = rng.normal(9, 1)
            ln_avg_connected_cities = rng.normal(4.8, .5) + .3 * c_effect + .08 * trend
            ln_common_city = rng.normal(3.0, .5) + .2 * c_effect + .15 * trend
            internet_diffusion = (.25 + trend) * tel2005 + rng.uniform(.01, .03)
            ln_internet_iv = np.log(internet_diffusion)
            ln_internet = 2 + .6 * ln_internet_iv + .4 * trend + rng.normal(0, .2)
            ln_direct = 1 + .5 * ln_avg_connected_cities + .3 * c_effect + rng.normal(0, .4)
            ln_seats = 10 + .4 * ln_common_city + .2 * c_effect + rng.normal(0, .3)
            base = 3 + .2 * ln_gdp_per_capita + .2 * c_effect + .08 * year / 2025
            names = ["commercial_export", "commercial_import", "transport_export",
                     "transport_import", "travel_export", "travel_import"]
            ys = {f"ln_{name}": base + .15 * ln_direct + .07 * ln_seats +
                  .1 * ln_internet + .02 * ln_direct * ln_internet + rng.normal(0, .35)
                  for name in names}
            rows.append({"partner": f"SYNTHETIC_{i:03d}", "year": year,
                         "ln_direct_cities": ln_direct,
                         "ln_average_seats_per_direct_city": ln_seats,
                         "ln_internet_penetration": ln_internet,
                         "ln_average_connect_city_number": ln_avg_connected_cities,
                         "ln_common_city_number": ln_common_city,
                         "ln_internet_diffusion_exposure": ln_internet_iv,
                         "rta": rta, "lsbci": lsbci,
                         "ln_exchange_rate": ln_exchange_rate,
                         "ln_gdp_per_capita": ln_gdp_per_capita,
                         **ys})
    path.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {len(rows)} SYNTHETIC observations to {path} (NOT real data).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/synthetic_example.csv")
    p.add_argument("--partners", type=int, default=24)
    a = p.parse_args()
    generate(Path(a.output), a.partners)
