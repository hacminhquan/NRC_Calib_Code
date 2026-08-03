"""The exact 57 tabular datasets configured by the QRT upstream repository."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Upstream dataset identifier and source group."""

    name: str
    group: str


_GROUPS: dict[str, tuple[str, ...]] = {
    "uci": ("CPU", "Yacht", "MPG", "Energy", "Crime", "Fish", "Concrete", "Airfoil", "Kin8nm", "Power", "Naval", "Protein"),
    "oml_297": ("wine_quality", "isolet", "cpu_act", "sulfur", "Brazilian_houses", "Ailerons", "MiamiHousing2016", "pol", "elevators", "Bike_Sharing_Demand", "fifa", "california", "superconduct", "house_sales", "house_16H", "diamonds", "medical_charges", "year", "nyc-taxi-green-dec-2016"),
    "oml_299": ("analcatdata_supreme", "Mercedes_Benz_Greener_Manufacturing", "visualizing_soil", "yprop_4_1", "OnlineNewsPopularity", "black_friday", "SGEMM_GPU_kernel_performance", "particulate-matter-ukair-2017"),
    "oml_269": ("tecator", "boston", "MIP-2016-regression", "socmob", "Moneyball", "house_prices_nominal", "us_crime", "quake", "space_ga", "abalone", "SAT11-HAND-runtime-regression", "Santander_transaction_value", "colleges", "topo_2_1", "Allstate_Claims_Severity", "Yolanda", "Buzzinsocialmedia_Twitter", "Airlines_DepDelay_10M"),
}


def qrt57_manifest() -> tuple[DatasetSpec, ...]:
    """Return the 57 non-toy QRT datasets in upstream-config order."""
    manifest = tuple(DatasetSpec(name, group) for group, names in _GROUPS.items() for name in names)
    if len(manifest) != 57:
        raise RuntimeError(f"Expected 57 QRT datasets, found {len(manifest)}")
    return manifest
