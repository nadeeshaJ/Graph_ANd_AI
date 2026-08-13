"""Turn the ethnobotanical JSON into Neo4j-ready CSV tables and EDA charts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "data" / "ethnobotany_master_data_v2.json"
DEFAULT_OUT = ROOT / "outputs"

AILMENT_VARIANTS = {
    "Cold": ["Cold", "Colds", "Bad cold", "Cold/Bronchitis"],
    "Bronchitis": ["Bronchitis", "Bronchial", "Coughs", "Congestion"],
    "Wounds": ["Wounds", "Healing"],
    "Burns": ["Burns", "Burns ointment"],
    "Boils": ["Boils"],
    "Fever": ["Fever", "Scarlet fever historically"],
    "Arthritis": ["Arthritis"],
    "Rheumatism": ["Rheumatism", "Rheumatic pains"],
    "Digestion": ["Digestion", "Stomach", "Constipation", "Laxative", "Bad tummy (soot)"],
    "Calming": ["Calming", "Sleep", "Relaxation", "Feeling down", "Depression"],
    "Circulation": ["Circulation", "Blood tonic", "Iron", "Blood"],
    "Skin": ["Skin", "Skin irritations", "Chapped hands", "Bruising"],
    "Poison": ["Poison", "Poison historically", "Drawing poison"],
    "Women's Health": ["Pregnancy", "Soothing womb after birth"],
    "Bone & Muscle": ["Broken bones", "Twisted ankle", "Sore back", "Bad back", "Hips pain", "Breathing pain"],
    "Stings": ["Bee sting", "Insect stings", "Stings", "Nettle stings"],
    "Eye": ["Eye sty (thorns)"],
    "Measles": ["Measles"],
    "Headache": ["Headache", "Migraines"],
    "General Health": [
        "General health",
        "Health tonic",
        "Tonic",
        "Health fortified",
        "Spring cleanse",
        "Detoxification",
        "Blood cleaning",
    ],
    "Respiratory": ["Asthma"],
}

AILMENT_CATEGORY = {
    "Cold": "Respiratory",
    "Bronchitis": "Respiratory",
    "Respiratory": "Respiratory",
    "Wounds": "Skin/Wound",
    "Burns": "Skin/Wound",
    "Boils": "Skin/Wound",
    "Skin": "Skin/Wound",
    "Stings": "Skin/Wound",
    "Arthritis": "Pain/Musculoskeletal",
    "Rheumatism": "Pain/Musculoskeletal",
    "Bone & Muscle": "Pain/Musculoskeletal",
    "Digestion": "Digestive",
    "Fever": "Infection/Fever",
    "Measles": "Infection/Fever",
    "Calming": "Emotional Wellbeing",
    "Circulation": "Circulatory",
    "Women's Health": "Women's Health",
    "Headache": "Neurological",
    "Eye": "Eye",
    "General Health": "General Tonic",
    "Poison": "Toxic/Poison",
}


def clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def slugify(value) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_toxicity(value: str) -> str:
    text = clean_text(value)
    if "Highly Toxic" in text:
        return "High"
    if "Medium" in text:
        return "Medium"
    return "Low"


def normalize_ailment(usage: str) -> str:
    usage = clean_text(usage)
    for standard, variants in AILMENT_VARIANTS.items():
        if usage in variants:
            return standard
    return usage


def ailment_category(ailment: str) -> str:
    return AILMENT_CATEGORY.get(ailment, "Other")


def normalize_source_type(form: str, location: str) -> str:
    form = clean_text(form).lower()
    location = clean_text(location).lower()
    if "wild" in form:
        return "Wild"
    if "restricted" in form:
        return "Restricted"
    if "dry" in form:
        return "Retail Dry"
    if "fresh" in form or "potted" in form or "oil" in form or "seeds" in form:
        return "Retail Fresh"
    if "imported" in location:
        return "Imported"
    return "Other"


def build_tables(records: list[dict]) -> dict[str, pd.DataFrame]:
    plants_rows = []
    plant_ailment_rows = []
    plant_location_rows = []
    plant_shop_rows = []
    plant_family_rows = []
    categories_rows = []
    seen_categories = set()

    for rec in records:
        plant_id = slugify(rec["scientific_name"] or rec["plant_name"])
        plants_rows.append(
            {
                "plant_id": plant_id,
                "plant_name": clean_text(rec["plant_name"]),
                "scientific_name": clean_text(rec["scientific_name"]),
                "irish_name": clean_text(rec["irish_name"]),
                "is_famine_food": bool(rec["is_famine_food"]),
                "cultural_importance": clean_text(rec["cultural_importance"]),
                "toxicity_label": clean_text(rec["toxicity_level"]),
                "toxicity_class": normalize_toxicity(rec["toxicity_level"]),
            }
        )
        plant_family_rows.append(
            {
                "plant_id": plant_id,
                "family_name": clean_text(rec["plant_family"]),
            }
        )

        for usage in rec.get("usages", []):
            ailment = normalize_ailment(usage)
            category = ailment_category(ailment)
            plant_ailment_rows.append(
                {
                    "plant_id": plant_id,
                    "raw_usage": clean_text(usage),
                    "ailment_name": ailment,
                    "category_name": category,
                }
            )
            if category not in seen_categories:
                categories_rows.append({"category_name": category})
                seen_categories.add(category)

        for loc in rec.get("wild_locations", []):
            plant_location_rows.append(
                {
                    "plant_id": plant_id,
                    "location_name": clean_text(loc),
                }
            )

        for shop in rec.get("dublin_sources", []):
            form = clean_text(shop.get("form"))
            address = clean_text(shop.get("address"))
            plant_shop_rows.append(
                {
                    "plant_id": plant_id,
                    "shop_name": clean_text(shop.get("shop_name")),
                    "address": address,
                    "form": form,
                    "source_type": normalize_source_type(form, address),
                }
            )

    return {
        "plants": pd.DataFrame(plants_rows).drop_duplicates(),
        "plant_ailments": pd.DataFrame(plant_ailment_rows).drop_duplicates(),
        "plant_locations": pd.DataFrame(plant_location_rows).drop_duplicates(),
        "plant_shops": pd.DataFrame(plant_shop_rows).drop_duplicates(),
        "plant_families": pd.DataFrame(plant_family_rows).drop_duplicates(),
        "categories": pd.DataFrame(categories_rows).drop_duplicates(),
    }


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path} {frame.shape}")


def write_charts(tables: dict[str, pd.DataFrame], chart_dir: Path) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    plants = tables["plants"]
    ailments = tables["plant_ailments"]

    usage_counts = (
        ailments.groupby("plant_id")["ailment_name"]
        .nunique()
        .reset_index(name="ailment_count")
        .merge(plants[["plant_id", "plant_name"]], on="plant_id")
        .sort_values("ailment_count", ascending=False)
    )
    top = usage_counts.head(10)
    plt.figure(figsize=(10, 6))
    plt.bar(top["plant_name"], top["ailment_count"])
    plt.xticks(rotation=70, ha="right")
    plt.title("Top 10 most versatile plants")
    plt.ylabel("Distinct ailments")
    plt.tight_layout()
    plt.savefig(chart_dir / "top_versatile_plants.png", dpi=150)
    plt.close()

    category_counts = ailments["category_name"].value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(category_counts.index, category_counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Ailment categories")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(chart_dir / "ailment_categories.png", dpi=150)
    plt.close()
    print(f"wrote charts in {chart_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    print(f"loaded {len(records)} plants from {args.input}")
    tables = build_tables(records)
    write_tables(tables, args.out)
    write_charts(tables, args.out / "charts")


if __name__ == "__main__":
    main()
