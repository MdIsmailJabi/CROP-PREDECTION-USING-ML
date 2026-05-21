"""Market Analysis Module"""
import pandas as pd
import numpy as np

MARKET_DATA = {
    "Rice":  {"price": 2183, "demand": "High",   "export": "High",   "profit": 28, "season": 120, "water": 1200, "color": "#2ecc71"},
    "Ragi":  {"price": 3846, "demand": "Medium", "export": "Medium", "profit": 35, "season": 100, "water": 450,  "color": "#3498db"},
    "Jowar": {"price": 2015, "demand": "Low",    "export": "Low",    "profit": 18, "season": 110, "water": 400,  "color": "#e74c3c"},
    "Corn":  {"price": 1760, "demand": "Medium", "export": "High",   "profit": 30, "season": 90,  "water": 600,  "color": "#f39c12"},
    "Wheat": {"price": 2275, "demand": "High",   "export": "High",   "profit": 25, "season": 130, "water": 400,  "color": "#9b59b6"},
}

DEMAND_SCORE = {"High": 3, "Medium": 2, "Low": 1}
EXPORT_SCORE = {"High": 3, "Medium": 2, "Low": 1}

def get_market_info(crop_name):
    info = MARKET_DATA.get(crop_name, {})
    if not info:
        return {}
    score = (info["profit"] / 10) + DEMAND_SCORE[info["demand"]] + EXPORT_SCORE[info["export"]]
    viability = "Excellent" if score >= 8 else "Good" if score >= 6 else "Moderate"
    return {
        "crop": crop_name,
        "market_price": info["price"],
        "demand_level": info["demand"],
        "export_potential": info["export"],
        "profit_margin": info["profit"],
        "growing_season_days": info["season"],
        "water_requirement_mm": info["water"],
        "market_viability": viability,
        "composite_score": round(score, 2),
        "color": info["color"],
    }

def get_all_market_data():
    return [get_market_info(c) for c in MARKET_DATA.keys()]

def get_top_market_crops(region_type=None):
    all_data = get_all_market_data()
    sorted_data = sorted(all_data, key=lambda x: x["composite_score"], reverse=True)
    return sorted_data[:3]
