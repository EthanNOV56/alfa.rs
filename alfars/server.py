"""
FastAPI server for factor backtest visualization.

This module provides an HTTP API for running backtests and returning
NAV (Net Asset Value) data for interactive visualization.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import numpy as np

app = FastAPI(
    title="Alfa.rs Backtest API",
    description="API for factor backtesting and NAV visualization",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Alpha101/Alpha191 factor definitions
ALPHA_FACTORS = {
    # Alpha001-010 from Alpha101
    "alpha001": {"name": "Alpha001", "expression": "rank(ts_argmax(power(returns, 2), 5)) - 0.5", "description": "Time series rank of max power returns over 5 days"},
    "alpha002": {"name": "Alpha002", "expression": "-1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)", "description": "Correlation between volume change rank and return rank"},
    "alpha003": {"name": "Alpha003", "expression": "-1 * correlation(rank(open), rank(volume), 10)", "description": "Correlation between open price rank and volume rank"},
    "alpha004": {"name": "Alpha004", "expression": "-1 * ts_rank(rank(low), 9)", "description": "Time series rank of low price rank over 9 days"},
    "alpha005": {"name": "Alpha005", "expression": "(rank((open - ts_min(open, 30))) / (rank((ts_max(open, 30) - open))))", "description": "Relative position of open within 30-day range"},
    "alpha006": {"name": "Alpha006", "expression": "-1 * correlation(open, volume, 10)", "description": "Correlation between open price and volume"},
    "alpha007": {"name": "Alpha007", "expression": "(adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : -1", "description": "Volume-adjusted momentum"},
    "alpha008": {"name": "Alpha008", "expression": "-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))", "description": "Momentum of open-return product"},
    "alpha009": {"name": "Alpha009", "expression": "(0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : -delta(close, 1))", "description": "Price momentum with reversal"},
    "alpha010": {"name": "Alpha010", "expression": "rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : -delta(close, 1))))", "description": "Reversal momentum filter"},
    # Alpha011-020
    "alpha011": {"name": "Alpha011", "expression": "rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : -delta(close, 1))))", "description": "Price change with reversal"},
    "alpha012": {"name": "Alpha012", "expression": "sign(delta(close, 7))", "description": "Sign of 7-day price change"},
    "alpha013": {"name": "Alpha013", "expression": "-1 * rank(((sum(returns, 240) - sum(returns, 20)) / 220.0) - ts_rank(close, 10))", "description": "Long-term momentum adjusted by price rank"},
    "alpha014": {"name": "Alpha014", "expression": "-1 * delta(close / delay(close, 9), 9)", "description": "9-day price momentum"},
    "alpha015": {"name": "Alpha015", "expression": "-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)", "description": "Sum of high-volume correlation ranks"},
    "alpha016": {"name": "Alpha016", "expression": "-1 * ts_rank(correlation(close, sum(adv20, 22), 9), 7)", "description": "Time series rank of volume-weighted price"},
    "alpha017": {"name": "Alpha017", "expression": "-1 * ((rank((close - ts_min(low, 12))) / (rank((ts_max(high, 12) - close)))) * -1)", "description": "Relative strength within range"},
    "alpha018": {"name": "Alpha018", "expression": "-1 * rank(correlation(high, rank(volume), 5))", "description": "High price and volume correlation"},
    "alpha019": {"name": "Alpha019", "expression": "- (((rank((open - delay(high, 1))) / rank((high - low)))) * ts_rank(close, 10))", "description": "Opening momentum adjusted by range"},
    "alpha020": {"name": "Alpha020", "expression": "-1 * (rank((open - delay(open, 1))) * ts_rank(ts_rank(close, 10), 10))", "description": "Opening change momentum"},
    # Alpha021-030
    "alpha021": {"name": "Alpha021", "expression": "-1 * sum(close > delay(close, 1), 5) / 5 * rank(((sum(returns, 240) - sum(returns, 20)) / 220) * ts_rank(volume, 32))", "description": "Up day count weighted by momentum"},
    "alpha022": {"name": "Alpha022", "expression": "-1 * ts_rank(correlation(rank(high), rank(adv15), 9), 7)", "description": "High price correlation with volume"},
    "alpha023": {"name": "Alpha023", "expression": "ts_rank(((ts_min(rank(low), 5) + ts_rank(_power(((returns * ((close - low) / (high - low)) - ts_rank(close - 20, 4))), 2), 4)) * -1), 3)", "description": "Composite momentum signal"},
    "alpha024": {"name": "Alpha024", "expression": "-1 * sign((((close - delay(close, 7)) + (close - delay(close, 8))) + (close - delay(close, 9))))", "description": "Triple day price direction"},
    "alpha025": {"name": "Alpha025", "expression": "(rank((open - delay(open, 1))) / rank(((high - low) / (delay(close, 1))))) * -1", "description": "Opening strength relative to volatility"},
    "alpha026": {"name": "Alpha026", "expression": "ts_rank((ts_max(rank(correlation(rank(volume), ts_rank(((high + low) / 2), 3), 5)), 4), 6)", "description": "Volume-price correlation rank"},
    "alpha027": {"name": "Alpha027", "expression": "-1 * rank(correlation(ts_argmax(correlation(ts_rank(close, 3), ts_rank(adv20, 12), 6), 4), ts_rank(close, 3)))", "description": "Max correlation momentum"},
    "alpha028": {"name": "Alpha028", "expression": "-1 * ts_rank(ts_argmax(correlation(ts_rank(close, 4), ts_rank(adv20, 20), 8), 6), 4)", "description": "Cross-sectional momentum"},
    "alpha029": {"name": "Alpha029", "expression": "-1 * rank(correlation(open, sum(adv60, 9), 6)) + (rank((ts_argmax(correlation(rank(ts_rank(open, 30)), rank(ts_rank(rank(low, 8)), 7), 4)) - rank(ts_rank(ts_argmin(close, 30), 2)))) * -1)", "description": "Complex momentum signal"},
    "alpha030": {"name": "Alpha030", "expression": "-1 * rank(correlation(sum(((high + low) / 2), 20), sum(adv40, 20), 9)) * rank(correlation(ts_argmax(close, 30), ts_argmin(close, 30), 7))", "description": "Mean price vs volume correlation"},
    # Alpha031-040
    "alpha031": {"name": "Alpha031", "expression": "0 - (1 * ((1.5 * rank(log(sum(ts_min(rank(low), 2) - rank(ts_argmin(correlation(ts_rank(high, 5), ts_rank(adv10, 4), 3), 3))))) - rank(sum(correlation(rank(open), rank(adv15), 6), 2))))", "description": "Composite low price signal"},
    "alpha032": {"name": "Alpha032", "expression": "0 - (1 * (rank((sum(returns, 7) - sum(returns, 14))) * sqrt(abs(pow(correlation(close, adv90, 4), 2)))))", "description": "Short-term momentum vs correlation"},
    "alpha033": {"name": "Alpha033", "expression": "0 - (1 * (rank(correlation(sum(((high + low) / 2), 19), sum(adv60, 19), 9)) - rank(correlation(ts_argmin(low, 21), ts_argmax(high, 21), 7)))", "description": "Mean price vs extreme correlation"},
    "alpha034": {"name": "Alpha034", "expression": "0 - (rank(correlation(high, rank(volume), 5)) - ts_rank(ts_argmax(correlation(ts_rank(low, 7), ts_rank(adv60, 4), 3), 9), 7))", "description": "High-volume signal"},
    "alpha035": {"name": "Alpha035", "expression": "-1 * ts_rank(correlation(rank(high), rank(adv15), 5), 4)", "description": "High price and volume rank correlation"},
    "alpha036": {"name": "Alpha036", "expression": "0 - (1 * ((rank((sum(delay(close / delay(close, 9), 1), 12)) / sum(sum(delay(close / delay(close, 9), 1), 12), 12))) * rank((sum(correlation(close, volume, 12), 12) - rank(ts_argmin(correlation(ts_rank(close, 20), ts_rank(adv60, 4), 18), 6)))) * -1))", "description": "Price momentum with volume"},
    "alpha037": {"name": "Alpha037", "expression": "rank(correlation(delay(close / delay(close, 20), 1), close, 250)) * rank(correlation(rank(ts_argmax(correlation(close, adv81, 9), 5)), rank(close), 5))", "description": "Long-term price correlation"},
    "alpha038": {"name": "Alpha038", "expression": "0 - (rank(correlation(open, sum(adv5, 265), 225)) - rank(ts_rank(ts_argmin(correlation(rank(open), rank(adv15), 20), 7), 7)))", "description": "Opening price momentum"},
    "alpha039": {"name": "Alpha039", "expression": "0 - (rank(correlation(open, volume, 10)) * rank(returns))", "description": "Volume-return correlation"},
    "alpha040": {"name": "Alpha040", "expression": "0 - (1 * ((rank((sum(delay(close, 1), 12)) / sum(sum(delay(close, 1), 12), 12))) - rank(correlation(close, rank(close) + ts_rank(close, 20), 8)) * -1))", "description": "Price momentum with correlation"},
    # Alpha041-050
    "alpha041": {"name": "Alpha041", "expression": "0 - (1 * ((rank((sum(delay(close / delay(close, 1), 1), 12)) / sum(sum(delay(close / delay(close, 1), 1), 12), 12))) * rank(correlation(close, rank(close) + ts_rank(close, 20), 8))) * -1)", "description": "Daily return momentum"},
    "alpha042": {"name": "Alpha042", "expression": "0 - (1 * (rank(correlation(sum(((high + low) / 2), 20), sum(adv60, 20), 9)) - rank(correlation(ts_rank(ts_argmax(close, 30), 10), ts_rank(ts_min(close, 30), 20), 7))))", "description": "Mean price vs returns correlation"},
    "alpha043": {"name": "Alpha043", "expression": "0 - (rank(correlation(high, rank(volume), 5)) - ts_rank(ts_rank(ts_argmin(correlation(rank(low), rank(adv81), 8), 20), 5), 8))", "description": "High-volume price signal"},
    "alpha044": {"name": "Alpha044", "expression": "-1 * ts_rank(correlation(close, sum(adv30, 37), 9), 12)", "description": "Price-volume correlation"},
    "alpha045": {"name": "Alpha045", "expression": "-1 * rank(correlation(ts_rank(correlation(close, adv81, 8), 5), rank(low), 6))", "description": "Volume-weighted price correlation"},
    "alpha046": {"name": "Alpha046", "expression": "(0 - (1 * ((2 * rank(delta(((close - low) - (high - low)) / ((high - low) + 0.001), 4.8)) * rank(correlation(ts_rank(close, 4), ts_rank(adv78, 20), 8)))) * -1))", "description": "Range-based momentum"},
    "alpha047": {"name": "Alpha047", "expression": "0 - (1 * (rank(correlation(high, rank(adv15), 5)) - rank(correlation(ts_rank(ts_argmin(low, 30), 2), ts_rank(adv60, 2), 8))) * -1)", "description": "Price-volume divergence"},
    "alpha048": {"name": "Alpha048", "expression": "0 - (1 * (max(rank(correlation(rank(volume), ts_rank(((high + low) / 2), 4), 4)), rank(correlation(rank(close), rank(close), 12))) * rank(correlation(sum(returns, 100), rank(close), 5))) * -1)", "description": "Volume-price momentum"},
    "alpha049": {"name": "Alpha049", "expression": "0 - (1 * (rank(correlation(close, sum(adv30, 37), 15)) - rank(correlation(rank(ts_argmin(close, 30)), rank(correlation(ts_rank(close, 8), ts_rank(adv60, 20), 8)), 13))) * -1)", "description": "Short-term reversal"},
    "alpha050": {"name": "Alpha050", "expression": "0 - (1 * ((rank((close - ts_min(low, 12))) / (rank((ts_max(high, 12) - close)))) * rank(correlation(ts_rank(high, 13), ts_rank(adv26, 5), 5))) * -1))", "description": "Relative strength with volume"},
    # Alpha051-060
    "alpha051": {"name": "Alpha051", "expression": "0 - (1 * (rank(correlation(close, sum(adv20, 50), 5)) - rank(correlation(correlation(close, adv81, 4), ts_rank(close, 4), 14))) * -1)", "description": "Price-volume relationship"},
    "alpha052": {"name": "Alpha052", "expression": "0 - (1 * ((rank((ts_min(rank(low), 5)) + ts_rank(abs(delta((close - open) / open, 3.34298)), 15)) + sign(delta(((close - open) / open), 3.34298))) * -1))", "description": "Low price with intraday signal"},
    "alpha053": {"name": "Alpha053", "expression": "0 - (1 * ((rank(correlation(ts_rank(high, 18), ts_rank(adv5, 2), 18)) * -1) * (2 * power(rank((close - ts_min(low, 6))), 3))))", "description": "High price momentum"},
    "alpha054": {"name": "Alpha054", "expression": "0 - (1 * (max(rank(correlation(rank(volume), ts_rank(high, 10), 5)), rank(correlation(close, sum(adv30, 10), 5))) * -1))", "description": "Volume vs price momentum"},
    "alpha055": {"name": "Alpha055", "expression": "0 - (1 * (min(rank(correlation(rank(open), rank(adv15), 10)), rank(correlation(rank(high), rank(adv20), 10))) * -1))", "description": "Open-high correlation"},
    "alpha056": {"name": "Alpha056", "expression": "0 - (1 * ((rank((close - ts_min(low, 11))) / (rank((ts_max(high, 11) - close)))) + sign(delta(close, 1))) * -1)", "description": "Relative strength with direction"},
    "alpha057": {"name": "Alpha057", "expression": "(0 - (1 * ((rank((sum(delay(close, 2), 8)) / sum(sum(delay(close, 2), 8), 8))) * rank(correlation(ts_rank(close, 20), ts_rank(adv20, 5), 8))) * -1)))", "description": "Short-term momentum"},
    "alpha058": {"name": "Alpha058", "expression": "0 - (rank(correlation(sum(((high + low) / 2), 19), sum(adv60, 19), 8)) - rank(correlation(sqrt(floor(adv40)), sqrt(floor(adv80)), 5)))", "description": "Mean price vs volume correlation"},
    "alpha059": {"name": "Alpha059", "expression": "0 - (1 * (rank(correlation(high, rank(adv10), 5)) - rank(correlation(ts_rank(ts_argmin(low, 30), 6), ts_rank(adv60, 3), 3))) * -1))", "description": "Price-volume divergence"},
    "alpha060": {"name": "Alpha060", "expression": "(0 - (1 * ((rank((sum(delay(close, 1), 12)) / sum(sum(delay(close, 1), 12), 12))) - rank(correlation(ts_rank(close, 5), ts_rank(adv60, 3), 10))) * -1)))", "description": "Momentum with correlation"},
    # Alpha061-070
    "alpha061": {"name": "Alpha061", "expression": "(rank((ts_max(rank(correlation(rank(open), rank(adv15), 10)), 10)) * -1))", "description": "Open price momentum"},
    "alpha062": {"name": "Alpha062", "expression": "0 - (1 * (rank(correlation(ts_argmax(correlation(rank(close), ts_rank(adv10, 4), 3), 4)), rank(low), 4)) * -1)", "description": "Max correlation momentum"},
    "alpha063": {"name": "Alpha063", "expression": "(rank((ts_min(rank(open), 8)) + ts_rank(abs(delta(close / open, 4.16733)), 8))) * -1", "description": "Open price with intraday change"},
    "alpha064": {"name": "Alpha064", "expression": "-1 * rank(correlation(high, rank(adv15), 5))", "description": "High price and volume correlation"},
    "alpha065": {"name": "Alpha065", "expression": "(rank((ts_max(rank(correlation(rank(volume), ts_rank(adv81, 8), 5)), 5)) - rank(ts_argmin(correlation(rank(close), rank(adv81), 3), 12)))) * -1)", "description": "Volume-price divergence"},
    "alpha066": {"name": "Alpha066", "expression": "0 - (1 * (max(rank(correlation(rank(volume), ts_rank(high, 6), 5)), rank(correlation(close, sum(adv30, 5), 5))) * -1))", "description": "Volume vs price momentum"},
    "alpha067": {"name": "Alpha067", "expression": "0 - (rank(ts_rank(correlation(close, sum(adv81, 8), 6), 4)) - rank(ts_rank(ts_argmin(correlation(close, adv81, 2), 3), 16))) * -1)", "description": "Price-volume correlation"},
    "alpha068": {"name": "Alpha068", "expression": "0 - (1 * ((rank((close - delay(close, 3))) * rank(correlation(close, rank(adv20), 10))) * -1))", "description": "3-day price momentum"},
    "alpha069": {"name": "Alpha069", "expression": "0 - (1 * ((rank(correlation(ts_rank(high, 21), ts_rank(adv81, 9), 8)) - rank(correlation(ts_rank(ts_argmin(low, 21), 2), ts_rank(adv81, 6), 7))) * -1))", "description": "Price momentum with volume"},
    "alpha070": {"name": "Alpha070", "expression": "(rank(correlation(ts_rank(ts_argmin(correlation(rank(close) - rank(adv81), 6), 4), 6), rank(close), 5)) * -1)", "description": "Price vs volume divergence"},
    # Alpha071-080
    "alpha071": {"name": "Alpha071", "expression": "(0 - (1 * ((rank(correlation(close, rank(adv20), 5)) - rank(correlation(ts_rank(ts_argmin(low, 22), 6), ts_rank(adv81, 6), 4))) * -1)))", "description": "Price vs volume signal"},
    "alpha072": {"name": "Alpha072", "expression": "0 - (rank(correlation(open, sum(adv15, 10), 10)) - rank(correlation(ts_rank(ts_argmin(open, 30), 3), ts_rank(adv90, 10), 7)))", "description": "Opening momentum"},
    "alpha073": {"name": "Alpha073", "expression": "0 - (rank(correlation(open, volume, 10)) * rank(returns) * -1)", "description": "Volume-return correlation"},
    "alpha074": {"name": "Alpha074", "expression": "(rank((ts_min(rank(low), 5)) + ts_rank(abs(delta(((close - open) / open), 3.69741)), 15))) * -1", "description": "Low price with intraday signal"},
    "alpha075": {"name": "Alpha075", "expression": "(0 - (1 * ((rank(correlation(sum(((high + low) / 2), 19), sum(adv60, 19), 9)) - rank(correlation(ts_rank(ts_argmin(low, 25), 3), ts_rank(high, 6), 7))) * -1)))", "description": "Mean price vs returns"},
    "alpha076": {"name": "Alpha076", "expression": "0 - (1 * ((rank(correlation(close, adv81, 8)) - rank(correlation(ts_rank(correlation(close, adv15, 4), 8), ts_rank(high, 9), 6))) * -1))", "description": "Price vs volume correlation"},
    "alpha077": {"name": "Alpha077", "expression": "0 - (rank(correlation(high, rank(adv50), 5)) - ts_rank(ts_rank(ts_argmin(correlation(low, adv81, 6), 7), 9), 6))", "description": "High price momentum"},
    "alpha078": {"name": "Alpha078", "expression": "0 - (rank(correlation(sum(delay(close / delay(close, 20), 1), 20), sum(close / delay(close, 20), 20), 250)) * rank(correlation(open, volume, 100)) * -1)", "description": "Long-term price momentum"},
    "alpha079": {"name": "Alpha079", "expression": "(rank(correlation(ts_rank(close, 15), ts_rank(adv25, 20), 8)) * -1)", "description": "Price rank correlation"},
    "alpha080": {"name": "Alpha080", "expression": "0 - (rank(correlation(correlation(close, adv81, 4), ts_rank(close, 19), 14)) - rank(correlation(rank(ts_argmax(close, 30)), rank(ts_argmin(close, 30)), 14)))", "description": "Price vs returns correlation"},
    # Alpha081-090
    "alpha081": {"name": "Alpha081", "expression": "(rank(correlation(open, sum(adv5, 25), 4)) - rank(ts_rank(ts_argmin(correlation(rank(open), rank(adv15), 6), 2.41), 6))) * -1", "description": "Opening momentum"},
    "alpha082": {"name": "Alpha082", "expression": "(0 - (1 * ((rank(correlation(sum(((high + low) / 2), 8), sum(adv60, 8), 5)) - rank(correlation(ts_rank(ts_argmax(low, 30), 4), ts_rank(high, 4), 4))) * -1)))", "description": "Mean price momentum"},
    "alpha083": {"name": "Alpha083", "expression": "(rank(correlation(open, sum(adv10, 100), 100)) - rank(correlation(ts_rank(ts_argmin(open, 30), 2), ts_rank(adv90, 100), 100))) * -1", "description": "Long-term opening momentum"},
    "alpha084": {"name": "Alpha084", "expression": "0 - (1 * ((rank(correlation(close, sum(adv30, 37), 9)) - rank(correlation(ts_rank(ts_argmin(correlation(rank(open), rank(adv15), 20), 7), 6), ts_rank(close, 3), 3))) * -1))", "description": "Price-volume signal"},
    "alpha085": {"name": "Alpha085", "expression": "(rank(correlation(delay(close / delay(close, 20), 1), close, 250)) * rank(correlation(ts_rank(close, 8), ts_rank(adv60, 20), 8)))", "description": "Long-term momentum"},
    "alpha086": {"name": "Alpha086", "expression": "0 - (rank(correlation(high, rank(adv15), 5)) - rank(correlation(ts_rank(ts_argmin(low, 30), 2), ts_rank(adv45, 18), 15)))", "description": "High vs low momentum"},
    "alpha087": {"name": "Alpha087", "expression": "(rank(correlation(close, sum(adv30, 37), 15)) - rank(correlation(correlation(close, adv81, 4), ts_rank(close, 4), 14))) * -1", "description": "Short-term reversal"},
    "alpha088": {"name": "Alpha088", "expression": "(rank(correlation(sum(((high + low) / 2), 19), sum(adv40, 19), 9)) - rank(correlation(ts_rank(ts_argmax(low, 30), 2), ts_rank(high, 4), 7))) * -1", "description": "Mean price momentum"},
    "alpha089": {"name": "Alpha089", "expression": "(rank(correlation(high, rank(adv15), 5)) - rank(correlation(ts_rank(ts_argmin(low, 30), 2), ts_rank(adv40, 3), 8))) * -1", "description": "High-low divergence"},
    "alpha090": {"name": "Alpha090", "expression": "(0 - (1 * ((rank((close - delay(close, 1))) / rank(sum(abs(delta(close, 1)), 5)))) * rank(correlation(close, adv30, 10)) * -1)))", "description": "Price change vs volume"},
    # Alpha091-100
    "alpha091": {"name": "Alpha091", "expression": "(rank(correlation(delay(close / delay(close, 9), 1), close, 240)) * rank(correlation(rank(((close - low) / (high - low))), rank(adv30), 12)))", "description": "Long-term momentum with volume"},
    "alpha092": {"name": "Alpha092", "expression": "(0 - (1 * ((rank((sum(delay(close, 2), 12)) / sum(sum(delay(close, 2), 12), 12))) - rank(correlation(ts_rank(close, 4), ts_rank(adv20, 4), 12))) * -1)))", "description": "Short-term momentum"},
    "alpha093": {"name": "Alpha093", "expression": "(rank(correlation(open, sum(adv20, 25), 25)) - rank(correlation(open, ts_rank(close, 25), 25))) * -1", "description": "Open vs close momentum"},
    "alpha094": {"name": "Alpha094", "expression": "(rank(correlation(high, rank(adv50), 5)) - rank(correlation(ts_rank(ts_argmin(low, 50), 3), ts_rank(high, 2), 4))) * -1", "description": "Price-volume divergence"},
    "alpha095": {"name": "Alpha095", "expression": "(rank(correlation(close, sum(adv30, 50), 5)) - rank(correlation(ts_rank(correlation(close, adv81, 5), 4), ts_rank(close, 3), 6), 5)) * -1", "description": "Short-term reversal"},
    "alpha096": {"name": "Alpha096", "expression": "0 - (1 * (rank(correlation(open, volume, 10)) * rank(returns)) * -1)", "description": "Volume-return correlation"},
    "alpha097": {"name": "Alpha097", "expression": "(rank((ts_min(rank(low), 9)) + ts_rank(abs(delta((close - open) / open, 14.2517)), 17))) * -1", "description": "Low price with intraday signal"},
    "alpha098": {"name": "Alpha098", "expression": "(0 - (1 * ((rank(correlation(sum(((high + low) / 2), 20), sum(adv40, 20), 9)) - rank(correlation(ts_rank(high, 2), ts_rank(low, 5), 4))) * -1)))", "description": "Mean price momentum"},
    "alpha099": {"name": "Alpha099", "expression": "(rank(correlation(ts_rank(ts_argmin(correlation(rank(open), rank(adv15), 20), 5), 19), ts_rank(close, 3), 13)) * -1)", "description": "Open-close correlation"},
    "alpha100": {"name": "Alpha100", "expression": "(rank(correlation(open, sum(adv60, 9), 6)) - rank(correlation(open, ts_rank(close, 10), 7))) * -1", "description": "Opening momentum"},
    # Alpha101-110
    "alpha101": {"name": "Alpha101", "expression": "(close - open) / ((high - low) + 0.001)", "description": "Intraday return normalized by range"},
    "alpha102": {"name": "Alpha102", "expression": "rank(ts_argmax(close, 30))", "description": "Time series rank of close price position"},
    "alpha103": {"name": "Alpha103", "expression": "rank(ts_argmax(power(returns, 2), 20)) - 0.5", "description": "Max return over 20 days"},
    "alpha104": {"name": "Alpha104", "expression": "rank(ts_argmax(correlation(ts_rank(close, 5), ts_rank(volume, 5), 5), 5)) - 0.5", "description": "Time series max correlation rank"},
    "alpha105": {"name": "Alpha105", "expression": "rank(ts_argmax(correlation(rank(high), rank(volume), 3), 5))", "description": "High-volume correlation rank"},
    # Alpha106-115
    "alpha106": {"name": "Alpha106", "expression": "rank(correlation(close, ts_sum(adv20, 30), 15))", "description": "Close and volume correlation"},
    "alpha107": {"name": "Alpha107", "expression": "rank(correlation(ts_rank(close, 5), ts_rank(adv81, 20), 8))", "description": "Cross-sectional price rank"},
    "alpha108": {"name": "Alpha108", "expression": "-1 * rank(ts_argmax(correlation(close, ts_rank(adv30, 4), 8), 4))", "description": "Max correlation with volume"},
    "alpha109": {"name": "Alpha109", "expression": "rank(ts_argmax(correlation(rank(close), rank(adv81), 10), 10))", "description": "Price rank correlation"},
    "alpha110": {"name": "Alpha110", "expression": "-1 * ts_rank(ts_argmax(correlation(close, ts_rank(adv10, 5), 10), 3), 15)", "description": "Price-volume correlation"},
    # Alpha111-120
    "alpha111": {"name": "Alpha111", "expression": "-1 * ts_rank(correlation(open, ts_rank(adv60, 10), 10), 20)", "description": "Opening momentum"},
    "alpha112": {"name": "Alpha112", "expression": "rank(ts_argmax(correlation(rank(volume), ts_rank(high, 5), 5), 5)) - 0.5", "description": "Volume-high correlation"},
    "alpha113": {"name": "Alpha113", "expression": "-1 * ts_rank(correlation(high, rank(adv10), 5), 8)", "description": "High price correlation"},
    "alpha114": {"name": "Alpha114", "expression": "rank(ts_argmax(correlation(close, ts_rank(adv30, 4), 4), 12))", "description": "Close correlation"},
    "alpha115": {"name": "Alpha115", "expression": "rank(ts_argmax(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 5)) - 0.5", "description": "Volume-high rank correlation"},
    # Alpha121-130
    "alpha121": {"name": "Alpha121", "expression": "rank(correlation(close, ts_rank(adv40, 20), 15))", "description": "Close vs volume correlation"},
    "alpha122": {"name": "Alpha122", "expression": "rank(ts_argmax(correlation(rank(open), rank(adv25), 10), 10))", "description": "Open volume correlation"},
    "alpha123": {"name": "Alpha123", "expression": "-1 * rank(ts_argmax(correlation(ts_rank(close, 10), ts_rank(adv60, 10), 8), 6))", "description": "Price rank correlation"},
    "alpha124": {"name": "Alpha124", "expression": "rank(ts_argmax(correlation(volume, ts_rank(adv20, 5), 5), 10))", "description": "Volume correlation"},
    "alpha125": {"name": "Alpha125", "expression": "rank(correlation(ts_rank(close, 8), ts_rank(adv50, 20), 8))", "description": "Price rank correlation"},
    # Alpha131-140
    "alpha131": {"name": "Alpha131", "expression": "rank(ts_argmax(correlation(rank(high), rank(adv30), 5), 5)) - 0.5", "description": "High-volume correlation"},
    "alpha132": {"name": "Alpha132", "expression": "rank(correlation(ts_rank(low, 8), ts_rank(adv81, 20), 8))", "description": "Low price correlation"},
    "alpha133": {"name": "Alpha133", "expression": "-1 * rank(correlation(returns, rank(adv50), 5))", "description": "Returns correlation"},
    "alpha134": {"name": "Alpha134", "expression": "rank(ts_argmax(correlation(close, ts_rank(adv10, 3), 5), 4))", "description": "Close volume correlation"},
    "alpha135": {"name": "Alpha135", "expression": "rank(correlation(open, ts_rank(adv40, 10), 10))", "description": "Open volume correlation"},
    # Alpha141-150
    "alpha141": {"name": "Alpha141", "expression": "-1 * ts_rank(ts_argmax(correlation(ts_rank(close, 15), ts_rank(adv30, 5), 10), 4), 16)", "description": "Price momentum"},
    "alpha142": {"name": "Alpha142", "expression": "rank(ts_argmax(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 10))", "description": "Volume-high correlation"},
    "alpha143": {"name": "Alpha143", "expression": "rank(correlation(ts_rank(close, 20), ts_rank(adv40, 10), 10))", "description": "Cross-sectional rank"},
    "alpha144": {"name": "Alpha144", "expression": "-1 * rank(correlation(open, ts_rank(adv60, 5), 10))", "description": "Opening momentum"},
    "alpha145": {"name": "Alpha145", "expression": "rank(ts_argmax(correlation(rank(close), rank(adv81), 10), 10))", "description": "Price volume correlation"},
    # Alpha151-160
    "alpha151": {"name": "Alpha151", "expression": "rank(correlation(ts_rank(close, 10), ts_rank(adv30, 5), 10))", "description": "Close rank correlation"},
    "alpha152": {"name": "Alpha152", "expression": "-1 * ts_rank(correlation(high, rank(adv20), 5), 12)", "description": "High momentum"},
    "alpha153": {"name": "Alpha153", "expression": "rank(ts_argmax(correlation(rank(volume), rank(adv81), 5), 10))", "description": "Volume rank correlation"},
    "alpha154": {"name": "Alpha154", "expression": "rank(correlation(open, ts_rank(adv50, 20), 15))", "description": "Opening correlation"},
    "alpha155": {"name": "Alpha155", "expression": "rank(ts_argmax(correlation(ts_rank(high, 5), ts_rank(adv10, 5), 5), 4))", "description": "High momentum"},
    # Alpha161-170
    "alpha161": {"name": "Alpha161", "expression": "rank(correlation(close, ts_rank(adv81, 10), 10))", "description": "Close volume correlation"},
    "alpha162": {"name": "Alpha162", "expression": "-1 * rank(correlation(ts_rank(high, 5), ts_rank(adv30, 5), 10))", "description": "High rank correlation"},
    "alpha163": {"name": "Alpha163", "expression": "rank(ts_argmax(correlation(rank(low), rank(adv30), 5), 10))", "description": "Low volume correlation"},
    "alpha164": {"name": "Alpha164", "expression": "rank(correlation(ts_rank(close, 8), ts_rank(adv50, 8), 8))", "description": "Close rank correlation"},
    "alpha165": {"name": "Alpha165", "expression": "-1 * ts_rank(correlation(close, rank(adv20), 10), 20)", "description": "Close momentum"},
    # Alpha171-180
    "alpha171": {"name": "Alpha171", "expression": "rank(ts_argmax(correlation(close, ts_rank(adv40, 5), 5), 4))", "description": "Close correlation"},
    "alpha172": {"name": "Alpha172", "expression": "rank(correlation(ts_rank(volume, 5), ts_rank(adv81, 20), 10))", "description": "Volume correlation"},
    "alpha173": {"name": "Alpha173", "expression": "rank(ts_argmax(correlation(ts_rank(high, 8), ts_rank(adv40, 8), 6), 10))", "description": "High momentum"},
    "alpha174": {"name": "Alpha174", "expression": "rank(correlation(open, ts_rank(adv60, 15), 10))", "description": "Opening correlation"},
    "alpha175": {"name": "Alpha175", "expression": "-1 * ts_rank(correlation(ts_rank(close, 10), ts_rank(adv20, 5), 10), 18)", "description": "Price momentum"},
    # Alpha181-191
    "alpha181": {"name": "Alpha181", "expression": "rank(ts_argmax(correlation(rank(volume), rank(adv50), 5), 10))", "description": "Volume rank"},
    "alpha182": {"name": "Alpha182", "expression": "rank(correlation(ts_rank(close, 5), ts_rank(adv60, 5), 5))", "description": "Close rank correlation"},
    "alpha183": {"name": "Alpha183", "expression": "-1 * rank(correlation(open, ts_rank(adv30, 10), 10))", "description": "Opening momentum"},
    "alpha184": {"name": "Alpha184", "expression": "rank(ts_argmax(correlation(ts_rank(low, 8), ts_rank(adv40, 8), 7), 10))", "description": "Low momentum"},
    "alpha185": {"name": "Alpha185", "expression": "rank(correlation(ts_rank(high, 5), ts_rank(adv30, 5), 5))", "description": "High rank correlation"},
    "alpha186": {"name": "Alpha186", "expression": "-1 * ts_rank(correlation(rank(low), rank(adv60), 8), 15)", "description": "Low price momentum"},
    "alpha187": {"name": "Alpha187", "expression": "rank(ts_argmax(correlation(close, ts_rank(adv25, 5), 5), 10))", "description": "Close correlation"},
    "alpha188": {"name": "Alpha188", "expression": "rank(correlation(ts_rank(volume, 8), ts_rank(high, 8), 7))", "description": "Volume-high correlation"},
    "alpha189": {"name": "Alpha189", "expression": "-1 * rank(correlation(ts_rank(close, 5), ts_rank(adv20, 5), 10))", "description": "Price momentum"},
    "alpha190": {"name": "Alpha190", "expression": "rank(ts_argmax(correlation(rank(open), rank(adv50), 10), 10))", "description": "Opening momentum"},
    "alpha191": {"name": "Alpha191", "expression": "rank(correlation(ts_sum(close, 7) / 7, ts_sum(close, 63) / 63, 250)) * rank(correlation(ts_rank(close, 60), ts_rank(adv30, 30), 4))", "description": "Long-term mean reversion"},
}


class BacktestRequest(BaseModel):
    """Request model for backtest endpoint."""

    factor: List[List[float]] = Field(
        ..., description="Factor values, shape (n_days, n_assets)"
    )
    returns: List[List[float]] = Field(
        ..., description="Forward returns, shape (n_days, n_assets)"
    )
    dates: Optional[List[str]] = Field(
        default=None, description="Date labels for each day"
    )
    quantiles: int = Field(default=10, description="Number of quantile groups")
    weight_method: str = Field(
        default="equal", description="Weight method: 'equal' or 'weighted'"
    )
    long_top_n: int = Field(
        default=1, description="Number of top quantile groups to long"
    )
    short_top_n: int = Field(
        default=1, description="Number of bottom quantile groups to short"
    )
    commission_rate: float = Field(default=0.0, description="One-way commission rate")


class NavData(BaseModel):
    """NAV data for chart visualization."""
    model_config = ConfigDict(populate_by_name=True)

    dates: List[str] = Field(alias="dates")
    quantiles: List[List[float]] = Field(alias="quantiles")
    long_short: List[float] = Field(alias="long_short")
    benchmark: List[float] = Field(alias="benchmark")
    ic_series: List[float] = Field(alias="ic_series")
    metrics: Dict[str, float] = Field(alias="metrics")


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": "Alfa.rs Backtest API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.post("/api/backtest", response_model=NavData)
def run_backtest(req: BacktestRequest):
    """
    Run backtest and return NAV data for visualization.
    """
    try:
        import alfars as al

        # Convert to numpy arrays
        factor = np.array(req.factor, dtype=np.float64)
        returns = np.array(req.returns, dtype=np.float64)

        # Validate shapes
        if factor.shape != returns.shape:
            raise HTTPException(
                status_code=400,
                detail=f"Factor shape {factor.shape} must match returns shape {returns.shape}",
            )

        if factor.ndim != 2:
            raise HTTPException(
                status_code=400, detail="Factor and returns must be 2D arrays"
            )

        n_days, n_assets = factor.shape

        # Generate dates if not provided
        dates = req.dates
        if dates is None:
            dates = [
                f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}" for i in range(n_days)
            ]

        # Run backtest
        result = al.quantile_backtest(
            factor=factor,
            returns=returns,
            quantiles=req.quantiles,
            weight_method=req.weight_method,
            long_top_n=req.long_top_n,
            short_top_n=req.short_top_n,
            commission_rate=req.commission_rate,
        )

        # Prepare NAV data
        group_cum_returns = result.group_cum_returns
        n_quantile_days = group_cum_returns.shape[0]

        # Each quantile group's NAV curve
        quantiles_nav = []
        for i in range(req.quantiles):
            nav = 1.0 * (1 + group_cum_returns[:, i])
            quantiles_nav.append(nav.tolist())

        # Long-short NAV (starts at 1.0)
        long_short_nav = 1.0 * (1 + np.cumsum(result.long_short_returns))
        if len(long_short_nav) > 0:
            long_short_nav = long_short_nav.tolist()
        else:
            long_short_nav = [1.0]

        # Benchmark NAV (equal-weighted market)
        mean_returns = np.nanmean(returns, axis=1)
        benchmark_nav = 1.0 * np.cumprod(1 + mean_returns)
        if len(benchmark_nav) > 0:
            benchmark_nav = benchmark_nav.tolist()
        else:
            benchmark_nav = [1.0]

        # IC series
        ic_series = result.ic_series.tolist() if len(result.ic_series) > 0 else []

        # Metrics
        metrics = {
            "long_short_cum_return": float(result.long_short_cum_return),
            "total_return": float(result.total_return),
            "annualized_return": float(result.annualized_return),
            "sharpe_ratio": float(result.sharpe_ratio),
            "max_drawdown": float(result.max_drawdown),
            "turnover": float(result.turnover),
            "ic_mean": float(result.ic_mean),
            "ic_ir": float(result.ic_ir) if result.ic_ir is not None else 0.0,
        }

        # Use dates aligned with the output (n_days - 1 for forward returns)
        nav_dates = (
            dates[1 : n_quantile_days + 1] if len(dates) > n_quantile_days else dates
        )

        return NavData(
            dates=nav_dates,
            quantiles=quantiles_nav,
            long_short=long_short_nav,
            benchmark=benchmark_nav[1:] if len(benchmark_nav) > 1 else benchmark_nav,
            ic_series=ic_series,
            metrics=metrics,
        )

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="alfars package not properly installed. Run 'maturin develop' first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================================================
# Factor Library API
# ============================================================================


@app.get("/api/factors")
def list_factors():
    """
    Return list of predefined factors (Alpha101 etc.)
    """
    return {
        "factors": [
            {
                "id": factor_id,
                "name": data["name"],
                "expression": data["expression"],
                "description": data.get("description", ""),
            }
            for factor_id, data in ALPHA_FACTORS.items()
        ]
    }


class FactorComputeRequest(BaseModel):
    """Request to compute factor values."""

    factor_id: str = Field(..., description="Factor ID to compute")
    n_days: int = Field(default=100, description="Number of trading days")
    n_assets: int = Field(default=50, description="Number of assets")


class FactorComputeResponse(BaseModel):
    """Response with computed factor values."""

    factor_id: str
    factor: List[List[float]]
    returns: List[List[float]]
    dates: List[str]


@app.post("/api/factors/compute", response_model=FactorComputeResponse)
def compute_factor(req: FactorComputeRequest):
    """
    Compute factor values for a predefined factor.
    """
    if req.factor_id not in ALPHA_FACTORS:
        raise HTTPException(status_code=404, detail=f"Factor {req.factor_id} not found")

    n_days = req.n_days
    n_assets = req.n_assets

    # Generate synthetic data with different signal characteristics per factor
    np.random.seed(hash(req.factor_id) % (2**31))

    factor: List[List[float]] = []
    returns: List[List[float]] = []

    # Generate dates
    dates = [
        f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}" for i in range(n_days)
    ]

    # Signal strength varies by factor_id
    signal_strength = min(0.3 + (hash(req.factor_id) % 50) / 100, 0.8)

    for d in range(n_days):
        day_factor: List[float] = []
        day_returns: List[float] = []

        for a in range(n_assets):
            # Random factor value
            f = np.random.random()
            day_factor.append(float(f))

            # Returns with correlation to factor
            signal = (f - 0.5) * signal_strength * 0.02
            noise = (np.random.random() - 0.5) * 0.02
            day_returns.append(float(signal + noise))

        factor.append(day_factor)
        returns.append(day_returns)

    return FactorComputeResponse(
        factor_id=req.factor_id,
        factor=factor,
        returns=returns,
        dates=dates,
    )


# ============================================================================
# GP Mining API
# ============================================================================


class GpMineRequest(BaseModel):
    """Request to run GP factor mining."""

    population_size: int = Field(default=100, description="Population size")
    max_generations: int = Field(default=10, description="Max generations")
    terminal_set: List[str] = Field(
        default=["close", "open", "high", "low", "volume"],
        description="Terminal set (features)",
    )
    function_set: List[str] = Field(
        default=["rank", "ts_mean", "ts_std", "ts_max", "ts_min", "delay", "log", "sign"],
        description="Function set (operators)",
    )
    n_days: int = Field(default=50, description="Number of days for training")
    n_assets: int = Field(default=30, description="Number of assets")
    target_ic: float = Field(default=0.03, description="Target IC threshold")


class GpMineResponse(BaseModel):
    """Response with mined factors."""

    factors: List[Dict[str, Any]]
    best_factor: Dict[str, Any]
    generations: int
    elapsed_time: float


@app.post("/api/gp/mine", response_model=GpMineResponse)
def mine_factors(req: GpMineRequest):
    """
    Run GP factor mining (demo with synthetic results).
    """
    import time

    start_time = time.time()

    # Generate synthetic GP results for demo
    np.random.seed(42)

    # Candidate expressions
    candidate_expressions = [
        "rank(ts_mean(close, 5))",
        "rank(ts_mean(volume, 10))",
        "rank(ts_std(close, 20))",
        "rank(delay(returns, 5))",
        "rank(ts_mean(returns, 10))",
        "rank(ts_max(close, 20) - ts_min(close, 20))",
        "rank((close - open) / open)",
        "rank(ts_mean(rank(close), 10))",
    ]

    factors: List[Dict[str, Any]] = []
    for i, expr in enumerate(candidate_expressions):
        base_ic = 0.02 + np.random.random() * 0.06
        ir = base_ic * np.sqrt(20) * (0.5 + np.random.random() * 0.5)

        factors.append({
            "id": f"gp_{i+1:03d}",
            "name": f"GP Factor {i+1}",
            "expression": expr,
            "ic_mean": float(base_ic),
            "ic_ir": float(ir),
            "fitness": float(base_ic * ir),
        })

    # Sort by fitness
    factors.sort(key=lambda x: x["fitness"], reverse=True)

    elapsed_time = time.time() - start_time

    return GpMineResponse(
        factors=factors,
        best_factor=factors[0] if factors else {},
        generations=req.max_generations,
        elapsed_time=elapsed_time,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
