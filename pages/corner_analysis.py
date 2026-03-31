# Library imports
from pathlib import Path
import json
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from mplsoccer import VerticalPitch
from scipy.ndimage import gaussian_filter

from utils.page_components import add_common_page_elements
from classes.corner_description import CornerDescription
from classes.corner_chat import CornerChat


RAW_EVENTS_DIR = Path("data/raw/statsbomb/events")
RAW_MATCHES_FILE = Path("data/raw/statsbomb/matches/competition_111_season_316.parquet")
RAW_LINEUPS_DIR = Path("data/raw/statsbomb/lineups")
COMPETITION_ID = 111
SEASON_ID = 316
TRANSITION_WINDOW_S = 10
VISUAL_Z_MIN = -2.0
VISUAL_Z_MAX = 2.0
VISUAL_X_PAD = 0.25
HIGH = 0.6
LOW = -0.6
CLEAR_DIFF = 0.75


def _parse_json(value):
    if isinstance(value, str):
        s = value.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return value
    return value


def _to_name(value):
    value = _parse_json(value)
    if isinstance(value, dict):
        return value.get("name")
    return value


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _team_display_name(value):
    value = _parse_json(value)
    if isinstance(value, dict):
        for key in ("home_team_name", "away_team_name", "name"):
            if key in value and value[key]:
                return str(value[key])
    return str(value) if value is not None else "N/A"


def _extract_xy(series: pd.Series, idx: int) -> pd.Series:
    return pd.to_numeric(
        series.apply(
            lambda x: x[idx]
            if isinstance(x, (list, tuple)) and len(x) > idx
            else (
                _parse_json(x)[idx]
                if isinstance(_parse_json(x), list) and len(_parse_json(x)) > idx
                else np.nan
            )
        ),
        errors="coerce",
    )


@st.cache_data(show_spinner=False)
def _load_matches() -> pd.DataFrame:
    if not RAW_MATCHES_FILE.exists():
        return pd.DataFrame()
    dfm = pd.read_parquet(RAW_MATCHES_FILE).copy()
    dfm["match_id"] = pd.to_numeric(dfm.get("match_id"), errors="coerce").astype("Int64")
    dfm = dfm[dfm["match_id"].notna()].copy()
    dfm["home_name"] = dfm["home_team"].apply(_team_display_name)
    dfm["away_name"] = dfm["away_team"].apply(_team_display_name)
    if "match_date" in dfm.columns:
        dfm = dfm.sort_values("match_date", ascending=False)
    return dfm


@st.cache_data(show_spinner=False)
def _load_events(match_ids: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for mid in match_ids:
        path = RAW_EVENTS_DIR / f"match_id={mid}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


@st.cache_data(show_spinner=False)
def _load_lineups(match_ids: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for mid in match_ids:
        path = RAW_LINEUPS_DIR / f"match_id={mid}.parquet"
        if path.exists():
            d = pd.read_parquet(path).copy()
            d["match_id"] = int(mid)
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    d = events.copy()
    if d.empty:
        return d

    if "team_name" not in d.columns and "team" in d.columns:
        d["team_name"] = d["team"].apply(_to_name)
    if "event_type_name" not in d.columns and "type" in d.columns:
        d["event_type_name"] = d["type"].apply(_to_name)
    if "player_name" not in d.columns and "player" in d.columns:
        d["player_name"] = d["player"].apply(_to_name)
    if "pass_type_name" not in d.columns and "pass_type" in d.columns:
        d["pass_type_name"] = d["pass_type"].apply(_to_name)
    if "pass_technique_name" not in d.columns and "pass_technique" in d.columns:
        d["pass_technique_name"] = d["pass_technique"].apply(_to_name)
    if "pass_recipient_name" not in d.columns and "pass_recipient" in d.columns:
        d["pass_recipient_name"] = d["pass_recipient"].apply(_to_name)
    if "pass_outcome_name" not in d.columns and "pass_outcome" in d.columns:
        d["pass_outcome_name"] = d["pass_outcome"].apply(_to_name)
    if "shot_outcome_name" not in d.columns and "shot_outcome" in d.columns:
        d["shot_outcome_name"] = d["shot_outcome"].apply(_to_name)
    if "play_pattern_name" not in d.columns and "play_pattern" in d.columns:
        d["play_pattern_name"] = d["play_pattern"].apply(_to_name)
    if "position_name" not in d.columns and "position" in d.columns:
        d["position_name"] = d["position"].apply(_to_name)

    for col in [
        "team_name",
        "event_type_name",
        "player_name",
        "pass_type_name",
        "pass_technique_name",
        "pass_recipient_name",
        "pass_outcome_name",
        "shot_outcome_name",
        "play_pattern_name",
        "position_name",
    ]:
        if col in d.columns:
            d[col] = d[col].apply(_to_name)

    if "location" in d.columns:
        d["location_x"] = _extract_xy(d["location"], 0)
        d["location_y"] = _extract_xy(d["location"], 1)
    if "pass_end_location" in d.columns:
        d["pass_end_x"] = _extract_xy(d["pass_end_location"], 0)
        d["pass_end_y"] = _extract_xy(d["pass_end_location"], 1)

    for bool_col in ["pass_shot_assist", "pass_goal_assist", "pass_inswinging", "pass_outswinging"]:
        if bool_col not in d.columns:
            d[bool_col] = False
        else:
            d[bool_col] = d[bool_col].apply(_truthy)

    d["minute"] = pd.to_numeric(d.get("minute", 0), errors="coerce").fillna(0).astype(int)
    d["second"] = pd.to_numeric(d.get("second", 0), errors="coerce").fillna(0.0)
    d["time_seconds"] = d["minute"] * 60.0 + d["second"]
    d["index"] = pd.to_numeric(d.get("index", 0), errors="coerce").fillna(0).astype(int)
    return d


def _prepare_offensive_corners(events: pd.DataFrame, team_name: str) -> pd.DataFrame:
    corners = events[
        (events["team_name"].astype(str) == str(team_name))
        & (events["event_type_name"].astype(str) == "Pass")
        & (events["pass_type_name"].astype(str) == "Corner")
    ].copy()
    if corners.empty:
        return corners

    corners = corners.dropna(subset=["location_x", "location_y", "pass_end_x", "pass_end_y"]).copy()
    corners["side"] = np.where(corners["location_y"] <= 40.0, "Left", "Right")
    corners["pass_technique_name"] = corners["pass_technique_name"].fillna("Short")
    corners["is_short"] = corners["pass_technique_name"].astype(str).str.lower().eq("short")
    corners["is_inswing"] = corners["pass_inswinging"].astype(bool)
    corners["is_outswing"] = corners["pass_outswinging"].astype(bool)
    dy = (corners["location_y"] - corners["pass_end_y"]).abs()
    corners["target_zone"] = np.select(
        [
            corners["is_short"],
            (~corners["is_short"]) & (dy < 37.0),
            (~corners["is_short"]) & (dy > 43.0),
        ],
        ["short", "near_post", "far_post"],
        default="central",
    )
    corners["depth_zone"] = np.select(
        [
            (~corners["is_short"]) & (corners["pass_end_x"] >= 114.0),
            (~corners["is_short"]) & (corners["pass_end_x"] < 114.0),
        ],
        ["six_yard", "deeper"],
        default="short",
    )

    if "pass_assisted_shot_id" in corners.columns and "id" in events.columns:
        shots = events[events["event_type_name"].astype(str) == "Shot"][["id", "shot_statsbomb_xg", "shot_outcome_name"]].copy()
        shots["shot_statsbomb_xg"] = pd.to_numeric(shots["shot_statsbomb_xg"], errors="coerce").fillna(0.0)
        shot_xg = shots.set_index("id")["shot_statsbomb_xg"] if not shots.empty else pd.Series(dtype=float)
        shot_goal = (
            shots.set_index("id")["shot_outcome_name"].astype(str).str.lower().eq("goal")
            if not shots.empty
            else pd.Series(dtype=bool)
        )
        corners["assist_xg"] = corners["pass_assisted_shot_id"].map(shot_xg).fillna(0.0)
        corners["is_goal_assist"] = corners["pass_assisted_shot_id"].map(shot_goal).fillna(False).astype(bool)
    else:
        corners["assist_xg"] = 0.0
        corners["is_goal_assist"] = False

    corners["corner_shot_assist"] = corners["pass_shot_assist"].astype(bool)
    corners["corner_goal_assist"] = corners["is_goal_assist"].astype(bool)
    return corners


def _prepare_defensive_corners(events: pd.DataFrame, team_name: str) -> pd.DataFrame:
    opp = events[
        (events["team_name"].astype(str) != str(team_name))
        & (events["event_type_name"].astype(str) == "Pass")
        & (events["pass_type_name"].astype(str) == "Corner")
    ].copy()
    if opp.empty:
        return opp
    opp = opp.dropna(subset=["location_x", "location_y", "pass_end_x", "pass_end_y"]).copy()
    opp["side"] = np.where(opp["location_y"] <= 40.0, "Left", "Right")
    return opp


def _build_phase_table(events: pd.DataFrame, corner_only: bool) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["team_name", "phase1_xg", "phase2_xg"])

    shots = events[events["event_type_name"].astype(str) == "Shot"].copy()
    if shots.empty:
        return pd.DataFrame(columns=["team_name", "phase1_xg", "phase2_xg"])

    shots["shot_xg"] = pd.to_numeric(shots.get("shot_statsbomb_xg"), errors="coerce").fillna(0.0)
    shots["set_piece_phase"] = pd.to_numeric(shots.get("set_piece_phase"), errors="coerce")
    shots["play_pattern_name"] = shots.get("play_pattern_name", "").astype(str)

    if corner_only:
        shots = shots[shots["play_pattern_name"] == "From Corner"].copy()
    else:
        shots = shots[shots["set_piece_phase"].notna()].copy()

    if shots.empty:
        return pd.DataFrame(columns=["team_name", "phase1_xg", "phase2_xg"])

    p1 = (
        shots[shots["set_piece_phase"] == 1.0]
        .groupby("team_name", dropna=True)["shot_xg"]
        .sum()
        .rename("phase1_xg")
    )
    p2 = (
        shots[shots["set_piece_phase"] == 2.0]
        .groupby("team_name", dropna=True)["shot_xg"]
        .sum()
        .rename("phase2_xg")
    )

    table = (
        pd.concat([p1, p2], axis=1)
        .fillna(0.0)
        .reset_index()
        .rename(columns={"index": "team_name"})
    )
    if "team_name" not in table.columns:
        table["team_name"] = []
    return table


def _timestamp_to_seconds(ts) -> float:
    if ts is None:
        return 0.0
    txt = str(ts)
    if txt.lower() in {"none", "nan", ""}:
        return 0.0
    try:
        hh, mm, ss = txt.split(":")
        return float(hh) * 3600.0 + float(mm) * 60.0 + float(ss)
    except Exception:
        return 0.0


def _period_offset_seconds(period: int) -> float:
    return {
        1: 0.0,
        2: 45.0 * 60.0,
        3: 90.0 * 60.0,
        4: 105.0 * 60.0,
    }.get(int(period), 0.0)


def _compute_player_minutes(lineups: pd.DataFrame, events: pd.DataFrame, team_name: str) -> pd.DataFrame:
    if lineups.empty:
        return pd.DataFrame(columns=["player_name", "player_minutes"])

    d = lineups.copy()
    d["team_name"] = d.get("team_name", "").astype(str)
    d = d[d["team_name"] == str(team_name)].copy()
    if d.empty:
        return pd.DataFrame(columns=["player_name", "player_minutes"])

    if events.empty or "match_id" not in events.columns:
        match_end_map = {}
    else:
        e = events.copy()
        e["time_seconds"] = pd.to_numeric(e.get("time_seconds"), errors="coerce").fillna(0.0)
        match_end_map = e.groupby("match_id", dropna=True)["time_seconds"].max().to_dict()

    records = []
    for _, row in d.iterrows():
        player = str(row.get("player_name", "")).strip()
        if player == "":
            continue
        mid = row.get("match_id")
        match_end = float(match_end_map.get(mid, 95.0 * 60.0))
        positions = _parse_json(row.get("positions"))
        if not isinstance(positions, list):
            positions = []
        total_sec = 0.0
        for p in positions:
            if not isinstance(p, dict):
                continue
            fp = int(p.get("from_period") or 1)
            tp = int(p.get("to_period") or fp)
            start = _period_offset_seconds(fp) + _timestamp_to_seconds(p.get("from"))
            to_val = p.get("to")
            if to_val in (None, "", "None"):
                end = match_end
            else:
                end = _period_offset_seconds(tp) + _timestamp_to_seconds(to_val)
            if np.isfinite(start) and np.isfinite(end):
                total_sec += max(0.0, end - start)
        records.append({"player_name": player, "player_minutes": total_sec / 60.0})

    if not records:
        return pd.DataFrame(columns=["player_name", "player_minutes"])
    out = pd.DataFrame(records)
    out = out.groupby("player_name", dropna=True)["player_minutes"].sum().reset_index()
    out["player_minutes"] = out["player_minutes"].round(1)
    return out


def _build_hops_player_table(events: pd.DataFrame, lineups: pd.DataFrame, team_name: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["player_name", "hops_rating", "hops_samples", "opp_quality", "player_minutes"])

    d = events.copy()
    d = d[d["team_name"].astype(str) == str(team_name)].copy()
    d = d[d["event_type_name"].astype(str) == "Duel"].copy()
    if "position_name" in d.columns:
        d = d[d["position_name"].astype(str) != "Goalkeeper"].copy()
    d["hops_rating"] = pd.to_numeric(d.get("duel_hops_rating"), errors="coerce")
    d["hops_opp_rating"] = pd.to_numeric(d.get("duel_hops_opponent_rating"), errors="coerce")
    d = d[d["hops_rating"].notna()].copy()
    d = d[d["player_name"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["player_name", "hops_rating", "hops_samples", "opp_quality", "player_minutes"])

    table = (
        d.groupby("player_name", dropna=True)
        .agg(
            hops_rating=("hops_rating", "mean"),
            hops_samples=("hops_rating", "size"),
            opp_quality=("hops_opp_rating", "mean"),
        )
        .reset_index()
        .sort_values(["hops_rating", "hops_samples"], ascending=[False, False])
    )
    table["hops_rating"] = table["hops_rating"].round(3)
    table["opp_quality"] = table["opp_quality"].fillna(0.0).round(3)
    mins = _compute_player_minutes(lineups, events, team_name)
    if not mins.empty:
        table = table.merge(mins, on="player_name", how="left")
    else:
        table["player_minutes"] = 0.0
    table["player_minutes"] = pd.to_numeric(table["player_minutes"], errors="coerce").fillna(0.0).round(1)
    return table


def _build_hops_pairs(left_rank: pd.DataFrame, right_rank: pd.DataFrame, max_players: int | None = None) -> pd.DataFrame:
    if left_rank.empty or right_rank.empty:
        return pd.DataFrame()
    l = left_rank.sort_values("hops_rating", ascending=False).reset_index(drop=True)
    r = right_rank.sort_values("hops_rating", ascending=True).reset_index(drop=True)
    if max_players is not None:
        l = l.head(int(max_players))
        r = r.head(int(max_players))
    n = min(len(l), len(r))
    if n == 0:
        return pd.DataFrame()
    pairs = pd.DataFrame(
        {
            "left_player": l.loc[: n - 1, "player_name"].tolist(),
            "left_rating": l.loc[: n - 1, "hops_rating"].astype(float).tolist(),
            "right_player": r.loc[: n - 1, "player_name"].tolist(),
            "right_rating": r.loc[: n - 1, "hops_rating"].astype(float).tolist(),
        }
    )
    pairs["delta"] = pairs["left_rating"] - pairs["right_rating"]
    return pairs


def _plot_hops_mismatch(pairs: pd.DataFrame, left_team: str, right_team: str):
    fig_h = max(4.8, 0.52 * max(len(pairs), 1) + 1.4)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    if pairs.empty:
        ax.text(0.5, 0.5, "Not enough HOPS data to build mismatch.", ha="center", va="center")
        ax.axis("off")
        return fig

    y = np.arange(len(pairs))
    ax.barh(y, -pairs["left_rating"], color="#cc001b", alpha=0.9)
    ax.barh(y, pairs["right_rating"], color="#1198c4", alpha=0.9)
    ax.axvline(0, color="#111827", linewidth=1.2)

    lim = max(float(pairs["left_rating"].max()), float(pairs["right_rating"].max()), 0.7) + 0.05
    ax.set_xlim(-lim * 1.6, lim * 1.6)
    for i, row in pairs.iterrows():
        ax.text(-lim * 1.03, i, str(row["left_player"]), ha="right", va="center", fontsize=10)
        ax.text(lim * 1.03, i, str(row["right_player"]), ha="left", va="center", fontsize=10)
        ax.text(-0.03, i, f"{float(row['left_rating']):.2f}", ha="right", va="center", fontsize=9, color="white")
        ax.text(0.03, i, f"{float(row['right_rating']):.2f}", ha="left", va="center", fontsize=9, color="white")

    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"HOPS Mismatch: {left_team} (strongest) vs {right_team} (weakest)")
    return fig


def _plot_phase_scatter_highlight(
    phase_df: pd.DataFrame,
    title: str,
    selected_team: str,
    rival_team: str,
):
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    if phase_df.empty:
        ax.text(0.5, 0.5, "No set-piece phase data available.", ha="center", va="center")
        ax.axis("off")
        return fig

    base = phase_df[
        (phase_df["team_name"].astype(str) != str(selected_team))
        & (phase_df["team_name"].astype(str) != str(rival_team))
    ]
    team_df = phase_df[phase_df["team_name"].astype(str) == str(selected_team)]
    rival_df = phase_df[phase_df["team_name"].astype(str) == str(rival_team)]

    if not base.empty:
        ax.scatter(base["phase1_xg"], base["phase2_xg"], s=60, color="#9ca3af", edgecolor="#6b7280", alpha=0.85)
    if not team_df.empty:
        ax.scatter(team_df["phase1_xg"], team_df["phase2_xg"], s=130, color="#cc001b", edgecolor="#111827", linewidth=0.9)
    if not rival_df.empty:
        ax.scatter(rival_df["phase1_xg"], rival_df["phase2_xg"], s=130, color="#003275", edgecolor="#111827", linewidth=0.9)

    for _, row in base.iterrows():
        ax.text(float(row["phase1_xg"]), float(row["phase2_xg"]) + 0.01, str(row["team_name"]), fontsize=8, ha="center")
    for _, row in team_df.iterrows():
        ax.text(float(row["phase1_xg"]), float(row["phase2_xg"]) + 0.01, str(row["team_name"]), fontsize=9, ha="center", color="#cc001b", fontweight="bold")
    for _, row in rival_df.iterrows():
        ax.text(float(row["phase1_xg"]), float(row["phase2_xg"]) + 0.01, str(row["team_name"]), fontsize=9, ha="center", color="#003275", fontweight="bold")

    ax.set_xlabel("Phase 1 xG")
    ax.set_ylabel("Phase 2 xG")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    return fig


def _short_corner_possession_analysis(
    events: pd.DataFrame,
    corners: pd.DataFrame,
    team_name: str,
    window_s: int = 5,
    shot_window_s: int = 20,
):
    if corners.empty:
        return pd.DataFrame(), {"Shot <=20s": 0, "No Shot <=20s": 0}

    short = corners[corners["is_short"]].copy()
    if short.empty:
        return pd.DataFrame(), {"Shot <=20s": 0, "No Shot <=20s": 0}

    rows = []
    shot_20_yes = 0
    shot_20_no = 0

    for _, c in short.iterrows():
        mid = c.get("match_id")
        pos = c.get("possession")
        idx = int(c.get("index", 0))
        t0 = float(c.get("time_seconds", 0.0))

        chain = events[
            (events["match_id"] == mid)
            & (events["possession"] == pos)
            & (events["index"] >= idx)
        ].copy()
        if chain.empty:
            continue

        w5 = chain[(chain["time_seconds"] >= t0) & (chain["time_seconds"] <= t0 + float(window_s))].copy()
        w20 = chain[(chain["time_seconds"] >= t0) & (chain["time_seconds"] <= t0 + float(shot_window_s))].copy()
        if w5.empty:
            continue

        team5 = w5[w5["team_name"].astype(str) == str(team_name)].copy()
        opp5 = w5[w5["team_name"].astype(str) != str(team_name)].copy()
        any_shot_5 = bool((team5["event_type_name"].astype(str) == "Shot").any())
        any_foul_won_5 = bool((team5["event_type_name"].astype(str) == "Foul Won").any())
        any_out_5 = bool(team5.get("out", pd.Series(False, index=team5.index)).apply(_truthy).any())
        any_opp_touch_5 = len(opp5) > 0

        loc = pd.to_numeric(team5.get("location_x"), errors="coerce")
        prog = float(loc.max() - float(c.get("location_x", 0.0))) if not loc.dropna().empty else 0.0
        prog = float(prog) if np.isfinite(prog) else 0.0

        if any_shot_5:
            outcome = "Shot"
        elif any_foul_won_5:
            outcome = "Won Foul"
        elif any_out_5:
            outcome = "Ball Out"
        elif any_opp_touch_5:
            outcome = "Lost Possession"
        else:
            outcome = "Kept Possession"

        any_shot_20 = bool(
            (
                w20[w20["team_name"].astype(str) == str(team_name)]["event_type_name"]
                .astype(str)
                .eq("Shot")
            ).any()
        )
        if any_shot_20:
            shot_20_yes += 1
        else:
            shot_20_no += 1

        rows.append(
            {
                "match_id": mid,
                "corner_index": idx,
                "outcome_5s": outcome,
                "progression_x_5s": prog,
                "shot_20s": any_shot_20,
            }
        )

    return pd.DataFrame(rows), {"Shot <=20s": int(shot_20_yes), "No Shot <=20s": int(shot_20_no)}


def _draw_short_corner_panel(ax_outcome, ax_progression, ax_shot20, short_df: pd.DataFrame, shot20_counts: dict):
    ax_outcome.set_title("Short-corner possession (first 5s)", fontsize=10.5, fontweight="bold")
    ax_progression.set_title("Progression in first 5s (x-axis)", fontsize=10.5, fontweight="bold")
    ax_shot20.set_title("Shot after short corner (<=20s)", fontsize=10.5, fontweight="bold")

    if short_df.empty:
        for ax in [ax_outcome, ax_progression, ax_shot20]:
            ax.text(0.5, 0.5, "No short-corner data", ha="center", va="center", transform=ax.transAxes, color="#6b7280")
            ax.axis("off")
        return

    order = ["Kept Possession", "Lost Possession", "Shot", "Won Foul", "Ball Out"]
    outcome = short_df["outcome_5s"].value_counts().reindex(order).fillna(0)
    ax_outcome.bar(outcome.index, outcome.values, color=["#10b981", "#ef4444", "#f59e0b", "#3b82f6", "#6b7280"])
    ax_outcome.tick_params(axis="x", rotation=15, labelsize=8)
    ax_outcome.grid(axis="y", alpha=0.2)

    vals = pd.to_numeric(short_df["progression_x_5s"], errors="coerce").fillna(0.0)
    ax_progression.hist(vals, bins=8, color="#2563eb", alpha=0.85, edgecolor="#1e3a8a")
    ax_progression.axvline(float(vals.mean()), color="#111827", linewidth=1.8, linestyle="--")
    ax_progression.text(float(vals.mean()), ax_progression.get_ylim()[1] * 0.92, f"Mean {vals.mean():.1f}", fontsize=8, ha="left")
    ax_progression.grid(axis="y", alpha=0.2)

    labels = list(shot20_counts.keys())
    values = list(shot20_counts.values())
    ax_shot20.bar(labels, values, color=["#cc001b", "#94a3b8"])
    for i, v in enumerate(values):
        ax_shot20.text(i, v + 0.1, str(int(v)), ha="center", fontsize=9)
    ax_shot20.grid(axis="y", alpha=0.2)


def _label_target_zone(corners: pd.DataFrame) -> pd.DataFrame:
    if corners.empty:
        return corners
    data = corners.copy()
    dy = (data["location_y"] - data["pass_end_y"]).abs()
    data["target_zone"] = np.select(
        [
            data["is_short"],
            (~data["is_short"]) & (dy < 37.0),
            (~data["is_short"]) & (dy > 43.0),
        ],
        ["short", "near_post", "far_post"],
        default="central",
    )
    return data


def _compute_transition_risk_per_corner(
    events: pd.DataFrame,
    corners: pd.DataFrame,
    team_name: str,
    window_s: int = TRANSITION_WINDOW_S,
) -> pd.Series:
    if corners.empty:
        return pd.Series(dtype=float, index=corners.index)
    required = {"match_id", "index", "time_seconds", "team_name", "possession"}
    if any(c not in events.columns for c in required):
        return pd.Series(0.0, index=corners.index)

    d = events.copy()
    d["obv_against_net_num"] = pd.to_numeric(d.get("obv_against_net"), errors="coerce").fillna(0.0)
    d["index"] = pd.to_numeric(d["index"], errors="coerce").fillna(0).astype(int)
    d["time_seconds"] = pd.to_numeric(d["time_seconds"], errors="coerce").fillna(0.0)

    risk_values = []
    for _, c in corners.iterrows():
        mid = c.get("match_id")
        idx = int(pd.to_numeric(c.get("index"), errors="coerce") if pd.notna(c.get("index")) else 0)
        t0 = float(pd.to_numeric(c.get("time_seconds"), errors="coerce") if pd.notna(c.get("time_seconds")) else 0.0)

        sub = d[
            (d["match_id"] == mid)
            & (d["index"] > idx)
            & (d["time_seconds"] >= t0)
            & (d["time_seconds"] <= t0 + float(window_s))
        ].copy()
        if sub.empty:
            risk_values.append(0.0)
            continue

        opp_sub = sub[sub["team_name"].astype(str) != str(team_name)].copy()
        if opp_sub.empty:
            risk_values.append(0.0)
            continue

        first_opp = opp_sub.sort_values("index").iloc[0]
        opp_possession = first_opp.get("possession")
        transition = opp_sub[opp_sub["possession"] == opp_possession].copy()
        if transition.empty:
            transition = opp_sub

        # Keep only threat against us (positive side).
        risk = float(transition["obv_against_net_num"].clip(lower=0.0).sum())
        risk_values.append(risk)

    return pd.Series(risk_values, index=corners.index, dtype=float)


@st.cache_data(show_spinner=False)
def _build_context_table(all_events_raw: pd.DataFrame, all_matches: pd.DataFrame) -> pd.DataFrame:
    events = _prepare_events(all_events_raw)
    if events.empty:
        return pd.DataFrame()

    teams = sorted(events["team_name"].dropna().astype(str).unique().tolist())
    if not teams:
        return pd.DataFrame()

    match_long = pd.concat(
        [
            all_matches[["match_id", "home_name"]].rename(columns={"home_name": "team_name"}),
            all_matches[["match_id", "away_name"]].rename(columns={"away_name": "team_name"}),
        ],
        ignore_index=True,
    )
    match_counts = (
        match_long.groupby("team_name", dropna=True)["match_id"]
        .nunique()
        .rename("matches")
    )

    rows = []
    for team in teams:
        c = _prepare_offensive_corners(events, team)
        c = _label_target_zone(c)
        n = len(c)
        if n == 0:
            rows.append(
                {
                    "team_name": team,
                    "corners": 0,
                    "corners_per_match": 0.0,
                    "left_pct": 0.0,
                    "short_pct": 0.0,
                    "inswing_pct": 0.0,
                    "outswing_pct": 0.0,
                    "near_post_pct": 0.0,
                    "central_pct": 0.0,
                    "far_post_pct": 0.0,
                    "shot_assist_rate": 0.0,
                    "goal_assist_rate": 0.0,
                    "assist_xg_per_corner": 0.0,
                    "corner_phase1_xg": 0.0,
                    "corner_phase2_xg": 0.0,
                    "corner_phase1_xg_per_match": 0.0,
                    "corner_phase2_xg_per_match": 0.0,
                    "risk_obv_against_per_corner": 0.0,
                    "risk_reward_index": 0.0,
                }
            )
            continue

        matches = int(match_counts.get(team, 0))
        corners_per_match = float(n / matches) if matches > 0 else 0.0
        transition_risk = _compute_transition_risk_per_corner(
            events=events,
            corners=c,
            team_name=team,
            window_s=TRANSITION_WINDOW_S,
        )
        risk = float(transition_risk.mean()) if not transition_risk.empty else 0.0
        reward = float(c["assist_xg"].mean()) if "assist_xg" in c.columns else 0.0
        team_shots = events[
            (events["team_name"].astype(str) == str(team))
            & (events["event_type_name"].astype(str) == "Shot")
            & (events["play_pattern_name"].astype(str) == "From Corner")
        ].copy()
        team_shots["set_piece_phase"] = pd.to_numeric(team_shots.get("set_piece_phase"), errors="coerce")
        team_shots["shot_xg"] = pd.to_numeric(team_shots.get("shot_statsbomb_xg"), errors="coerce").fillna(0.0)
        corner_phase1_xg = float(team_shots.loc[team_shots["set_piece_phase"] == 1.0, "shot_xg"].sum())
        corner_phase2_xg = float(team_shots.loc[team_shots["set_piece_phase"] == 2.0, "shot_xg"].sum())
        phase1_per_match = corner_phase1_xg / matches if matches > 0 else 0.0
        phase2_per_match = corner_phase2_xg / matches if matches > 0 else 0.0

        rows.append(
            {
                "team_name": team,
                "corners": int(n),
                "corners_per_match": corners_per_match,
                "left_pct": float((c["side"] == "Left").mean() * 100.0),
                "short_pct": float(c["is_short"].mean() * 100.0),
                "inswing_pct": float(c["is_inswing"].mean() * 100.0),
                "outswing_pct": float(c["is_outswing"].mean() * 100.0),
                "near_post_pct": float((c["target_zone"] == "near_post").mean() * 100.0),
                "central_pct": float((c["target_zone"] == "central").mean() * 100.0),
                "far_post_pct": float((c["target_zone"] == "far_post").mean() * 100.0),
                "shot_assist_rate": float(c["pass_shot_assist"].mean() * 100.0),
                "goal_assist_rate": float(c["is_goal_assist"].mean() * 100.0),
                "assist_xg_per_corner": reward,
                "corner_phase1_xg": corner_phase1_xg,
                "corner_phase2_xg": corner_phase2_xg,
                "corner_phase1_xg_per_match": phase1_per_match,
                "corner_phase2_xg_per_match": phase2_per_match,
                "risk_obv_against_per_corner": float(risk),
                "risk_reward_index": float(reward - risk),
            }
        )

    return pd.DataFrame(rows).sort_values("team_name").reset_index(drop=True)


def _extract_hops_rating_from_skills(skills_value) -> float:
    s = _parse_json(skills_value)
    if isinstance(s, dict):
        hops = s.get("HOPS")
        if isinstance(hops, dict):
            return float(pd.to_numeric(hops.get("rating"), errors="coerce"))
    return float("nan")


def _minutes_from_positions(positions_value, default_end_min: float = 95.0) -> float:
    positions = _parse_json(positions_value)
    if not isinstance(positions, list):
        return 0.0
    total_sec = 0.0
    for p in positions:
        if not isinstance(p, dict):
            continue
        fp = int(p.get("from_period") or 1)
        tp = int(p.get("to_period") or fp)
        start = _period_offset_seconds(fp) + _timestamp_to_seconds(p.get("from"))
        to_val = p.get("to")
        if to_val in (None, "", "None"):
            end = _period_offset_seconds(tp) + default_end_min * 60.0
        else:
            end = _period_offset_seconds(tp) + _timestamp_to_seconds(to_val)
        if np.isfinite(start) and np.isfinite(end):
            total_sec += max(0.0, end - start)
    return total_sec / 60.0


def _build_team_hops_metrics(lineups_all: pd.DataFrame) -> pd.DataFrame:
    if lineups_all.empty:
        return pd.DataFrame(columns=["team_name", "team_top5_hops", "team_hops_weighted"])

    d = lineups_all.copy()
    d["team_name"] = d.get("team_name", "").astype(str)
    d["player_name"] = d.get("player_name", "").astype(str)
    d["hops_rating"] = d.get("skills").apply(_extract_hops_rating_from_skills)
    d["minutes"] = d.get("positions").apply(_minutes_from_positions)
    d = d[d["team_name"] != ""].copy()
    d = d[d["player_name"] != ""].copy()
    d = d[np.isfinite(d["hops_rating"])].copy()
    if d.empty:
        return pd.DataFrame(columns=["team_name", "team_top5_hops", "team_hops_weighted"])

    # Aggregate player level minutes and hops inside each team
    p = (
        d.groupby(["team_name", "player_name"], dropna=True)
        .agg(
            player_minutes=("minutes", "sum"),
            player_hops=("hops_rating", "mean"),
        )
        .reset_index()
    )
    p["player_minutes"] = pd.to_numeric(p["player_minutes"], errors="coerce").fillna(0.0)
    p["player_hops"] = pd.to_numeric(p["player_hops"], errors="coerce")

    rows = []
    for team, g in p.groupby("team_name", dropna=True):
        g = g.sort_values("player_hops", ascending=False).copy()
        top5 = g.head(5)
        team_top5_hops = float(top5["player_hops"].mean()) if not top5.empty else 0.0
        w = g[g["player_minutes"] > 0].copy()
        if w.empty:
            team_hops_weighted = float(g["player_hops"].mean()) if not g.empty else 0.0
        else:
            team_hops_weighted = float(np.average(w["player_hops"], weights=w["player_minutes"]))
        rows.append(
            {
                "team_name": str(team),
                "team_top5_hops": round(team_top5_hops, 4),
                "team_hops_weighted": round(team_hops_weighted, 4),
            }
        )
    return pd.DataFrame(rows)


def _zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        mu = x.mean()
        sd = x.std(ddof=0)
        if pd.isna(sd) or float(sd) == 0.0:
            out[c] = 0.0
        else:
            out[c] = (x - mu) / sd
    return out


def _plot_correlation_heatmap(corr: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    if corr.empty:
        ax.text(0.5, 0.5, "No correlation data available.", ha="center", va="center")
        ax.axis("off")
        return fig

    mat = corr.values
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=9)
    fig.tight_layout()
    return fig


def _plot_weighted_ranking(rank_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    if rank_df.empty:
        ax.text(0.5, 0.5, "No ranking data.", ha="center", va="center")
        ax.axis("off")
        return fig
    d = rank_df.sort_values("composite_score", ascending=True).copy()
    colors = ["#cc001b" if i == len(d) - 1 else "#94a3b8" for i in range(len(d))]
    ax.barh(d["team_name"], d["composite_score"], color=colors)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Composite score (weighted z-score)")
    ax.grid(axis="x", alpha=0.2)
    for i, v in enumerate(d["composite_score"].tolist()):
        ax.text(float(v), i, f" {v:.2f}", va="center", fontsize=8.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _build_quality_scores(context_df: pd.DataFrame) -> pd.DataFrame:
    if context_df.empty:
        return pd.DataFrame()

    d = context_df.copy()
    needed = [
        "near_post_pct",
        "far_post_pct",
        "assist_xg_per_corner",
        "goal_assist_rate",
        "shot_assist_rate",
        "corner_phase1_xg_per_match",
        "corner_phase2_xg_per_match",
        "inswing_pct",
        "short_pct",
        "outswing_pct",
        "team_top5_hops",
        "risk_obv_against_per_corner",
    ]
    for c in needed:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Raw quality features from your definitions
    d["q_target_zone"] = d["near_post_pct"] - d["far_post_pct"]
    d["q_shot_goal_threat"] = (
        d["assist_xg_per_corner"] * 0.55
        + d["goal_assist_rate"] * 0.15
        + d["shot_assist_rate"] * 0.30
    )
    # Positive side -> stronger phase 1 profile, Negative side -> stronger phase 2 profile.
    d["q_threat_style"] = d["corner_phase1_xg_per_match"] - d["corner_phase2_xg_per_match"]
    # Positive side -> short-corner dominant profile, Negative side -> direct-corner dominant profile.
    d["q_short_vs_direct"] = d["short_pct"] - (100.0 - d["short_pct"])
    # Positive side -> open-foot dominant delivery (outswing), Negative side -> closed-foot dominant delivery (inswing).
    d["q_open_vs_closed"] = d["outswing_pct"] - d["inswing_pct"]
    d["q_header_threat"] = d["team_top5_hops"]
    # Lower transition risk is better -> invert sign
    d["q_transition_risk"] = -d["risk_obv_against_per_corner"]

    q_cols = [
        "q_target_zone",
        "q_shot_goal_threat",
        "q_threat_style",
        "q_short_vs_direct",
        "q_open_vs_closed",
        "q_header_threat",
        "q_transition_risk",
    ]
    # Keep partial availability: each quality can be plotted with its own non-null teams.
    q_df = d[["team_name"] + q_cols].copy().reset_index(drop=True)
    if q_df.empty:
        return q_df
    q_df = _zscore_frame(q_df, q_cols)
    return q_df


def _label_quality(z, high: float = HIGH, low: float = LOW) -> str:
    try:
        v = float(pd.to_numeric(z, errors="coerce"))
    except Exception:
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    if v >= float(high):
        return "high"
    if v <= float(low):
        return "low"
    return "average"


def _compare_quality(z_team, z_opp, clear_diff: float = CLEAR_DIFF) -> str:
    try:
        diff = float(pd.to_numeric(z_team, errors="coerce")) - float(pd.to_numeric(z_opp, errors="coerce"))
    except Exception:
        return "unknown"
    if not np.isfinite(diff):
        return "unknown"
    if diff >= float(clear_diff):
        return "clear_advantage"
    if diff <= -float(clear_diff):
        return "clear_disadvantage"
    return "balanced"


def _target_zone_style_label(near_pct, central_pct, far_pct) -> str:
    near = float(pd.to_numeric(near_pct, errors="coerce")) if pd.notna(near_pct) else np.nan
    central = float(pd.to_numeric(central_pct, errors="coerce")) if pd.notna(central_pct) else np.nan
    far = float(pd.to_numeric(far_pct, errors="coerce")) if pd.notna(far_pct) else np.nan
    if not all(np.isfinite(v) for v in [near, central, far]):
        return "unknown"

    # Target zone is orientation, not quality: near-post, far-post, or mixed/central emphasis.
    diff = near - far
    if central >= max(near, far) and central >= 30.0:
        return "central-oriented / mixed"
    if abs(diff) < 8.0 and central >= 25.0:
        return "mixed profile with central usage"
    if diff >= 8.0:
        return "near-post oriented"
    if diff <= -8.0:
        return "far-post oriented"
    return "balanced near/far profile"


def _quality_level_text(quality_key: str, z_value) -> str:
    lvl = _label_quality(z_value, high=HIGH, low=LOW)
    mapping = {
        "q_shot_goal_threat": {"high": "strong threat output", "average": "average threat output", "low": "limited threat output"},
        "q_threat_style": {"high": "initial-delivery dominant", "average": "balanced phase profile", "low": "second-ball dominant"},
        "q_short_vs_direct": {"high": "short-corner leaning", "average": "mixed short/direct", "low": "direct-corner leaning"},
        "q_open_vs_closed": {"high": "open-foot leaning", "average": "mixed open/closed", "low": "closed-foot leaning"},
        "q_header_threat": {"high": "strong aerial threat", "average": "average aerial threat", "low": "limited aerial threat"},
        "q_transition_risk": {"high": "strong transition protection", "average": "average transition protection", "low": "higher transition exposure"},
    }
    return mapping.get(quality_key, {}).get(lvl, "unknown")


def _plot_quality_reference(quality_df: pd.DataFrame, selected_team: str):
    fig, ax = plt.subplots(figsize=(12, 7))
    bg = "#062f27"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    if quality_df.empty:
        ax.text(0.5, 0.5, "No quality data available.", ha="center", va="center", color="white", fontsize=12)
        ax.axis("off")
        return fig

    labels = [
        ("q_target_zone", "Target zone"),
        ("q_shot_goal_threat", "Shot-goal threat"),
        ("q_threat_style", "Threat style"),
        ("q_short_vs_direct", "Short vs direct"),
        ("q_open_vs_closed", "Open vs closed"),
        ("q_header_threat", "Header threat"),
        ("q_transition_risk", "Transition risk"),
    ]

    y = np.arange(len(labels))[::-1]
    for i, (col, label) in enumerate(labels):
        yy = y[i]
        x = pd.to_numeric(quality_df[col], errors="coerce").dropna().clip(lower=VISUAL_Z_MIN, upper=VISUAL_Z_MAX)
        # league dots
        ax.scatter(x, np.full(len(x), yy), s=38, color="#3aa76d", alpha=0.32, edgecolors="none")
        # selected team square
        t = quality_df[quality_df["team_name"].astype(str) == str(selected_team)]
        if not t.empty:
            tx = float(np.clip(float(t.iloc[0][col]), VISUAL_Z_MIN, VISUAL_Z_MAX))
            ax.scatter([tx], [yy], s=70, marker="s", color="#f3f4f6", edgecolors="#111827", linewidth=0.8, zorder=4)
        ax.text(VISUAL_Z_MIN - VISUAL_X_PAD + 0.02, yy, label, color="#e5e7eb", fontsize=11, va="center", ha="left")

    ax.axvline(0, color="#94a3b8", linewidth=1.2, alpha=0.7)
    ax.set_xlim(VISUAL_Z_MIN - VISUAL_X_PAD, VISUAL_Z_MAX + VISUAL_X_PAD)
    ax.set_ylim(-1, len(labels))
    ax.set_yticks([])
    ax.set_xticks([-2, 0, 2])
    ax.set_xticklabels(["Worse", "Average", "Better"], color="#e5e7eb", fontsize=11)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.set_title("How does this team profile in corner qualities?", color="white", fontsize=16, pad=16, fontweight="bold")
    fig.tight_layout()
    return fig


def _plot_quality_reference_altair(quality_df: pd.DataFrame, selected_team: str, compare_team=None):
    if quality_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point()

    quality_specs = [
        ("q_target_zone", "Target zone"),
        ("q_shot_goal_threat", "Shot-goal threat"),
        ("q_threat_style", "Threat style"),
        ("q_short_vs_direct", "Short vs direct"),
        ("q_open_vs_closed", "Open vs closed"),
        ("q_header_threat", "Header threat"),
        ("q_transition_risk", "Transition risk"),
    ]
    quality_order = [q[1] for q in quality_specs]

    rows = []
    for col, label in quality_specs:
        tmp = quality_df[["team_name", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna(subset=[col])
        for _, r in tmp.iterrows():
            x_raw = float(r[col])
            x = max(VISUAL_Z_MIN, min(VISUAL_Z_MAX, x_raw))
            x_snap = round(x / 0.10) * 0.10
            rows.append(
                {
                    "team_name": str(r["team_name"]),
                    "quality": label,
                    "x_raw": x_raw,
                    "x_plot": x_snap,
                    "is_selected": str(r["team_name"]) == str(selected_team),
                    "is_compare": (compare_team is not None) and (str(r["team_name"]) == str(compare_team)),
                }
            )
    if not rows:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point()

    df = pd.DataFrame(rows)
    league = df[(~df["is_selected"]) & (~df["is_compare"])].copy()
    compare = df[df["is_compare"]].copy()
    selected = df[df["is_selected"]].copy()

    x_enc = alt.X(
        "x_plot:Q",
        scale=alt.Scale(domain=[VISUAL_Z_MIN - VISUAL_X_PAD, VISUAL_Z_MAX + VISUAL_X_PAD]),
        axis=alt.Axis(
            values=[-2, 0, 2],
            labelExpr="datum.value == -2 ? 'Worse' : datum.value == 0 ? 'Average' : datum.value == 2 ? 'Better' : ''",
            labelColor="#a9c4bd",
            title=None,
            tickSize=0,
            domain=False,
            grid=False,
            labelFontSize=14,
            labelPadding=12,
        ),
    )
    y_enc = alt.Y(
        "quality:N",
        sort=quality_order,
        axis=alt.Axis(title=None, labels=False, ticks=False, domain=False, grid=False),
    )

    rules_df = pd.DataFrame({"quality": quality_order})
    h_rules = alt.Chart(rules_df).mark_rule(color="#88a59d", opacity=0.32, strokeWidth=1).encode(y=alt.Y("quality:N", sort=quality_order))

    cat_labels_df = pd.DataFrame({"quality": quality_order, "x": [0.0] * len(quality_order)})
    cat_labels = (
        alt.Chart(cat_labels_df)
        .mark_text(color="#dce7e3", fontSize=15, fontWeight="bold", align="center", baseline="bottom", dy=-18)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[VISUAL_Z_MIN - VISUAL_X_PAD, VISUAL_Z_MAX + VISUAL_X_PAD]), axis=None),
            y=alt.Y("quality:N", sort=quality_order),
            text="quality:N",
        )
    )

    side_meanings = {
        "Target zone": ("More far-post profile", "More near-post profile"),
        "Shot-goal threat": ("Lower shot/xG/goal-assist threat", "Higher shot/xG/goal-assist threat"),
        "Threat style": ("Phase 2 dominant profile", "Phase 1 dominant profile"),
        "Short vs direct": ("More direct-corner profile", "More short-corner profile"),
        "Open vs closed": ("More closed-foot delivery", "More open-foot delivery"),
        "Header threat": ("Weaker aerial threat (Top-5 HOPS)", "Stronger aerial threat (Top-5 HOPS)"),
        "Transition risk": ("Higher post-corner transition risk", "Lower post-corner transition risk"),
    }
    meanings_df = pd.DataFrame(
        {
            "quality": quality_order,
            "x_left": [VISUAL_Z_MIN - VISUAL_X_PAD + 0.02] * len(quality_order),
            "x_right": [VISUAL_Z_MAX + VISUAL_X_PAD - 0.02] * len(quality_order),
            "left_txt": [side_meanings[q][0] for q in quality_order],
            "right_txt": [side_meanings[q][1] for q in quality_order],
        }
    )
    left_meanings = (
        alt.Chart(meanings_df)
        .mark_text(color="#a9c4bd", fontSize=10.5, align="left", baseline="top", dy=10)
        .encode(
            x=alt.X("x_left:Q", scale=alt.Scale(domain=[VISUAL_Z_MIN - VISUAL_X_PAD, VISUAL_Z_MAX + VISUAL_X_PAD]), axis=None),
            y=alt.Y("quality:N", sort=quality_order),
            text="left_txt:N",
        )
    )
    right_meanings = (
        alt.Chart(meanings_df)
        .mark_text(color="#a9c4bd", fontSize=10.5, align="right", baseline="top", dy=10)
        .encode(
            x=alt.X("x_right:Q", scale=alt.Scale(domain=[VISUAL_Z_MIN - VISUAL_X_PAD, VISUAL_Z_MAX + VISUAL_X_PAD]), axis=None),
            y=alt.Y("quality:N", sort=quality_order),
            text="right_txt:N",
        )
    )

    league_points = (
        alt.Chart(league)
        .mark_circle(size=88, color="#58e898", opacity=0.55)
        .encode(
            x=x_enc,
            y=y_enc,
            tooltip=[
                alt.Tooltip("team_name:N", title="Team"),
                alt.Tooltip("quality:N", title="Quality"),
                alt.Tooltip("x_raw:Q", title="z-score", format=".2f"),
            ],
        )
    )

    selected_points = (
        alt.Chart(selected)
        .mark_square(size=155, color="#f3f4f6", stroke="#0f172a", strokeWidth=1)
        .encode(
            x=x_enc,
            y=y_enc,
            tooltip=[
                alt.Tooltip("team_name:N", title="Team"),
                alt.Tooltip("quality:N", title="Quality"),
                alt.Tooltip("x_raw:Q", title="z-score", format=".2f"),
            ],
        )
    )

    compare_points = (
        alt.Chart(compare)
        .mark_square(size=155, color="#f59e0b", stroke="#7c2d12", strokeWidth=1)
        .encode(
            x=x_enc,
            y=y_enc,
            tooltip=[
                alt.Tooltip("team_name:N", title="Team"),
                alt.Tooltip("quality:N", title="Quality"),
                alt.Tooltip("x_raw:Q", title="z-score", format=".2f"),
            ],
        )
    )

    chart = (
        alt.layer(h_rules, league_points, compare_points, selected_points, cat_labels, left_meanings, right_meanings)
        .properties(
            width=1180,
            height=620,
            title="How does this team profile in corner qualities?",
        )
        .configure(background="#062f27")
        .configure_view(stroke=None)
        .configure_title(color="#dce7e3", fontSize=26, fontWeight="bold", anchor="middle", dy=-4)
    )
    return chart


def _infer_comparison_team_from_prompt(user_prompt: str, available_teams: list, selected_team: str):
    def _normalize_team_text(text: str) -> str:
        txt = str(text or "").lower()
        txt = "".join(ch for ch in unicodedata.normalize("NFKD", txt) if not unicodedata.combining(ch))
        txt = re.sub(r"[^a-z0-9\s]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def _team_aliases(team_name: str) -> set:
        stop_tokens = {
            "ca",
            "club",
            "atletico",
            "fc",
            "cf",
            "sc",
            "cd",
            "de",
            "del",
            "la",
            "el",
            "los",
            "las",
        }
        norm = _normalize_team_text(team_name)
        if not norm:
            return set()
        tokens = norm.split()
        core = [t for t in tokens if t not in stop_tokens]

        aliases = {norm}
        if core:
            aliases.add(" ".join(core))
        if len(core) == 1 and len(core[0]) >= 4:
            aliases.add(core[0])
        if len(core) >= 2:
            aliases.add(" ".join(core[:2]))
            aliases.add(" ".join(core[-2:]))
        return {a for a in aliases if len(a) >= 3}

    prompt = _normalize_team_text(user_prompt)
    if not prompt:
        return None

    comparison_tokens = ["compare", "comparar", "compara", "vs", "versus", "contra", "frente a"]
    if not any(tok in prompt for tok in comparison_tokens):
        return None

    selected_norm = _normalize_team_text(selected_team)
    alias_bag = {}
    for team in available_teams:
        team_txt = str(team)
        team_norm = _normalize_team_text(team_txt)
        if team_txt == str(selected_team) or team_norm == selected_norm:
            continue
        for alias in _team_aliases(team_txt):
            alias_bag.setdefault(alias, set()).add(team_txt)

    # Keep only aliases that map to exactly one team to avoid ambiguous short names.
    alias_to_team = {alias: list(teams)[0] for alias, teams in alias_bag.items() if len(teams) == 1}

    candidates = []
    for alias, team_txt in alias_to_team.items():
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        for m in re.finditer(pattern, prompt):
            candidates.append((m.start(), -len(alias), team_txt))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]

def _plot_quality_reference_interactive(quality_df: pd.DataFrame, selected_team: str):
    if quality_df.empty:
        return go.Figure()

    quality_specs = [
        ("q_target_zone", "Target zone"),
        ("q_shot_goal_threat", "Shot-goal threat"),
        ("q_threat_style", "Threat style"),
        ("q_short_vs_direct", "Short vs direct"),
        ("q_open_vs_closed", "Open vs closed"),
        ("q_header_threat", "Header threat"),
        ("q_transition_risk", "Transition risk"),
    ]

    bg = "#062f27"
    fg = "#dce7e3"
    quality_order = [q[1] for q in quality_specs]
    y_map = {label: (len(quality_order) - 1 - i) for i, label in enumerate(quality_order)}

    # Build long table for traces
    rows = []
    for col, label in quality_specs:
        tmp = quality_df[["team_name", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna(subset=[col])
        for _, r in tmp.iterrows():
            # Clip and snap to a small grid for a cleaner "dot-line" reference look.
            x_raw = float(r[col])
            x = max(VISUAL_Z_MIN, min(VISUAL_Z_MAX, x_raw))
            x_snap = round(x / 0.10) * 0.10
            rows.append(
                {
                    "team_name": str(r["team_name"]),
                    "quality": label,
                    "y": y_map[label],
                    "x_raw": x_raw,
                    "x_plot": x_snap,
                    "is_selected": str(r["team_name"]) == str(selected_team),
                }
            )
    if not rows:
        return go.Figure()
    long_df = pd.DataFrame(rows)

    league_df = long_df[~long_df["is_selected"]].copy()
    sel_df = long_df[long_df["is_selected"]].copy()
    fig = go.Figure()

    # Horizontal guides per quality
    for label in quality_order:
        yy = y_map[label]
        fig.add_shape(
            type="line",
            x0=VISUAL_Z_MIN + 0.03,
            y0=yy,
            x1=VISUAL_Z_MAX - 0.03,
            y1=yy,
            line=dict(color="rgba(159,189,182,0.45)", width=1),
            layer="below",
        )

    # Center line
    fig.add_shape(
        type="line",
        x0=0,
        y0=-0.7,
        x1=0,
        y1=len(quality_order) - 0.3,
        line=dict(color="rgba(159,189,182,0.75)", width=1.5),
        layer="below",
    )

    if not league_df.empty:
        fig.add_trace(
            go.Scatter(
                x=league_df["x_plot"],
                y=league_df["y"],
                mode="markers",
                name="Other teams",
                marker=dict(size=10, color="rgba(88,232,152,0.85)", line=dict(width=0)),
                customdata=league_df[["team_name", "quality", "x_raw"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}: %{customdata[2]:.2f}<extra></extra>",
            )
        )

    if not sel_df.empty:
        fig.add_trace(
            go.Scatter(
                x=np.clip(sel_df["x_raw"].astype(float), VISUAL_Z_MIN, VISUAL_Z_MAX),
                y=sel_df["y"],
                mode="markers",
                name=selected_team,
                marker=dict(
                    symbol="square",
                    size=12,
                    color="#f3f4f6",
                    line=dict(color="#0f172a", width=1.0),
                ),
                customdata=sel_df[["team_name", "quality", "x_raw"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}: %{customdata[2]:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="How does this team profile in corner qualities?",
            x=0.5,
            font=dict(size=19, color=fg),
        ),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=fg),
        height=640,
        margin=dict(l=130, r=30, t=72, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="right", x=1.0),
    )
    fig.update_xaxes(
        range=[VISUAL_Z_MIN - VISUAL_X_PAD, VISUAL_Z_MAX + VISUAL_X_PAD],
        showgrid=False,
        tickvals=[-2, 0, 2],
        ticktext=["Low", "Average", "High"],
        zeroline=False,
        color=fg,
    )
    fig.update_yaxes(
        tickvals=[y_map[q] for q in quality_order],
        ticktext=quality_order,
        showgrid=False,
        zeroline=False,
        color=fg,
    )
    return fig


def _plot_distribution(context_df: pd.DataFrame, metric: str, selected_team: str, title: str, color: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    if context_df.empty or metric not in context_df.columns:
        ax.text(0.5, 0.5, "No context data available.", ha="center", va="center")
        ax.axis("off")
        return fig

    x = pd.to_numeric(context_df[metric], errors="coerce").dropna()
    if x.empty:
        ax.text(0.5, 0.5, "No context data available.", ha="center", va="center")
        ax.axis("off")
        return fig

    ax.hist(x, bins=min(10, max(5, int(np.sqrt(len(x)) + 1))), color="#cbd5e1", edgecolor="#475569", alpha=0.85)
    team_row = context_df[context_df["team_name"].astype(str) == str(selected_team)]
    if not team_row.empty:
        value = float(team_row.iloc[0][metric])
        ax.axvline(value, color=color, linewidth=2.4)
        ax.text(value, ax.get_ylim()[1] * 0.96, selected_team, color=color, fontsize=9, ha="left", va="top")
    ax.set_title(title)
    ax.set_ylabel("Teams")
    ax.grid(axis="y", alpha=0.2)
    return fig


def _plot_heatmap(corners: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    pitch = VerticalPitch(
        half=True,
        pitch_type="statsbomb",
        pitch_color="#f8fafc",
        line_color="#111827",
        line_zorder=4,
    )
    pitch.draw(ax=ax)
    ax.set_title(title)
    if corners.empty:
        ax.text(40, 50, "No data in current filter.", ha="center", va="center", color="#6b7280")
        return fig

    b = pitch.bin_statistic(
        corners["pass_end_x"],
        corners["pass_end_y"],
        statistic="count",
        bins=(14, 10),
    )
    pitch.heatmap(b, ax=ax, cmap="Reds", edgecolor="#d1d5db", alpha=0.78)
    pitch.label_heatmap(
        b,
        ax=ax,
        color="#111827",
        fontsize=9,
        str_format="{:.0f}",
        exclude_zeros=True,
        ha="center",
        va="center",
    )
    return fig


def _plot_top_players(corners: pd.DataFrame, column: str, title: str, color: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    if corners.empty or column not in corners.columns:
        ax.text(0.5, 0.5, "No data in current filter.", ha="center", va="center")
        ax.axis("off")
        return fig
    top = corners[column].fillna("N/A").astype(str).value_counts().head(8).sort_values(ascending=True)
    if top.empty:
        ax.text(0.5, 0.5, "No data in current filter.", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.barh(top.index, top.values, color=color)
    ax.set_xlabel("Corners")
    return fig


def _plot_corner_paths(corners: pd.DataFrame, title: str, color: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    pitch = VerticalPitch(
        half=True,
        pitch_type="statsbomb",
        pitch_color="#f8fafc",
        line_color="#111827",
        line_zorder=4,
    )
    pitch.draw(ax=ax)
    ax.set_title(title)

    if corners.empty:
        ax.text(40, 50, "No data in current filter.", ha="center", va="center", color="#6b7280")
        return fig

    pitch.lines(
        corners["location_x"],
        corners["location_y"],
        corners["pass_end_x"],
        corners["pass_end_y"],
        lw=1.5,
        comet=True,
        transparent=True,
        color=color,
        alpha=0.65,
        ax=ax,
        zorder=2,
    )

    goal_col = corners["is_goal_assist"] if "is_goal_assist" in corners.columns else pd.Series(False, index=corners.index)
    shot_col = corners["pass_shot_assist"] if "pass_shot_assist" in corners.columns else pd.Series(False, index=corners.index)

    regular = corners[~goal_col & ~shot_col]
    shot_assist = corners[shot_col & ~goal_col]
    goal_assist = corners[goal_col]

    if not regular.empty:
        pitch.scatter(
            regular["pass_end_x"],
            regular["pass_end_y"],
            s=36,
            c="#9ca3af",
            edgecolors="none",
            ax=ax,
            zorder=3,
            alpha=0.85,
        )
    if not shot_assist.empty:
        pitch.scatter(
            shot_assist["pass_end_x"],
            shot_assist["pass_end_y"],
            s=58,
            c="#f59e0b",
            edgecolors="#111827",
            linewidth=0.4,
            ax=ax,
            zorder=4,
            alpha=0.95,
        )
    if not goal_assist.empty:
        pitch.scatter(
            goal_assist["pass_end_x"],
            goal_assist["pass_end_y"],
            s=72,
            c="#10b981",
            edgecolors="#111827",
            linewidth=0.5,
            ax=ax,
            zorder=5,
            alpha=0.98,
        )

    return fig


def _draw_rank_bar(
    ax,
    series: pd.Series,
    title: str,
    color: str,
    top_n: int = 6,
    ytick_side: str = "left",
):
    ax.set_title(title, fontsize=11, fontweight="bold")
    if series is None or series.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#6b7280")
        ax.axis("off")
        return
    data = series.head(top_n).sort_values(ascending=True)
    y = np.arange(len(data))
    ax.barh(y, data.values, color=color, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(data.index.astype(str), fontsize=9)
    if ytick_side == "right":
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelright=True, labelleft=False, pad=6)
    else:
        ax.yaxis.tick_left()
        ax.tick_params(axis="y", labelleft=True, labelright=False, pad=2)
    ax.set_xlabel("Corners", fontsize=9)
    ax.grid(axis="x", alpha=0.16)
    for i, v in enumerate(data.values):
        ax.text(float(v) + 0.1, i, str(int(v)), va="center", fontsize=9)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)


def _draw_side_pitch(ax, corners: pd.DataFrame, side_name: str, color: str):
    pitch = VerticalPitch(
        half=True,
        pitch_type="statsbomb",
        pitch_color="#f8fafc",
        line_color="#111827",
        line_zorder=5,
        pad_left=-14,
        pad_right=-14,
        pad_bottom=-34,
    )
    pitch.draw(ax=ax)
    ax.set_title(f"{side_name} side", fontsize=11, fontweight="bold", pad=14)

    if corners.empty:
        ax.text(40, 52, "No corners", ha="center", va="center", color="#6b7280")
        return

    pitch.lines(
        corners["location_x"],
        corners["location_y"],
        corners["pass_end_x"],
        corners["pass_end_y"],
        lw=1.3,
        comet=True,
        transparent=True,
        color=color,
        alpha=0.62,
        ax=ax,
        zorder=2,
    )

    bins = pitch.bin_statistic(corners["pass_end_x"], corners["pass_end_y"], statistic="count", bins=(15, 10))
    pitch.heatmap(bins, ax=ax, cmap="Reds", edgecolor="#d1d5db", alpha=0.45)
    pitch.label_heatmap(
        bins,
        ax=ax,
        color="#111827",
        fontsize=8,
        str_format="{:.0f}",
        exclude_zeros=True,
        ha="center",
        va="center",
        zorder=6,
    )

    total = len(corners)
    short_pct = corners["is_short"].mean() * 100.0 if "is_short" in corners.columns else 0.0
    inswing_pct = corners["is_inswing"].mean() * 100.0 if "is_inswing" in corners.columns else 0.0
    outswing_pct = corners["is_outswing"].mean() * 100.0 if "is_outswing" in corners.columns else 0.0
    ax.text(
        0.5,
        0.90,
        f"N={total} | Short {short_pct:.0f}% | In {inswing_pct:.0f}% | Out {outswing_pct:.0f}%",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#374151",
    )


def _build_corner_profile_figure(corners: pd.DataFrame, team_name: str):
    fig = plt.figure(figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("#f3f4f6")
    fig.suptitle(f"{team_name} - Offensive Corner Profile", fontsize=17, fontweight="bold", y=0.965)
    gs = fig.add_gridspec(11, 24, left=0.03, right=0.97, top=0.89, bottom=0.06, wspace=0.9, hspace=1.1)

    left = corners[corners["side"] == "Left"].copy() if not corners.empty else corners
    right = corners[corners["side"] == "Right"].copy() if not corners.empty else corners
    left_targets = left[~left["is_short"]]["pass_recipient_name"].dropna().astype(str).value_counts() if not left.empty else pd.Series(dtype=int)
    right_targets = right[~right["is_short"]]["pass_recipient_name"].dropna().astype(str).value_counts() if not right.empty else pd.Series(dtype=int)
    left_takers = left["player_name"].dropna().astype(str).value_counts() if not left.empty else pd.Series(dtype=int)
    right_takers = right["player_name"].dropna().astype(str).value_counts() if not right.empty else pd.Series(dtype=int)

    ax_l_takers = fig.add_subplot(gs[0:5, 0:6])
    ax_l_pitch = fig.add_subplot(gs[0:5, 7:17])
    ax_l_targets = fig.add_subplot(gs[0:5, 18:24])
    ax_r_takers = fig.add_subplot(gs[6:11, 0:6])
    ax_r_pitch = fig.add_subplot(gs[6:11, 7:17])
    ax_r_targets = fig.add_subplot(gs[6:11, 18:24])

    _draw_rank_bar(ax_l_takers, left_takers, "Top takers (Left)", "#cc001b")
    _draw_side_pitch(ax_l_pitch, left, "Left", "#cc001b")
    _draw_rank_bar(ax_l_targets, left_targets, "Top targets (Left)", "#003275", ytick_side="right")

    _draw_rank_bar(ax_r_takers, right_takers, "Top takers (Right)", "#cc001b")
    _draw_side_pitch(ax_r_pitch, right, "Right", "#003275")
    _draw_rank_bar(ax_r_targets, right_targets, "Top targets (Right)", "#003275", ytick_side="right")
    return fig


def _build_corner_dashboard_figure(
    corners: pd.DataFrame,
    team_name: str,
    short_corner_df: pd.DataFrame,
    short_corner_shot20: dict,
):
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("#f3f4f6")
    fig.suptitle(f"{team_name} - Corner Dashboard", fontsize=17, fontweight="bold", y=0.965)
    gs = fig.add_gridspec(10, 24, left=0.03, right=0.97, top=0.89, bottom=0.07, wspace=0.8, hspace=1.0)

    left = corners[corners["side"] == "Left"].copy() if not corners.empty else corners
    right = corners[corners["side"] == "Right"].copy() if not corners.empty else corners

    ax_left = fig.add_subplot(gs[0:7, 0:11])
    ax_right = fig.add_subplot(gs[0:7, 13:24])
    _draw_side_pitch(ax_left, left, "Left", "#cc001b")
    _draw_side_pitch(ax_right, right, "Right", "#003275")

    ax_metrics = fig.add_subplot(gs[7:8, 0:24])
    if corners.empty:
        ax_metrics.text(0.5, 0.5, "No corners in current filter.", ha="center", va="center", fontsize=12, color="#6b7280")
        ax_metrics.axis("off")
        return fig

    z = _label_target_zone(corners)
    text = (
        f"Total corners: {len(corners)}   |   Shot assists: {int(corners['pass_shot_assist'].sum())}   |   "
        f"Goal assists: {int(corners['is_goal_assist'].sum())}   |   Assist xG: {float(corners['assist_xg'].sum()):.2f}\n"
        f"Short: {z['is_short'].mean()*100:.1f}%   Near post: {(z['target_zone']=='near_post').mean()*100:.1f}%   "
        f"Central: {(z['target_zone']=='central').mean()*100:.1f}%   Far post: {(z['target_zone']=='far_post').mean()*100:.1f}%   "
        f"Inswing: {z['is_inswing'].mean()*100:.1f}%   Outswing: {z['is_outswing'].mean()*100:.1f}%"
    )
    ax_metrics.text(0.5, 0.58, text, ha="center", va="center", fontsize=10.2, color="#111827")
    ax_metrics.axis("off")

    ax_sc1 = fig.add_subplot(gs[8:10, 0:8])
    ax_sc2 = fig.add_subplot(gs[8:10, 8:16])
    ax_sc3 = fig.add_subplot(gs[8:10, 16:24])
    _draw_short_corner_panel(ax_sc1, ax_sc2, ax_sc3, short_corner_df, short_corner_shot20)
    return fig


def _build_defensive_corner_figure(def_corners: pd.DataFrame, team_name: str):
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("#f3f4f6")
    fig.suptitle(f"{team_name} - Defensive Corners", fontsize=17, fontweight="bold", y=0.965)
    gs = fig.add_gridspec(10, 24, left=0.03, right=0.97, top=0.89, bottom=0.07, wspace=0.8, hspace=1.0)

    ax_left = fig.add_subplot(gs[0:7, 0:11])
    ax_right = fig.add_subplot(gs[0:7, 13:24])
    left = def_corners[def_corners["side"] == "Left"].copy() if not def_corners.empty else def_corners
    right = def_corners[def_corners["side"] == "Right"].copy() if not def_corners.empty else def_corners
    _draw_side_pitch(ax_left, left, "Against (Left)", "#4b5563")
    _draw_side_pitch(ax_right, right, "Against (Right)", "#374151")

    ax_bottom_left = fig.add_subplot(gs[7:10, 0:12])
    ax_bottom_right = fig.add_subplot(gs[7:10, 12:24])
    if def_corners.empty:
        ax_bottom_left.axis("off")
        ax_bottom_right.axis("off")
        ax_bottom_left.text(0.5, 0.5, "No defensive corners in filter.", ha="center", va="center", color="#6b7280")
        return fig

    opp_takers = def_corners["player_name"].dropna().astype(str).value_counts()
    opp_teams = def_corners["team_name"].dropna().astype(str).value_counts()
    _draw_rank_bar(ax_bottom_left, opp_takers, "Top opponent takers", "#4b5563")
    _draw_rank_bar(ax_bottom_right, opp_teams, "Top opponent teams", "#6b7280")
    return fig


def _plot_delivery_heatmap_dashboard(ax, pitch, box_df: pd.DataFrame, side_df: pd.DataFrame, side_title: str):
    ax.set_facecolor("#f3f4f6")
    pitch.draw(ax=ax)
    ax.set_title(side_title, fontsize=12, fontweight="bold")
    if box_df.empty:
        ax.text(40, 60, "No data", ha="center", va="center", fontsize=11, color="#6b7280")
        return

    stat = pitch.bin_statistic(box_df["pass_end_x"], box_df["pass_end_y"], statistic="count", bins=(15, 11), normalize=True)
    pitch.heatmap(stat, edgecolors="white", cmap=plt.cm.Reds, ax=ax, alpha=0.78, zorder=3)
    pitch.label_heatmap(stat, color="black", fontsize=9.5, fontweight="bold", ax=ax, str_format="{:.0%}", exclude_zeros=True)
    pitch.scatter(box_df["pass_end_x"], box_df["pass_end_y"], ax=ax, color="#5f6368", s=11, alpha=0.55, zorder=4)

    outs = int(round(100.0 * side_df["is_outswing"].mean())) if not side_df.empty else 0
    ins = int(round(100.0 * side_df["is_inswing"].mean())) if not side_df.empty else 0
    takers = side_df.groupby("player_name", dropna=True).size().sort_values(ascending=False) if not side_df.empty else pd.Series(dtype=int)
    if takers.empty:
        txt = f"Takers:\nN/A\nOutswing: {outs}% | Inswing: {ins}%"
    elif len(takers) == 1:
        txt = f"Takers:\n{takers.index[0]} ({int(round(100.0 * takers.iloc[0] / max(1, len(side_df))))}%)\nOutswing: {outs}% | Inswing: {ins}%"
    else:
        t1 = f"{takers.index[0]} ({int(round(100.0 * takers.iloc[0] / max(1, len(side_df))))}%)"
        t2 = f"{takers.index[1]} ({int(round(100.0 * takers.iloc[1] / max(1, len(side_df))))}%)"
        txt = f"Takers:\n{t1} | {t2}\nOutswing: {outs}% | Inswing: {ins}%"
    ax.text(0.5, 1.06, txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color="#111827")


def _plot_first_contact_bar_dashboard(ax, merged: pd.DataFrame, with_legend: bool = False):
    ax.set_title("First contact (top 5)", fontsize=11.5)
    ax.set_facecolor("#f3f4f6")
    if merged.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11, color="#6b7280")
        return
    m = merged.sort_values("first_contact_count", ascending=False).head(5).sort_values("first_contact_count", ascending=True)
    y = np.arange(len(m))
    h = 0.38 if len(m) == 1 else 0.62
    ax.barh(y, m["first_contact_count"], color="#d7dde6", ec="#334155", label="first contact", height=h)
    ax.barh(y, m["first_contact_shot"], color="#cc001b", ec="#334155", label="shot", height=h)
    ax.barh(y, m["first_contact_goal"], color="#003275", ec="#334155", label="goal", height=h)
    ax.set_yticks(y)
    ax.set_yticklabels(m["player"].astype(str), fontsize=9.5)
    max_x = float(m["first_contact_count"].max()) if len(m) else 1.0
    ax.set_xlim(0, max(3.0, max_x * (2.8 if len(m) == 1 else 1.35)))
    ax.set_ylim(-0.55, len(m) - 0.45)
    ax.grid(axis="x", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if with_legend:
        ax.legend(loc="lower right", fontsize=8.5)


def _plot_short_corner_possession_dashboard(ax, pitch_short, loc_df: pd.DataFrame, n_seconds: int):
    ax.set_facecolor("#f3f4f6")
    pitch_short.draw(ax=ax)
    ax.set_title(f"Possession at {n_seconds}s after short corner", fontsize=11)
    if loc_df.empty:
        ax.text(40, 60, "No data", ha="center", va="center", fontsize=10.5, color="#6b7280")
        return
    b = pitch_short.bin_statistic(loc_df["x"], loc_df["y"], statistic="count", bins=(30, 21))
    b["statistic"] = gaussian_filter(b["statistic"], 1)
    pitch_short.heatmap(b, edgecolors="white", cmap=plt.cm.Reds, ax=ax, alpha=0.78, zorder=3)


def _plot_shots_after_short_dashboard(ax, shots_df: pd.DataFrame):
    ax.set_title("Shots within 20s\n(excluding direct first-contact shot)", fontsize=11)
    ax.set_facecolor("#f3f4f6")
    if shots_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10.5, color="#6b7280")
        return
    m = shots_df.head(5).sort_values("shot_count", ascending=True)
    y = np.arange(len(m))
    h = 0.38 if len(m) == 1 else 0.62
    ax.barh(y, m["shot_count"], color="#cc001b", ec="#334155", height=h)
    ax.set_yticks(y)
    ax.set_yticklabels(m["player"].astype(str), fontsize=9.5)
    max_x = float(m["shot_count"].max()) if len(m) else 1.0
    ax.set_xlim(0, max(3.0, max_x * (2.8 if len(m) == 1 else 1.35)))
    ax.set_ylim(-0.55, len(m) - 0.45)
    ax.grid(axis="x", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _to_list(value):
    if isinstance(value, list):
        return value
    parsed = _parse_json(value)
    if isinstance(parsed, list):
        return parsed
    return []


def _build_defensive_corner_dashboard(events: pd.DataFrame, team_name: str):
    d = events.copy()
    if d.empty or "id" not in d.columns:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("#f3f4f6")
        ax.axis("off")
        ax.text(0.5, 0.5, "No defensive-corner data available.", ha="center", va="center", fontsize=13, color="#6b7280")
        return fig

    d["related_events_list"] = d.get("related_events", pd.Series([[]] * len(d))).apply(_to_list)

    # Opponent corners against selected team
    df = d[
        (d["team_name"].astype(str) != str(team_name))
        & (d["event_type_name"].astype(str) == "Pass")
        & (d["pass_type_name"].astype(str) == "Corner")
    ].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("#f3f4f6")
        ax.axis("off")
        ax.text(0.5, 0.5, "No opponent corners in current selection.", ha="center", va="center", fontsize=13, color="#6b7280")
        return fig

    df = df.dropna(subset=["location_x", "location_y", "pass_end_x", "pass_end_y"]).copy()
    df["related_player"] = None
    df["rival_first_touch_x"] = np.nan
    df["rival_first_touch_y"] = np.nan
    df["rival_first_touch_team"] = None

    event_types_contact = {"Interception", "Recovery", "Ball Recovery", "Clearance"}
    direct_touch_types = {"Ball Receipt*", "Shot", "Pass", "Carry", "Miscontrol"}
    aerial_cols = ["aerial_won", "clearance_aerial_won", "pass_aerial_won", "miscontrol_aerial_won", "shot_aerial_won"]

    def _first_contact_event(pass_row: pd.Series):
        rel_ids = pass_row.get("related_events_list", [])
        if not isinstance(rel_ids, list) or len(rel_ids) == 0:
            return None
        rel = d[d["id"].isin(rel_ids)].copy()
        if rel.empty:
            return None
        if "position_name" in rel.columns:
            rel = rel[rel["position_name"].astype(str) != "Goalkeeper"].copy()
        if rel.empty:
            return None
        if "index" in rel.columns:
            rel = rel.sort_values("index")

        direct = rel[rel["event_type_name"].astype(str).isin(event_types_contact)].copy()
        if not direct.empty:
            return direct.iloc[0]

        aerial_mask = pd.Series(False, index=rel.index)
        for c in aerial_cols:
            if c in rel.columns:
                aerial_mask = aerial_mask | rel[c].apply(_truthy)
        aerial = rel[aerial_mask].copy()
        if not aerial.empty:
            return aerial.iloc[0]

        duel = rel[rel["event_type_name"].astype(str) == "Duel"].copy()
        if not duel.empty:
            dt = duel.get("duel_type_name", pd.Series("", index=duel.index)).astype(str).str.lower()
            dout = duel.get("duel_outcome_name", pd.Series("", index=duel.index)).astype(str).str.lower()
            duel = duel[dt.str.contains("aerial", na=False) & (dt.str.contains("won", na=False) | dout.str.contains("won|success", regex=True, na=False))]
            if not duel.empty:
                return duel.iloc[0]

        pinter = rel[
            (rel["event_type_name"].astype(str) == "Pass")
            & (rel["pass_type_name"].astype(str) == "Interception")
        ].copy()
        if not pinter.empty:
            return pinter.iloc[0]

        direct_touch = rel[rel["event_type_name"].astype(str).isin(direct_touch_types)].copy()
        if not direct_touch.empty:
            return direct_touch.iloc[0]
        return None

    for idx, ev in df.iterrows():
        first_evt = _first_contact_event(ev)
        if first_evt is None:
            continue
        first_team = str(first_evt.get("team_name", ""))
        first_player = str(first_evt.get("player_name", "")).strip()
        if first_team == str(team_name):
            if first_player:
                df.at[idx, "related_player"] = first_player
        else:
            x = pd.to_numeric(first_evt.get("location_x"), errors="coerce")
            y = pd.to_numeric(first_evt.get("location_y"), errors="coerce")
            rival_team = str(first_evt.get("team_name", "")).strip()
            df.at[idx, "rival_first_touch_team"] = rival_team if rival_team else str(ev.get("team_name", "")).strip()
            if pd.notna(x) and pd.notna(y):
                df.at[idx, "rival_first_touch_x"] = float(x)
                df.at[idx, "rival_first_touch_y"] = float(y)

    df = df[df["location_x"] >= 60].copy()
    left = df[(df["location_y"] <= 40) & (df["location_x"] >= 60)].copy()
    right = df[(df["location_y"] >= 40) & (df["location_x"] >= 60)].copy()

    fig, axs = plt.subplots(3, 2, figsize=(16, 14))
    fig.patch.set_facecolor("#f3f4f6")
    pitch = VerticalPitch(half=True, pitch_type="statsbomb", pitch_color="none", line_color="#111", line_zorder=5, pad_left=-15, pad_right=-15, pad_bottom=-37)

    def _plot_zonal(ax, data: pd.DataFrame, title: str):
        ax.set_facecolor("#f3f4f6")
        pitch.draw(ax=ax)
        d2 = data[(data["pass_end_y"] > 18) & (data["pass_end_y"] < 62)].copy()
        if d2.empty:
            ax.set_title(title, fontsize=13, weight="bold")
            ax.text(40, 55, "No data", ha="center", va="center", fontsize=11, color="#6b7280")
            return
        d2["success"] = np.where(d2["pass_outcome_name"].isna(), 0, 1)
        b_mean = pitch.bin_statistic(d2["pass_end_x"], d2["pass_end_y"], values=d2["success"], statistic="mean", bins=(20, 13))
        b_mean["statistic"] = np.where(b_mean["statistic"] == 0, 0.001, b_mean["statistic"])
        b_mean["statistic"] = np.nan_to_num(b_mean["statistic"])
        b_count = pitch.bin_statistic(d2["pass_end_x"], d2["pass_end_y"], statistic="count", bins=(20, 13))
        pitch.heatmap(b_mean, ax=ax, cmap=plt.cm.Greys, edgecolor="lightgrey", alpha=0.75)
        pitch.label_heatmap(b_mean, color="black", fontsize=10, ax=ax, str_format="{:.0%}", exclude_zeros=True, ha="center", va="bottom")
        pitch.label_heatmap(b_count, color="black", fontsize=8, ax=ax, str_format="({:0.0f})", exclude_zeros=True, ha="center", va="top")
        rival_touch = d2.dropna(subset=["rival_first_touch_x", "rival_first_touch_y"]).copy()
        if not rival_touch.empty:
            pitch.scatter(rival_touch["rival_first_touch_x"], rival_touch["rival_first_touch_y"], ax=ax, s=52, c="#003275", alpha=0.23, edgecolors="none", zorder=6)
        ax.set_title(title, fontsize=13, weight="bold")

    def _bar_players(ax, data: pd.DataFrame, title: str):
        ax.set_facecolor("#f3f4f6")
        g = data.groupby("related_player", dropna=True).size().reset_index(name="first_contact")
        g = g[g["related_player"].astype(str).str.strip() != ""].copy()
        g = g.sort_values("first_contact", ascending=True).tail(10)
        if g.empty:
            ax.axis("off")
            ax.set_title(title, fontsize=12)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10, color="#6b7280")
            return
        y = np.arange(len(g))
        ax.barh(y, g["first_contact"], color="#cc001b", lw=1.2, edgecolor="#1f2937", height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(g["related_player"].astype(str), fontsize=9)
        ax.tick_params(bottom=False, top=False, labelbottom=False)
        for i, v in enumerate(g["first_contact"].tolist()):
            ax.text(float(v) - 0.08, i, f"{int(v)}", color="white", fontsize=9, ha="right", va="center", fontweight="bold")
        for s in ["right", "left", "top", "bottom"]:
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", alpha=0.14, color="#9CA3AF")
        ax.set_title(title, fontsize=12)

    def _bar_rival_teams(ax, data: pd.DataFrame, title: str):
        ax.set_facecolor("#f3f4f6")
        g = data.dropna(subset=["rival_first_touch_team"]).copy()
        g["rival_first_touch_team"] = g["rival_first_touch_team"].astype(str).str.strip()
        g = g[g["rival_first_touch_team"] != ""]
        if g.empty:
            ax.axis("off")
            ax.set_title(title, fontsize=12)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10, color="#6b7280")
            return
        g = g.groupby("rival_first_touch_team", dropna=True).size().reset_index(name="rival_first_touch").sort_values("rival_first_touch", ascending=True).tail(10)
        y = np.arange(len(g))
        ax.barh(y, g["rival_first_touch"], color="#4b5563", edgecolor="#1f2937", height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(g["rival_first_touch_team"].astype(str), fontsize=9)
        ax.tick_params(bottom=False, top=False, labelbottom=False)
        for i, v in enumerate(g["rival_first_touch"].tolist()):
            ax.text(float(v) - 0.08, i, f"{int(v)}", color="white", fontsize=9, ha="right", va="center", fontweight="bold")
        for s in ["right", "left", "top", "bottom"]:
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", alpha=0.14, color="#9CA3AF")
        ax.set_title(title, fontsize=12)

    _plot_zonal(axs[0, 0], left, "Left-side opponent corners")
    _plot_zonal(axs[0, 1], right, "Right-side opponent corners")
    _bar_players(axs[1, 0], left, "Defensive first contact players (Left)")
    _bar_players(axs[1, 1], right, "Defensive first contact players (Right)")
    _bar_rival_teams(axs[2, 0], left, "Rival first-touch by team (Left)")
    _bar_rival_teams(axs[2, 1], right, "Rival first-touch by team (Right)")

    fig.suptitle(f"{team_name} - Defensive Corners Dashboard", x=0.03, y=0.975, ha="left", fontsize=22, fontweight="bold", color="#111827")
    fig.text(
        0.03,
        0.947,
        "Percentage = successful first-contact by zone. Number in brackets = volume to zone.\n"
        "Red bars = your defensive first contacts. Bottom row = rival first-touch volume by team.",
        fontsize=10,
        color="#4b5563",
        ha="left",
        va="top",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def _build_corner_dashboard_page(events: pd.DataFrame, team_name: str, n_seconds: int = 5):
    d = events.copy()
    corners = _prepare_offensive_corners(d, team_name)
    if corners.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor("#f3f4f6")
        ax.axis("off")
        ax.text(0.5, 0.5, "No corners for this selection.", ha="center", va="center", fontsize=14, color="#6b7280")
        return fig

    left = corners[corners["side"] == "Left"].copy()
    right = corners[corners["side"] == "Right"].copy()
    left_box = left[left["pass_end_y"] > 18].copy()
    left_short = left[left["pass_end_y"] <= 18].copy()
    right_box = right[right["pass_end_y"] < 62].copy()
    right_short = right[right["pass_end_y"] >= 62].copy()

    def _first_contact_frames(box_df: pd.DataFrame, side_df: pd.DataFrame) -> pd.DataFrame:
        c = box_df.groupby("pass_recipient_name", dropna=True).size().reset_index(name="first_contact_count").rename(columns={"pass_recipient_name": "player"})
        s = box_df[box_df["corner_shot_assist"] | box_df["corner_goal_assist"]].groupby("pass_recipient_name", dropna=True).size().reset_index(name="first_contact_shot").rename(columns={"pass_recipient_name": "player"})
        g = side_df[side_df["corner_goal_assist"]].groupby("pass_recipient_name", dropna=True).size().reset_index(name="first_contact_goal").rename(columns={"pass_recipient_name": "player"})
        out = c.merge(s, on="player", how="left").merge(g, on="player", how="left")
        out["first_contact_shot"] = out["first_contact_shot"].fillna(0).astype(int)
        out["first_contact_goal"] = out["first_contact_goal"].fillna(0).astype(int)
        return out.sort_values("first_contact_count", ascending=False)

    left_fc = _first_contact_frames(left_box, left)
    right_fc = _first_contact_frames(right_box, right)

    def _possession_after_short(short_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, ev in short_df.iterrows():
            mid = ev.get("match_id")
            poss = ev.get("possession")
            start = float(ev.get("time_seconds", 0.0))
            if pd.isna(mid) or pd.isna(poss):
                continue
            sub = d[
                (d["match_id"] == mid)
                & (d["possession"] == poss)
                & (d["time_seconds"] > start)
                & ((d["time_seconds"] - start) <= float(n_seconds))
                & d["location_x"].notna()
                & d["location_y"].notna()
            ].copy()
            if sub.empty:
                continue
            sub["time_diff"] = sub["time_seconds"] - start
            t = sub.sort_values("time_diff", ascending=False).iloc[0]
            x, y = float(t["location_x"]), float(t["location_y"])
            if str(t.get("team_name", "")) != str(team_name):
                x = 120.0 - x
                y = 80.0 - y
            rows.append({"x": x, "y": y})
        return pd.DataFrame(rows)

    left_pos = _possession_after_short(left_short)
    right_pos = _possession_after_short(right_short)

    def _shots_after_side(side_df: pd.DataFrame) -> pd.DataFrame:
        names = []
        for _, ev in side_df.iterrows():
            mid = ev.get("match_id")
            start = float(ev.get("time_seconds", 0.0))
            pid = ev.get("id")
            sub = d[
                (d["match_id"] == mid)
                & (d["event_type_name"].astype(str) == "Shot")
                & (d["team_name"].astype(str) == str(team_name))
                & (d["time_seconds"] > start)
                & ((d["time_seconds"] - start) <= 20.0)
            ].copy()
            if "shot_key_pass_id" in sub.columns and pd.notna(pid):
                sub = sub[sub["shot_key_pass_id"] != pid]
            if sub.empty:
                continue
            names.extend(sub["player_name"].dropna().astype(str).tolist())
        if not names:
            return pd.DataFrame(columns=["player", "shot_count"])
        out = pd.Series(names).value_counts().reset_index()
        out.columns = ["player", "shot_count"]
        return out

    left_shots20 = _shots_after_side(left)
    right_shots20 = _shots_after_side(right)

    pitch = VerticalPitch(
        half=True,
        pitch_type="statsbomb",
        pitch_color="none",
        line_color="black",
        line_zorder=5,
        pad_bottom=-30,
        pad_left=-15,
        pad_right=-15,
    )
    pitch_short = VerticalPitch(half=True, pitch_type="statsbomb", pitch_color="none", line_color="black", line_zorder=5, pad_bottom=-12, pad_left=0, pad_right=0)

    fig, axes = plt.subplots(4, 2, figsize=(16, 17))
    fig.patch.set_facecolor("#f3f4f6")

    _plot_delivery_heatmap_dashboard(axes[0, 0], pitch, left_box, left, "From left side")
    _plot_delivery_heatmap_dashboard(axes[0, 1], pitch, right_box, right, "From right side")
    _plot_first_contact_bar_dashboard(axes[1, 0], left_fc, with_legend=True)
    _plot_first_contact_bar_dashboard(axes[1, 1], right_fc, with_legend=False)
    _plot_short_corner_possession_dashboard(axes[2, 0], pitch_short, left_pos, n_seconds)
    _plot_short_corner_possession_dashboard(axes[2, 1], pitch_short, right_pos, n_seconds)
    _plot_shots_after_short_dashboard(axes[3, 0], left_shots20)
    _plot_shots_after_short_dashboard(axes[3, 1], right_shots20)

    fig.suptitle(f"{team_name} - Offensive Corner Dashboard", x=0.03, y=0.986, ha="left", fontsize=23, fontweight="bold", color="#111827")
    fig.text(
        0.03,
        0.962,
        "Left and right side view: delivery quality, first contact, short-corner possession and follow-up shots.",
        fontsize=10.5,
        color="#4b5563",
        ha="left",
    )
    fig.text(0.27, 0.938, "From left side", fontsize=14, fontweight="bold", color="#111827", ha="center")
    fig.text(0.74, 0.938, "From right side", fontsize=14, fontweight="bold", color="#111827", ha="center")
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    return fig


def _build_quality_explained_report(context_df: pd.DataFrame, selected_team: str) -> str:
    q_df = _build_quality_scores(context_df)
    if q_df.empty or "team_name" not in q_df.columns:
        return "No quality data available for textual report."

    row = q_df[q_df["team_name"].astype(str) == str(selected_team)]
    if row.empty:
        return f"No quality row available for {selected_team}."
    r = row.iloc[0]

    quality_specs = [
        ("q_target_zone", "Target zone", "More far-post profile", "More near-post profile"),
        ("q_shot_goal_threat", "Shot-goal threat", "Lower shot/xG/goal-assist threat", "Higher shot/xG/goal-assist threat"),
        ("q_threat_style", "Threat style", "Phase 2 dominant profile", "Phase 1 dominant profile"),
        ("q_short_vs_direct", "Short vs direct", "More direct-corner profile", "More short-corner profile"),
        ("q_open_vs_closed", "Open vs closed", "More closed-foot delivery", "More open-foot delivery"),
        ("q_header_threat", "Header threat", "Weaker aerial threat (Top-5 HOPS)", "Stronger aerial threat (Top-5 HOPS)"),
        ("q_transition_risk", "Transition risk", "Higher post-corner transition risk", "Lower post-corner transition risk"),
    ]

    def _band(v: float) -> str:
        if v >= 1.0:
            return "Strong"
        if v >= 0.35:
            return "Moderate"
        if v <= -1.0:
            return "Strong"
        if v <= -0.35:
            return "Moderate"
        return "Neutral"

    lines = ["Quality-by-quality report"]
    for col, label, neg_txt, pos_txt in quality_specs:
        v = float(pd.to_numeric(r.get(col), errors="coerce"))
        side_txt = pos_txt if v > 0 else neg_txt if v < 0 else "Balanced profile"
        lines.append(f"- {label}: {_band(v)} signal ({v:+.2f} z). {side_txt}.")
    return "\n".join(lines)


def _build_quality_positioning_text(context_df: pd.DataFrame, selected_team: str) -> str:
    q_df = _build_quality_scores(context_df)
    if q_df.empty or "team_name" not in q_df.columns:
        return "No quality positioning data available."

    quality_specs = [
        ("q_target_zone", "Target zone"),
        ("q_shot_goal_threat", "Shot-goal threat"),
        ("q_threat_style", "Threat style"),
        ("q_short_vs_direct", "Short vs direct"),
        ("q_open_vs_closed", "Open vs closed"),
        ("q_header_threat", "Header threat"),
        ("q_transition_risk", "Transition risk"),
    ]
    n = int(len(q_df))
    if n <= 1:
        return "Not enough teams to compute league positioning."

    lines = ["League positioning by quality"]
    for col, label in quality_specs:
        d = q_df[["team_name", col]].copy()
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=[col])
        if d.empty or str(selected_team) not in d["team_name"].astype(str).tolist():
            continue
        d = d.sort_values(col, ascending=False).reset_index(drop=True)
        d["rank"] = np.arange(1, len(d) + 1)
        row = d[d["team_name"].astype(str) == str(selected_team)].iloc[0]
        rank = int(row["rank"])
        val = float(row[col])
        lines.append(f"- {label}: rank {rank}/{len(d)} ({val:+.2f} z).")
    return "\n".join(lines)


def _build_corner_wordalisation_context(
    selected_team: str,
    context_df: pd.DataFrame,
    off_corners: pd.DataFrame,
    def_corners: pd.DataFrame,
    events: pd.DataFrame,
    lineups: pd.DataFrame,
) -> str:
    n_off = int(len(off_corners))
    n_def = int(len(def_corners))

    short_pct = float(off_corners["is_short"].mean() * 100.0) if "is_short" in off_corners.columns and n_off > 0 else 0.0
    inswing_pct = float(off_corners["is_inswing"].mean() * 100.0) if "is_inswing" in off_corners.columns and n_off > 0 else 0.0
    outswing_pct = float(off_corners["is_outswing"].mean() * 100.0) if "is_outswing" in off_corners.columns and n_off > 0 else 0.0
    mixed_pct = max(0.0, 100.0 - inswing_pct - outswing_pct - short_pct)

    zone_counts = (
        off_corners["target_zone"].astype(str).value_counts(normalize=True) * 100.0
        if "target_zone" in off_corners.columns and n_off > 0
        else pd.Series(dtype=float)
    )
    near_pct = float(zone_counts.get("near_post", 0.0))
    central_pct = float(zone_counts.get("central", 0.0))
    far_pct = float(zone_counts.get("far_post", 0.0))
    edge_box_pct = float(zone_counts.get("short", 0.0))

    first_contact_rate = float(off_corners["pass_recipient_name"].notna().mean() * 100.0) if "pass_recipient_name" in off_corners.columns and n_off > 0 else 0.0
    shot_rate = float(off_corners["corner_shot_assist"].mean()) if "corner_shot_assist" in off_corners.columns and n_off > 0 else 0.0
    xg_per_corner = float(off_corners["assist_xg"].mean()) if "assist_xg" in off_corners.columns and n_off > 0 else 0.0
    total_xg = float(off_corners["assist_xg"].sum()) if "assist_xg" in off_corners.columns and n_off > 0 else 0.0

    shots_corner_for = events[
        (events["team_name"].astype(str) == str(selected_team))
        & (events["event_type_name"].astype(str) == "Shot")
        & (events["play_pattern_name"].astype(str) == "From Corner")
    ].copy()
    shots_corner_against = events[
        (events["team_name"].astype(str) != str(selected_team))
        & (events["event_type_name"].astype(str) == "Shot")
        & (events["play_pattern_name"].astype(str) == "From Corner")
    ].copy()

    shots_corner_for["set_piece_phase"] = pd.to_numeric(shots_corner_for.get("set_piece_phase"), errors="coerce")
    shots_corner_against["shot_xg"] = pd.to_numeric(shots_corner_against.get("shot_statsbomb_xg"), errors="coerce").fillna(0.0)

    phase2_share = float((shots_corner_for["set_piece_phase"] == 2.0).mean() * 100.0) if len(shots_corner_for) > 0 else 0.0
    conceded_xg_total = float(shots_corner_against["shot_xg"].sum()) if not shots_corner_against.empty else 0.0
    conceded_xg_per_opp_corner = (conceded_xg_total / n_def) if n_def > 0 else 0.0
    conceded_body_part = (
        shots_corner_against["shot_body_part_name"].astype(str).value_counts(normalize=True).mul(100.0).head(3).to_dict()
        if not shots_corner_against.empty and "shot_body_part_name" in shots_corner_against.columns
        else {}
    )
    q_df = _build_quality_scores(context_df)
    q_cols = [
        "team_name",
        "q_target_zone",
        "q_shot_goal_threat",
        "q_threat_style",
        "q_short_vs_direct",
        "q_open_vs_closed",
        "q_header_threat",
        "q_transition_risk",
    ]
    q_cols = [c for c in q_cols if c in q_df.columns]
    league_quality_table = q_df[q_cols].copy() if q_cols else pd.DataFrame()
    target_zone_style_map = {}
    if {"team_name", "near_post_pct", "central_pct", "far_post_pct"}.issubset(set(context_df.columns)):
        for _, r in context_df[["team_name", "near_post_pct", "central_pct", "far_post_pct"]].iterrows():
            team = str(r["team_name"])
            target_zone_style_map[team] = _target_zone_style_label(r.get("near_post_pct"), r.get("central_pct"), r.get("far_post_pct"))
    if not league_quality_table.empty:
        for col in [c for c in q_cols if c != "team_name"]:
            league_quality_table[col] = pd.to_numeric(league_quality_table[col], errors="coerce").round(3)
        league_quality_table = league_quality_table.rename(
            columns={
                "team_name": "team",
                "q_target_zone": "target zone tendency",
                "q_shot_goal_threat": "shot and chance threat",
                "q_threat_style": "first-contact vs second-phase threat",
                "q_short_vs_direct": "short vs direct tendency",
                "q_open_vs_closed": "open vs closed delivery tendency",
                "q_header_threat": "aerial threat",
                "q_transition_risk": "transition protection after corners",
            }
        )
        if "team" in league_quality_table.columns:
            league_quality_table["target zone style"] = league_quality_table["team"].map(target_zone_style_map).fillna("unknown")

    metric_cols = [
        "team_name",
        "corners_per_match",
        "short_pct",
        "inswing_pct",
        "outswing_pct",
        "near_post_pct",
        "central_pct",
        "far_post_pct",
        "shot_assist_rate",
        "goal_assist_rate",
        "assist_xg_per_corner",
        "corner_phase1_xg_per_match",
        "corner_phase2_xg_per_match",
        "risk_obv_against_per_corner",
        "team_top5_hops",
        "team_hops_weighted",
    ]
    metric_cols = [c for c in metric_cols if c in context_df.columns]
    league_metric_table = context_df[metric_cols].copy() if metric_cols else pd.DataFrame()
    if not league_metric_table.empty:
        for col in [c for c in metric_cols if c != "team_name"]:
            league_metric_table[col] = pd.to_numeric(league_metric_table[col], errors="coerce").round(3)
        league_metric_table = league_metric_table.rename(
            columns={
                "team_name": "team",
                "corners_per_match": "corners per match",
                "short_pct": "short-corner share (%)",
                "inswing_pct": "inswing share (%)",
                "outswing_pct": "outswing share (%)",
                "near_post_pct": "near-post target share (%)",
                "central_pct": "central target share (%)",
                "far_post_pct": "far-post target share (%)",
                "shot_assist_rate": "shots created per corner",
                "goal_assist_rate": "goals assisted per corner",
                "assist_xg_per_corner": "xG created per corner",
                "corner_phase1_xg_per_match": "initial-delivery threat per match",
                "corner_phase2_xg_per_match": "second-ball threat per match",
                "risk_obv_against_per_corner": "transition danger conceded per corner",
                "team_top5_hops": "team aerial threat (top-5 HOPS)",
                "team_hops_weighted": "team aerial threat (weighted HOPS)",
            }
        )

    selected_quality_profile = {}
    selected_vs_team_quality_comparison = []
    selected_row = q_df[q_df["team_name"].astype(str) == str(selected_team)]
    if not selected_row.empty:
        sr = selected_row.iloc[0]
        selected_quality_profile = {
            "target zone style": target_zone_style_map.get(str(selected_team), "unknown"),
            "shot and chance threat": _quality_level_text("q_shot_goal_threat", sr.get("q_shot_goal_threat")),
            "threat style": _quality_level_text("q_threat_style", sr.get("q_threat_style")),
            "short vs direct": _quality_level_text("q_short_vs_direct", sr.get("q_short_vs_direct")),
            "open vs closed": _quality_level_text("q_open_vs_closed", sr.get("q_open_vs_closed")),
            "aerial threat": _quality_level_text("q_header_threat", sr.get("q_header_threat")),
            "transition protection": _quality_level_text("q_transition_risk", sr.get("q_transition_risk")),
        }
        compare_cols = [
            ("q_shot_goal_threat", "shot and chance threat"),
            ("q_threat_style", "threat style"),
            ("q_short_vs_direct", "short vs direct"),
            ("q_open_vs_closed", "open vs closed"),
            ("q_header_threat", "aerial threat"),
            ("q_transition_risk", "transition protection"),
        ]
        others = q_df[q_df["team_name"].astype(str) != str(selected_team)].copy()
        for _, orow in others.iterrows():
            opp_team = str(orow["team_name"])
            comp = {"team": opp_team}
            sel_tz = target_zone_style_map.get(str(selected_team), "unknown")
            opp_tz = target_zone_style_map.get(opp_team, "unknown")
            comp["target zone orientation"] = (
                "similar_orientation"
                if sel_tz == opp_tz and sel_tz != "unknown"
                else "different_orientation"
            )
            for col, label in compare_cols:
                comp[label] = _compare_quality(sr.get(col), orow.get(col), clear_diff=CLEAR_DIFF)
            selected_vs_team_quality_comparison.append(comp)

    available_teams = (
        sorted(context_df["team_name"].dropna().astype(str).unique().tolist())
        if "team_name" in context_df.columns
        else []
    )

    return (
        f"Team: {selected_team}\n"
        f"Available teams in context ({len(available_teams)}): {available_teams}\n"
        f"Corner sample sizes: offensive corners={n_off}, defensive corners faced={n_def}\n"
        f"Delivery profile (%): short={short_pct:.1f}, inswing={inswing_pct:.1f}, outswing={outswing_pct:.1f}, mixed={mixed_pct:.1f}\n"
        "Open/Closed mapping: open means outswing, closed means inswing.\n"
        f"Target zones (%): near post={near_pct:.1f}, central={central_pct:.1f}, far post={far_pct:.1f}, edge of box/short-target proxy={edge_box_pct:.1f}\n"
        f"First contact success signal: rate={first_contact_rate:.1f}%\n"
        f"Chance generation: shots per corner={shot_rate:.3f}, xG per corner={xg_per_corner:.3f}, total corner-assisted xG={total_xg:.3f}\n"
        f"Second-phase reliance: share of corner shots in second-phase situations={phase2_share:.1f}%\n"
        f"Defensive outcomes after own corners: total xG conceded from opponent corners={conceded_xg_total:.3f}, xG conceded per opponent corner={conceded_xg_per_opp_corner:.3f}, conceded shot body-part profile={conceded_body_part}\n"
        f"Selected team quality levels (high>={HIGH}, low<={LOW}): {selected_quality_profile}\n"
        f"Selected team vs each opponent (clear diff>={CLEAR_DIFF}): {selected_vs_team_quality_comparison}\n"
        f"League quality table (all teams): {league_quality_table.to_dict(orient='records') if not league_quality_table.empty else []}\n"
        f"League metric table (all teams): {league_metric_table.to_dict(orient='records') if not league_metric_table.empty else []}\n"
        "Definitions: phase 1 is first-contact play from the initial delivery; phase 2 is second-ball/recycled play after first contact.\n"
        "Data caveats: defensive structure (zonal/man/hybrid), blocker roles, and explicit second-ball recoveries are not directly tagged in this dataset."
    )


sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()
st.title("Corner Analysis")
st.caption(
    f"Data source: Competition {COMPETITION_ID}, Season {SEASON_ID} "
    f"({RAW_EVENTS_DIR.as_posix()})"
)

if not RAW_EVENTS_DIR.exists():
    st.error(f"Events folder not found: {RAW_EVENTS_DIR}")
    st.stop()

dfm = _load_matches()
if dfm.empty:
    st.error(f"Matches file missing or empty: {RAW_MATCHES_FILE}")
    st.stop()

event_ids = {
    int(p.stem.split("=")[1])
    for p in RAW_EVENTS_DIR.glob("match_id=*.parquet")
    if "=" in p.stem
}
dfm = dfm[dfm["match_id"].astype(int).isin(event_ids)].copy()
if dfm.empty:
    st.error("No matches with local event files were found.")
    st.stop()

teams = sorted(set(dfm["home_name"].dropna().astype(str)).union(set(dfm["away_name"].dropna().astype(str))))
if not teams:
    st.error("No teams available in matches data.")
    st.stop()

c1, c2 = st.columns([1, 2])
with c1:
    selected_team = st.selectbox("Team", teams)

team_matches = dfm[
    (dfm["home_name"].astype(str) == str(selected_team))
    | (dfm["away_name"].astype(str) == str(selected_team))
].copy()
if "match_date" in team_matches.columns:
    team_matches = team_matches.sort_values("match_date", ascending=False)

match_options = []
for _, row in team_matches.iterrows():
    mid = int(row["match_id"])
    date_txt = str(row.get("match_date", ""))[:10]
    match_options.append((mid, f"{date_txt} | {row['home_name']} vs {row['away_name']} | match_id={mid}"))

with c2:
    mode = st.radio("Matches", ["All", "Single", "Multiple"], horizontal=True)
    if mode == "All":
        selected_ids = [mid for mid, _ in match_options]
    elif mode == "Single":
        selected_one = st.selectbox("Select match", match_options, format_func=lambda x: x[1])
        selected_ids = [int(selected_one[0])]
    else:
        selected_many = st.multiselect(
            "Select matches",
            match_options,
            default=match_options,
            format_func=lambda x: x[1],
        )
        selected_ids = [int(x[0]) for x in selected_many]

if not selected_ids:
    st.warning("Please select at least one match.")
    st.stop()

events_raw = _load_events(tuple(selected_ids))
if events_raw.empty:
    st.warning("No event files found for this match selection.")
    st.stop()

events = _prepare_events(events_raw)
off_corners = _prepare_offensive_corners(events, selected_team)
def_corners = _prepare_defensive_corners(events, selected_team)
lineups = _load_lineups(tuple(selected_ids))
short_corner_df, short_corner_shot20 = _short_corner_possession_analysis(
    events,
    off_corners,
    selected_team,
    window_s=5,
    shot_window_s=20,
)

sel_matches = team_matches[team_matches["match_id"].astype(int).isin(selected_ids)].copy()
if not sel_matches.empty and "match_date" in sel_matches.columns:
    min_date = str(sel_matches["match_date"].min())[:10]
    max_date = str(sel_matches["match_date"].max())[:10]
    st.caption(f"{len(selected_ids)} match(es) selected | {min_date} to {max_date}")
else:
    st.caption(f"{len(selected_ids)} match(es) selected")

all_match_ids = tuple(sorted(dfm["match_id"].astype(int).unique().tolist()))
all_events_raw = _load_events(all_match_ids)
all_lineups_raw = _load_lineups(all_match_ids)
context_df = _build_context_table(all_events_raw, dfm)
team_hops_df = _build_team_hops_metrics(all_lineups_raw)
if not team_hops_df.empty and not context_df.empty:
    context_df = context_df.merge(team_hops_df, on="team_name", how="left")
else:
    context_df["team_top5_hops"] = np.nan
    context_df["team_hops_weighted"] = np.nan
all_events = _prepare_events(all_events_raw)
phase_all_df = _build_phase_table(all_events, corner_only=False)
phase_corner_df = _build_phase_table(all_events, corner_only=True)

tabs = st.tabs(
    [
        "Correlation Matrix",
        "Contextual Analysis",
        "Corner Profile",
        "Corner Dashboard",
        "Defensive Corners",
        "HOPS",
        "xG Scatter",
        "Raw Data",
        "Wordalisation",
    ]
)

with tabs[0]:
    st.subheader("Correlation Matrix (Offensive Corners)")
    st.caption(
        "Define corner style/performance qualities and link them to measurable data. "
        "Metrics are z-score normalized across teams before correlation."
    )

    default_metrics = [
        "assist_xg_per_corner",
        "corner_phase1_xg_per_match",
        "corner_phase2_xg_per_match",
        "short_pct",
        "inswing_pct",
        "outswing_pct",
        "near_post_pct",
        "central_pct",
        "far_post_pct",
        "shot_assist_rate",
        "goal_assist_rate",
        "risk_reward_index",
        "team_top5_hops",
    ]
    available_metrics = [m for m in default_metrics if m in context_df.columns]
    metric_labels = {
        "assist_xg_per_corner": "Assist xG / corner",
        "corner_phase1_xg_per_match": "Corner phase 1 xG / match",
        "corner_phase2_xg_per_match": "Corner phase 2 xG / match",
        "short_pct": "Short corners %",
        "inswing_pct": "Inswing %",
        "outswing_pct": "Outswing %",
        "near_post_pct": "Near-post %",
        "central_pct": "Central-zone %",
        "far_post_pct": "Far-post %",
        "shot_assist_rate": "Shot-assist rate %",
        "goal_assist_rate": "Goal-assist rate %",
        "risk_reward_index": "Risk vs reward index",
        "team_top5_hops": "Top-5 header HOPS",
        "team_hops_weighted": "Weighted team HOPS",
    }
    selected_metrics = st.multiselect(
        "Metrics for correlation",
        options=available_metrics,
        default=available_metrics,
        format_func=lambda x: metric_labels.get(x, x),
        key="corr_metrics_multiselect",
    )

    if len(selected_metrics) < 2:
        st.info("Select at least two metrics to compute a correlation matrix.")
    else:
        corr_input = context_df[["team_name"] + selected_metrics].copy()
        corr_input = corr_input.dropna(subset=selected_metrics, how="any").reset_index(drop=True)
        if corr_input.empty:
            st.info("No complete rows for the selected metric set.")
        else:
            z = _zscore_frame(corr_input.copy(), selected_metrics)
            corr = z[selected_metrics].corr()
            corr_renamed = corr.rename(index=metric_labels, columns=metric_labels)
            st.pyplot(_plot_correlation_heatmap(corr_renamed, "Correlation Matrix (z-score normalized features)"), clear_figure=True)

            st.caption(
                "Interpretation guide: high positive values suggest redundancy; "
                "low/negative values suggest complementary dimensions."
            )
            with st.expander("Team-level z-score table"):
                z_disp = z.copy()
                for c in selected_metrics:
                    z_disp[c] = pd.to_numeric(z_disp[c], errors="coerce").round(2)
                z_disp = z_disp.rename(columns=metric_labels)
                st.dataframe(z_disp.sort_values("team_name"), use_container_width=True, hide_index=True)

            st.markdown("**Quality Builder (Reference Viz)**")
            st.caption(
                "Quality profile (z-score by team), shown on a low-average-high axis."
            )
            q_df = _build_quality_scores(context_df)
            st.altair_chart(
                _plot_quality_reference_altair(q_df, selected_team),
                use_container_width=True,
            )

            st.caption("Side-meaning legend is integrated inside each quality row.")

            with st.expander("Quality z-score table"):
                if q_df.empty:
                    st.info("No complete data to build quality z-scores.")
                else:
                    rename_map = {
                        "q_target_zone": "Target zone",
                        "q_shot_goal_threat": "Shot-goal threat",
                        "q_threat_style": "Threat style",
                        "q_short_vs_direct": "Short vs direct",
                        "q_open_vs_closed": "Open vs closed",
                        "q_header_threat": "Header threat",
                        "q_transition_risk": "Transition risk",
                    }
                    q_show = q_df.rename(columns=rename_map).copy()
                    for c in rename_map.values():
                        q_show[c] = pd.to_numeric(q_show[c], errors="coerce").round(2)
                    st.dataframe(q_show.sort_values("team_name"), use_container_width=True, hide_index=True)

            st.markdown("**Weights & Ranking**")
            st.caption(
                "Set a weight per metric to build a composite profile score. "
                "Positive = reward, negative = penalize."
            )
            normalize_weights = st.checkbox(
                "Normalize weights (sum of absolute weights = 1)",
                value=True,
                key="corr_weights_normalize",
            )

            raw_weights = {}
            ncols = 3
            cols_ui = st.columns(ncols)
            for i, m in enumerate(selected_metrics):
                with cols_ui[i % ncols]:
                    raw_weights[m] = st.number_input(
                        metric_labels.get(m, m),
                        min_value=-3.0,
                        max_value=3.0,
                        value=1.0,
                        step=0.1,
                        key=f"corr_w_{m}",
                    )

            w = pd.Series(raw_weights, dtype=float)
            if normalize_weights:
                denom = float(np.abs(w).sum())
                if denom > 0:
                    w = w / denom

            rank_df = z[["team_name"] + selected_metrics].copy()
            rank_df["composite_score"] = rank_df[selected_metrics].mul(w, axis=1).sum(axis=1)
            rank_df = rank_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
            rank_df["rank"] = np.arange(1, len(rank_df) + 1)

            st.pyplot(
                _plot_weighted_ranking(
                    rank_df,
                    "Weighted Ranking from Selected Corner Metrics",
                ),
                clear_figure=True,
            )

            show = rank_df[["rank", "team_name", "composite_score"]].copy()
            show["composite_score"] = pd.to_numeric(show["composite_score"], errors="coerce").round(3)
            st.dataframe(show, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Contextual Analysis (Offensive Corners)")
    st.caption("Context engineering focus: style + performance profiles against league distribution.")
    if context_df.empty:
        st.info("No context data available to build league distributions.")
    else:
        team_row = context_df[context_df["team_name"].astype(str) == str(selected_team)]
        if team_row.empty:
            st.info("Selected team is not available in context table.")
        else:
            t = team_row.iloc[0]
            st.markdown("**Style profile**")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Corners per Match", f"{float(t['corners_per_match']):.2f}")
            s2.metric("Short Corners %", f"{float(t['short_pct']):.1f}%")
            s3.metric("Inswing %", f"{float(t['inswing_pct']):.1f}%")
            s4.metric("Outswing %", f"{float(t['outswing_pct']):.1f}%")

            st.markdown("**Performance profile**")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Shot Assist Rate", f"{float(t['shot_assist_rate']):.1f}%")
            p2.metric("Goal Assist Rate", f"{float(t['goal_assist_rate']):.1f}%")
            p3.metric("Phase 1 xG / Match", f"{float(t['corner_phase1_xg_per_match']):.3f}")
            p4.metric("Phase 2 xG / Match", f"{float(t['corner_phase2_xg_per_match']):.3f}")

            p5, p6 = st.columns(2)
            p5.metric("Assist xG / Corner", f"{float(t['assist_xg_per_corner']):.3f}")
            p6.metric("Risk vs Reward", f"{float(t['risk_reward_index']):.3f}")

            st.caption(
                f"Risk vs Reward = Assist xG per corner minus transition risk after corner "
                f"(sum of positive opponent OBV-against in first {TRANSITION_WINDOW_S}s, averaged per corner)."
            )

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(
                    _plot_distribution(
                        context_df,
                        "short_pct",
                        selected_team,
                        "Distribution: Short Corners %",
                        "#cc001b",
                    ),
                    clear_figure=True,
                )
            with c2:
                st.pyplot(
                    _plot_distribution(
                        context_df,
                        "inswing_pct",
                        selected_team,
                        "Distribution: Inswing %",
                        "#003275",
                    ),
                    clear_figure=True,
                )

            c2b1, c2b2 = st.columns(2)
            with c2b1:
                st.pyplot(
                    _plot_distribution(
                        context_df,
                        "outswing_pct",
                        selected_team,
                        "Distribution: Outswing %",
                        "#003275",
                    ),
                    clear_figure=True,
                )
            with c2b2:
                st.pyplot(
                    _plot_distribution(
                        context_df,
                        "near_post_pct",
                        selected_team,
                        "Distribution: Near-post %",
                        "#cc001b",
                    ),
                    clear_figure=True,
                )

            c2c1, c2c2 = st.columns(2)
            with c2c1:
                st.pyplot(
                    _plot_distribution(
                        context_df,
                        "central_pct",
                        selected_team,
                        "Distribution: Central-zone %",
                        "#003275",
                    ),
                    clear_figure=True,
                )
            with c2c2:
                st.pyplot(
                    _plot_distribution(
                        context_df,
                        "far_post_pct",
                        selected_team,
                        "Distribution: Far-post %",
                        "#cc001b",
                    ),
                    clear_figure=True,
                )

            c3, c4 = st.columns(2)
            with c3:
                st.pyplot(
                    _plot_distribution(
                        context_df,
                        "corner_phase1_xg_per_match",
                        selected_team,
                        "Distribution: Corner Phase 1 xG per Match",
                        "#cc001b",
                    ),
                    clear_figure=True,
                )
            with c4:
                st.pyplot(
                    _plot_distribution(
                        context_df,
                        "corner_phase2_xg_per_match",
                        selected_team,
                        "Distribution: Corner Phase 2 xG per Match",
                        "#003275",
                    ),
                    clear_figure=True,
                )

            c5, c6 = st.columns(2)
            with c5:
                st.pyplot(
                    _plot_distribution(
                        context_df,
                        "assist_xg_per_corner",
                        selected_team,
                        "Distribution: Assist xG per Corner",
                        "#cc001b",
                    ),
                    clear_figure=True,
                )
            with c6:
                st.pyplot(
                    _plot_distribution(
                        context_df,
                        "risk_reward_index",
                        selected_team,
                        "Distribution: Risk vs Reward Index",
                        "#003275",
                    ),
                    clear_figure=True,
                )

            with st.expander("Offensive context table"):
                cols = [
                    "team_name",
                    "corners_per_match",
                    "short_pct",
                    "inswing_pct",
                    "outswing_pct",
                    "near_post_pct",
                    "central_pct",
                    "far_post_pct",
                    "shot_assist_rate",
                    "goal_assist_rate",
                    "corner_phase1_xg_per_match",
                    "corner_phase2_xg_per_match",
                    "assist_xg_per_corner",
                    "risk_obv_against_per_corner",
                    "risk_reward_index",
                ]
                st.dataframe(
                    context_df[cols].sort_values("risk_reward_index", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

with tabs[2]:
    st.subheader("Offensive Corner Profile")
    st.caption("Pelota_Quieta-style layout: takers + pitch map + targets per side.")
    st.pyplot(_build_corner_profile_figure(off_corners, selected_team), clear_figure=True)

with tabs[3]:
    st.subheader("Corner Dashboard")
    n_sec = st.slider("Seconds after short corner (possession map)", min_value=3, max_value=10, value=5, step=1, key="corner_dash_nsec")
    st.caption("Pelota_Quieta-style dashboard: delivery, first contact, short-corner possession and shots after short corners.")
    st.pyplot(
        _build_corner_dashboard_page(
            events,
            selected_team,
            n_seconds=int(n_sec),
        ),
        clear_figure=True,
    )

with tabs[4]:
    st.subheader("Defensive Corners (Against Selected Team)")
    st.caption("Pelota_Quieta-style defensive dashboard: zonal success, first contact and rival first-touch profile.")
    st.pyplot(_build_defensive_corner_dashboard(events, selected_team), clear_figure=True)

with tabs[5]:
    st.subheader("HOPS")
    st.caption("Aerial mismatch exploration based on duel HOPS ratings from event data.")
    min_minutes = st.slider("Minimum minutes", min_value=0, max_value=500, value=150, step=10)
    rival_teams = [t for t in teams if str(t) != str(selected_team)]
    if not rival_teams:
        st.info("No rival teams available for HOPS comparison.")
        selected_rival = None
    else:
        selected_rival = st.selectbox("Rival team", rival_teams, index=0, key="hops_rival_team")

    hops_selected = _build_hops_player_table(events, lineups, selected_team)
    hops_selected = hops_selected[hops_selected["player_minutes"] >= float(min_minutes)].copy()
    hops_selected = hops_selected.sort_values(["hops_rating", "hops_samples"], ascending=[False, False])

    if hops_selected.empty:
        st.info("No HOPS player ratings for the selected team under this sample filter.")
    else:
        st.write(f"{selected_team} player HOPS ranking")
        st.dataframe(
            hops_selected.rename(
                columns={
                    "player_name": "Player",
                    "hops_rating": "HOPS",
                    "hops_samples": "Samples",
                    "opp_quality": "Opponent Quality",
                    "player_minutes": "Minutes",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    if selected_rival:
        hops_rival = _build_hops_player_table(events, lineups, selected_rival)
        hops_rival = hops_rival[hops_rival["player_minutes"] >= float(min_minutes)].copy()
        if not hops_rival.empty:
            st.write(f"{selected_rival} player HOPS ranking")
            st.dataframe(
                hops_rival.rename(
                    columns={
                        "player_name": "Player",
                        "hops_rating": "HOPS",
                        "hops_samples": "Samples",
                        "opp_quality": "Opponent Quality",
                        "player_minutes": "Minutes",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        pairs = _build_hops_pairs(hops_selected, hops_rival, max_players=None)
        if pairs.empty:
            st.info("Not enough players to build HOPS mismatch for this team/rival and minutes filter.")
        else:
            st.pyplot(_plot_hops_mismatch(pairs, selected_team, selected_rival), clear_figure=True)

with tabs[6]:
    st.subheader("xG Scatter")
    st.caption("Set-piece xG split by phase (all set pieces and corners-only), with team and rival highlights.")
    rival_scatter_teams = [t for t in teams if str(t) != str(selected_team)]
    if not rival_scatter_teams:
        st.info("No rival teams available for scatter comparison.")
        scatter_rival = None
    else:
        scatter_rival = st.selectbox("Rival highlight", rival_scatter_teams, index=0, key="scatter_rival_team")

    if scatter_rival:
        focus_team = st.selectbox(
            "Focus team",
            teams,
            index=teams.index(selected_team) if selected_team in teams else 0,
            key="scatter_focus_team",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(
                _plot_phase_scatter_highlight(
                    phase_all_df,
                    "All Set Pieces: Phase 1 xG vs Phase 2 xG",
                    selected_team=focus_team,
                    rival_team=scatter_rival,
                ),
                clear_figure=True,
            )
        with c2:
            st.pyplot(
                _plot_phase_scatter_highlight(
                    phase_corner_df,
                    "Corners Only: Phase 1 xG vs Phase 2 xG",
                    selected_team=focus_team,
                    rival_team=scatter_rival,
                ),
                clear_figure=True,
            )

with tabs[7]:
    st.subheader("Raw Corner Tables")
    st.write("Offensive corners")
    if off_corners.empty:
        st.info("No offensive corners in current filter.")
    else:
        off_cols = [
            "match_id",
            "minute",
            "team_name",
            "player_name",
            "pass_recipient_name",
            "side",
            "is_short",
            "is_inswing",
            "is_outswing",
            "pass_shot_assist",
            "is_goal_assist",
            "assist_xg",
            "location_x",
            "location_y",
            "pass_end_x",
            "pass_end_y",
        ]
        off_cols = [c for c in off_cols if c in off_corners.columns]
        st.dataframe(off_corners[off_cols].sort_values(["match_id", "minute"]), use_container_width=True)

    st.write("Defensive corners (opponents)")
    if def_corners.empty:
        st.info("No defensive corners in current filter.")
    else:
        def_cols = [
            "match_id",
            "minute",
            "team_name",
            "player_name",
            "side",
            "location_x",
            "location_y",
            "pass_end_x",
            "pass_end_y",
        ]
        def_cols = [c for c in def_cols if c in def_corners.columns]
        st.dataframe(def_corners[def_cols].sort_values(["match_id", "minute"]), use_container_width=True)

with tabs[8]:
    st.subheader("Corner Wordalisation")
    st.caption("LLM-based tactical wordalisation from corner event metrics with transparent transcript.")
    try:
        with open("model cards/model-card-corner-analysis.md", "r", encoding="utf8") as file:
            corner_model_card_text = file.read()
        st.expander("Model card for Corner Analysis", expanded=False).markdown(corner_model_card_text)
    except Exception:
        st.info("Model card file not found: model cards/model-card-corner-analysis.md")

    context_text = _build_corner_wordalisation_context(
        selected_team=selected_team,
        context_df=context_df,
        off_corners=off_corners,
        def_corners=def_corners,
        events=events,
        lineups=lineups,
    )
    chat_state_key = f"corner_wordalisation_messages::{selected_team}::{hash(tuple(sorted(selected_ids)))}"
    corner_description = CornerDescription(team_name=selected_team, context_text=context_text)
    corner_chat = CornerChat(state_key=chat_state_key, description=corner_description)

    with st.expander("Dataframe used", expanded=False):
        used_cols = [
            "team_name",
            "corners",
            "corners_per_match",
            "short_pct",
            "inswing_pct",
            "outswing_pct",
            "near_post_pct",
            "central_pct",
            "far_post_pct",
            "shot_assist_rate",
            "goal_assist_rate",
            "assist_xg_per_corner",
            "corner_phase1_xg_per_match",
            "corner_phase2_xg_per_match",
            "risk_obv_against_per_corner",
            "risk_reward_index",
            "team_top5_hops",
            "team_hops_weighted",
        ]
        used_cols = [c for c in used_cols if c in context_df.columns]
        st.dataframe(
            context_df[used_cols].sort_values("team_name"),
            use_container_width=True,
            hide_index=True,
        )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Generate example report", key="corner_wordalisation_example_btn"):
            example_query = (
                f"Generate a concise corner analysis report for {selected_team}. "
                "Use short paragraphs and focus on measurable strengths, inefficiencies, and tactical implications."
            )
            with st.spinner("Generating wordalisation..."):
                try:
                    corner_chat.ask(example_query)
                except Exception as e:
                    st.error(f"Wordalisation failed: {e}")
    with col_b:
        st.download_button(
            "Download context snapshot",
            data=context_text,
            file_name=f"corner_wordalisation_context_{selected_team}.txt",
            mime="text/plain",
            key="corner_wordalisation_download_context",
        )

    available_teams = (
        context_df["team_name"].dropna().astype(str).unique().tolist()
        if "team_name" in context_df.columns
        else []
    )
    q_word_df = _build_quality_scores(context_df)

    for i, msg in enumerate(corner_chat.history):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("role") == "assistant" and not q_word_df.empty:
                prev_user_prompt = ""
                if i > 0 and corner_chat.history[i - 1].get("role") == "user":
                    prev_user_prompt = str(corner_chat.history[i - 1].get("content", ""))
                compare_team = _infer_comparison_team_from_prompt(prev_user_prompt, available_teams, selected_team)
                st.markdown("**Team Quality Visual**")
                if compare_team:
                    st.caption(f"Comparison highlight: {selected_team} (white) vs {compare_team} (orange)")
                st.altair_chart(
                    _plot_quality_reference_altair(q_word_df, selected_team, compare_team=compare_team),
                    use_container_width=True,
                )

    with st.expander("Chat transcript", expanded=False):
        if corner_chat.last_transcript:
            st.write(corner_chat.last_transcript)
        else:
            st.info("No transcript yet. Generate an example report or send a chat message.")

    user_prompt = st.chat_input(
        f"Ask about {selected_team} corners (e.g. How does {selected_team} take corners?)",
        key="corner_wordalisation_chat_input",
    )
    if user_prompt:
        with st.spinner("Generating wordalisation..."):
            try:
                corner_chat.ask(user_prompt)
                st.rerun()
            except Exception as e:
                st.error(f"Wordalisation failed: {e}")

