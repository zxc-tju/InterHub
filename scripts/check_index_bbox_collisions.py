import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from index_cache_utils import load_index_scene_tracks


DEFAULT_INPUT_CSV = "data/3_paperplot_data/all_results.csv"
DEFAULT_CACHE_ROOT = "data/1_unified_cache"
DEFAULT_OUTPUT_DIR = "data/3_paperplot_data/bbox_collision_check"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check oriented bbox collisions for the full InterHub 2w index across cached datasets."
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="all_results")
    parser.add_argument("--vehicle-length", type=float, default=4.5)
    parser.add_argument("--vehicle-width", type=float, default=1.8)
    parser.add_argument("--heading-source", choices=["raw", "trajectory"], default="raw")
    parser.add_argument("--heading-window-frames", type=int, default=3)
    parser.add_argument("--heading-min-displacement", type=float, default=0.2)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--limit-rows", type=int, default=None)
    return parser.parse_args()


def oriented_box(x: float, y: float, heading: float, length: float, width: float) -> Polygon:
    half_length = length / 2.0
    half_width = width / 2.0
    corners = np.array(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    c = np.cos(heading)
    s = np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return Polygon(corners @ rot.T + np.array([x, y], dtype=float))


def finite_values(*values) -> bool:
    return all(np.isfinite(float(value)) for value in values)


def vehicle_extent(row, fallback_length: float, fallback_width: float) -> Tuple[float, float, str]:
    length = fallback_length
    width = fallback_width
    source = "fixed_default_vehicle_size"
    if "length" in row.index and pd.notna(row["length"]) and float(row["length"]) > 0:
        length = float(row["length"])
        source = "cache_length_width"
    if "width" in row.index and pd.notna(row["width"]) and float(row["width"]) > 0:
        width = float(row["width"])
        source = "cache_length_width" if source == "cache_length_width" else "mixed_cache_default"
    return float(length), float(width), source


def agent_frame_dict(
    scene_df: pd.DataFrame,
    agent_id: str,
    fallback_length: float,
    fallback_width: float,
    heading_col: str = "bbox_heading",
) -> Dict[int, Tuple[float, float, float, float, float, str]]:
    agent_df = scene_df[scene_df["agent_id_str"].eq(str(agent_id))]
    result = {}
    has_length = "length" in agent_df.columns
    has_width = "width" in agent_df.columns
    for row in agent_df.itertuples(index=False):
        length = fallback_length
        width = fallback_width
        extent_source = "fixed_default_vehicle_size"
        if has_length:
            row_length = getattr(row, "length")
            if pd.notna(row_length) and float(row_length) > 0:
                length = float(row_length)
                extent_source = "cache_length_width"
        if has_width:
            row_width = getattr(row, "width")
            if pd.notna(row_width) and float(row_width) > 0:
                width = float(row_width)
                extent_source = "cache_length_width" if extent_source == "cache_length_width" else "mixed_cache_default"
        result[int(row.scene_ts)] = (
            float(row.x),
            float(row.y),
            float(getattr(row, heading_col)),
            length,
            width,
            extent_source,
        )
    return result


def get_collision_check_window(row: pd.Series) -> Tuple[int, int, str]:
    start = int(row["start"])
    if "agent1_closest_frame" in row.index and "agent2_closest_frame" in row.index:
        end_candidates = [start]
        for col in ("agent1_closest_frame", "agent2_closest_frame"):
            value = row.get(col)
            if pd.notna(value) and np.isfinite(float(value)):
                end_candidates.append(int(value))
        return start, max(end_candidates), "start_to_key_agent_closest_frames"
    return min(start, int(row["end"])), max(start, int(row["end"])), "interaction_start_end"


def empty_metrics(row: pd.Series, error: str) -> Dict[str, object]:
    start = int(row["start"]) if pd.notna(row.get("start")) else pd.NA
    end = int(row["end"]) if pd.notna(row.get("end")) else start
    return {
        "bbox_collision": False,
        "bbox_check_start_frame": start,
        "bbox_check_end_frame": end,
        "bbox_check_window_source": "error",
        "bbox_collision_frames": "",
        "bbox_collision_frame_count": 0,
        "bbox_first_collision_frame": pd.NA,
        "bbox_max_overlap_area": 0.0,
        "bbox_min_center_distance": pd.NA,
        "bbox_checked_frame_count": 0,
        "bbox_missing_frame_count": 0,
        "bbox_extent_source": "",
        "bbox_collision_check_error": error,
    }


def key_agents_for_row(row: pd.Series) -> List[str]:
    agent_source = row.get("key_agents")
    if pd.isna(agent_source) or str(agent_source).strip().lower() in {"", "nan", "unknown"}:
        agent_source = row.get("track_id")
    return str(agent_source).split(";")[:2]


def check_row_collision(
    row: pd.Series,
    scene_df: pd.DataFrame,
    fallback_length: float,
    fallback_width: float,
) -> Dict[str, object]:
    key_agents = key_agents_for_row(row)
    if len(key_agents) != 2 or any(not agent or agent == "nan" for agent in key_agents):
        return empty_metrics(row, "invalid_key_agents")

    states1 = agent_frame_dict(scene_df, key_agents[0], fallback_length, fallback_width)
    states2 = agent_frame_dict(scene_df, key_agents[1], fallback_length, fallback_width)
    if not states1 or not states2:
        missing = [agent for agent, states in zip(key_agents, (states1, states2)) if not states]
        return empty_metrics(row, f"missing_key_agent_track:{';'.join(missing)}")

    start, end, window_source = get_collision_check_window(row)
    collision_frames: List[int] = []
    max_overlap_area = 0.0
    min_center_distance = None
    checked_frames = 0
    missing_frames = 0
    extent_sources = set()

    for frame in range(start, end + 1):
        state1 = states1.get(frame)
        state2 = states2.get(frame)
        if state1 is None or state2 is None:
            missing_frames += 1
            continue
        x1, y1, heading1, length1, width1, extent_source1 = state1
        x2, y2, heading2, length2, width2, extent_source2 = state2
        if not finite_values(x1, y1, heading1, length1, width1, x2, y2, heading2, length2, width2):
            missing_frames += 1
            continue

        extent_sources.update([extent_source1, extent_source2])
        checked_frames += 1
        center_distance = float(np.hypot(x1 - x2, y1 - y2))
        if min_center_distance is None or center_distance < min_center_distance:
            min_center_distance = center_distance

        box1 = oriented_box(x1, y1, heading1, length1, width1)
        box2 = oriented_box(x2, y2, heading2, length2, width2)
        overlap_area = float(box1.intersection(box2).area)
        if overlap_area > 0.0:
            collision_frames.append(frame)
            if overlap_area > max_overlap_area:
                max_overlap_area = overlap_area

    error = ""
    if checked_frames == 0:
        error = "no_common_valid_frames"
    elif missing_frames > 0:
        error = "missing_key_agent_frames"

    extent_source = "unknown"
    if extent_sources:
        extent_source = next(iter(extent_sources)) if len(extent_sources) == 1 else "mixed_cache_default"

    return {
        "bbox_collision": bool(collision_frames),
        "bbox_check_start_frame": int(start),
        "bbox_check_end_frame": int(end),
        "bbox_check_window_source": window_source,
        "bbox_collision_frames": ";".join(str(frame) for frame in collision_frames),
        "bbox_collision_frame_count": int(len(collision_frames)),
        "bbox_first_collision_frame": int(collision_frames[0]) if collision_frames else pd.NA,
        "bbox_max_overlap_area": float(max_overlap_area),
        "bbox_min_center_distance": float(min_center_distance) if min_center_distance is not None else pd.NA,
        "bbox_checked_frame_count": int(checked_frames),
        "bbox_missing_frame_count": int(missing_frames),
        "bbox_extent_source": extent_source,
        "bbox_collision_check_error": error,
    }


def bool_counts(series: pd.Series) -> Dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def repair_duplicate_errors(checked: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    checked = checked.copy()
    repair_cols = [
        "bbox_duplicate_repair_source_row_id",
        "bbox_duplicate_repair_source_dataset",
        "bbox_duplicate_repair_source_folder",
        "bbox_duplicate_repair_source_scenario_idx",
    ]
    for col in repair_cols:
        if col not in checked.columns:
            checked[col] = pd.NA

    key_cols = ["track_id", "start", "end", "intensity", "PET"]
    if any(col not in checked.columns for col in key_cols):
        return checked, []

    error_mask = checked["bbox_collision_check_error"].fillna("").astype(str).str.strip().ne("")
    ok = checked[~error_mask].copy()
    repairs = []
    bbox_cols = [col for col in checked.columns if col.startswith("bbox_")]
    skip_cols = set(repair_cols)

    for idx, row in checked[error_mask].iterrows():
        mask = pd.Series(True, index=ok.index)
        for col in key_cols:
            mask &= ok[col].astype(str).eq(str(row[col]))
        donors = ok[mask]
        if len(donors) != 1:
            continue

        donor = donors.iloc[0]
        for col in bbox_cols:
            if col not in skip_cols:
                checked.at[idx, col] = donor[col]
        checked.at[idx, "bbox_collision_check_error"] = ""
        checked.at[idx, "bbox_duplicate_repair_source_row_id"] = int(donor["source_row_id"])
        checked.at[idx, "bbox_duplicate_repair_source_dataset"] = donor["dataset"]
        checked.at[idx, "bbox_duplicate_repair_source_folder"] = donor["folder"]
        checked.at[idx, "bbox_duplicate_repair_source_scenario_idx"] = int(donor["scenario_idx"])
        repairs.append(
            {
                "source_row_id": int(row["source_row_id"]),
                "donor_source_row_id": int(donor["source_row_id"]),
                "track_id": str(row["track_id"]),
                "bbox_collision": bool(donor["bbox_collision"]),
                "donor_dataset": str(donor["dataset"]),
                "donor_folder": str(donor["folder"]),
                "donor_scenario_idx": int(donor["scenario_idx"]),
            }
        )

    return checked, repairs


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    cache_root = Path(args.cache_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interactions = pd.read_csv(input_csv)
    if args.limit_rows:
        interactions = interactions.head(args.limit_rows).copy()
    interactions = interactions.copy()
    interactions["source_row_id"] = np.arange(len(interactions))

    output_rows = []
    group_cols = ["dataset", "folder", "scenario_idx"]
    grouped = interactions.groupby(group_cols, sort=False)
    total_scenes = grouped.ngroups
    processed_rows = 0

    for scene_index, (_, scene_rows) in enumerate(grouped, start=1):
        first_row = scene_rows.iloc[0]
        scene_name = ""
        agent_file = ""
        needed_agents = set()
        for _, row in scene_rows.iterrows():
            needed_agents.update(key_agents_for_row(row))
        needed_agents = {agent for agent in needed_agents if agent and str(agent).lower() != "nan"}
        try:
            scene_df, scene_name, agent_path = load_index_scene_tracks(
                cache_root,
                first_row,
                args.heading_source,
                args.heading_window_frames,
                args.heading_min_displacement,
                heading_col="bbox_heading",
                agent_ids=needed_agents,
            )
            agent_file = str(agent_path)
            scene_error = ""
        except Exception as exc:
            scene_df = None
            scene_error = f"scene_load_error:{type(exc).__name__}:{exc}"

        for _, row in scene_rows.iterrows():
            result = row.to_dict()
            if scene_error:
                metrics = empty_metrics(row, scene_error)
            else:
                metrics = check_row_collision(row, scene_df, args.vehicle_length, args.vehicle_width)
            result.update(metrics)
            result.update(
                {
                    "bbox_scene_name": scene_name,
                    "bbox_agent_data_file": agent_file,
                    "bbox_vehicle_length_default_m": float(args.vehicle_length),
                    "bbox_vehicle_width_default_m": float(args.vehicle_width),
                    "bbox_heading_source": args.heading_source,
                    "bbox_heading_window_frames": int(args.heading_window_frames),
                    "bbox_heading_min_displacement_m": float(args.heading_min_displacement),
                }
            )
            output_rows.append(result)

        processed_rows += len(scene_rows)
        if args.progress_every > 0 and (scene_index % args.progress_every == 0 or scene_index == total_scenes):
            print(
                f"processed scenes {scene_index}/{total_scenes}; rows={processed_rows}/{len(interactions)}",
                flush=True,
            )

    checked = pd.DataFrame(output_rows).sort_values("source_row_id")
    checked, duplicate_repairs = repair_duplicate_errors(checked)
    collision_mask = checked["bbox_collision"].astype(bool)
    error_mask = checked["bbox_collision_check_error"].fillna("").astype(str).str.strip().ne("")
    no_collision_mask = (~collision_mask) & (~error_mask)

    checked_path = output_dir / f"{args.output_prefix}_bbox_collision_checked.csv"
    no_collision_path = output_dir / f"{args.output_prefix}_normal_no_bbox_collision.csv"
    collision_path = output_dir / f"{args.output_prefix}_bbox_collision_abnormal.csv"
    error_path = output_dir / f"{args.output_prefix}_bbox_collision_check_errors.csv"
    summary_path = output_dir / f"{args.output_prefix}_bbox_collision_summary.json"

    checked.to_csv(checked_path, index=False)
    checked.loc[no_collision_mask].to_csv(no_collision_path, index=False)
    checked.loc[collision_mask].to_csv(collision_path, index=False)
    checked.loc[error_mask].to_csv(error_path, index=False)

    summary = {
        "input_csv": str(input_csv),
        "cache_root": str(cache_root),
        "checked_csv": str(checked_path),
        "no_collision_csv": str(no_collision_path),
        "collision_abnormal_csv": str(collision_path),
        "error_csv": str(error_path),
        "rows": int(len(checked)),
        "scene_groups": int(total_scenes),
        "no_collision_rows": int(no_collision_mask.sum()),
        "bbox_collision_rows": int(collision_mask.sum()),
        "check_error_rows": int(error_mask.sum()),
        "bbox_collision_counts": bool_counts(checked["bbox_collision"]),
        "dataset_counts": {str(k): int(v) for k, v in checked["dataset"].value_counts(dropna=False).items()},
        "collision_by_dataset": {
            str(k): int(v) for k, v in checked.loc[collision_mask, "dataset"].value_counts(dropna=False).items()
        },
        "no_collision_by_dataset": {
            str(k): int(v) for k, v in checked.loc[no_collision_mask, "dataset"].value_counts(dropna=False).items()
        },
        "error_by_dataset": {
            str(k): int(v) for k, v in checked.loc[error_mask, "dataset"].value_counts(dropna=False).items()
        },
        "vehicle_length_default_m": float(args.vehicle_length),
        "vehicle_width_default_m": float(args.vehicle_width),
        "heading_source": args.heading_source,
        "heading_window_frames": int(args.heading_window_frames),
        "heading_min_displacement_m": float(args.heading_min_displacement),
        "extent_source_counts": bool_counts(checked["bbox_extent_source"]),
        "checked_frame_count_total": int(checked["bbox_checked_frame_count"].sum()),
        "missing_frame_count_total": int(checked["bbox_missing_frame_count"].sum()),
        "max_collision_frame_count": int(checked["bbox_collision_frame_count"].max()),
        "max_overlap_area": float(checked["bbox_max_overlap_area"].max()),
        "min_center_distance": float(checked["bbox_min_center_distance"].min()),
        "duplicate_repair_rows": duplicate_repairs,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
