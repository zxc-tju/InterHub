from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import dill
import pandas as pd

from vehicle_bbox_utils import add_motion_heading


def row_cache_root(cache_root: Path, folder: str, dataset: str) -> Path:
    candidates = [
        cache_root / str(folder) / str(dataset),
        cache_root / str(folder),
        cache_root / str(dataset),
    ]
    for candidate in candidates:
        if (candidate / "scenes_list.dill").exists():
            return candidate
    return candidates[0]


@lru_cache(maxsize=64)
def scene_name_by_data_idx(cache_root: str, folder: str, dataset: str) -> Dict[int, str]:
    root = row_cache_root(Path(cache_root), folder, dataset)
    scenes_path = root / "scenes_list.dill"
    if not scenes_path.exists():
        raise FileNotFoundError(scenes_path)

    scenes = dill.load(open(scenes_path, "rb"))
    mapping = {}
    for scene in scenes:
        data_idx = getattr(scene, "data_idx", None)
        if data_idx is None:
            data_idx = getattr(scene, "raw_data_idx", None)
        if data_idx is None:
            continue
        mapping[int(data_idx)] = str(getattr(scene, "name"))
    return mapping


def resolve_index_scene_dir(cache_root: Path, row: pd.Series) -> Tuple[Path, str]:
    folder = str(row["folder"])
    dataset = str(row["dataset"])
    scenario_idx = int(row["scenario_idx"])
    mapping = scene_name_by_data_idx(str(cache_root), folder, dataset)
    scene_name = mapping.get(scenario_idx)
    if scene_name is None:
        scene_name = fallback_scene_name_from_agents(cache_root, row)
    if scene_name is None:
        raise FileNotFoundError(
            f"scenario_idx {scenario_idx} not found in {cache_root / folder / dataset / 'scenes_list.dill'}"
        )

    scene_dir = row_cache_root(cache_root, folder, dataset) / scene_name
    if not scene_dir.exists():
        raise FileNotFoundError(scene_dir)
    return scene_dir, scene_name


def fallback_scene_name_from_agents(cache_root: Path, row: pd.Series):
    agent_source = row.get("key_agents")
    if pd.isna(agent_source) or str(agent_source).strip().lower() in {"", "nan", "unknown"}:
        agent_source = row.get("track_id")
    agents = [agent for agent in str(agent_source).split(";")[:2] if agent and agent.lower() != "nan"]
    if not agents or "_" not in agents[0]:
        return None

    prefix = agents[0].split("_", 1)[0]
    root = row_cache_root(cache_root, str(row["folder"]), str(row["dataset"]))
    candidates = sorted(path for path in root.glob(f"*_{prefix}") if path.is_dir())
    matches = []
    for candidate in candidates:
        agent_file = resolve_agent_data_file(candidate)
        agent_ids = set(pd.read_feather(agent_file, columns=["agent_id"])["agent_id"].astype(str).unique())
        if set(agents).issubset(agent_ids):
            matches.append(candidate.name)
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_agent_data_file(scene_dir: Path) -> Path:
    files = sorted(scene_dir.glob("agent_data_dt*.feather"))
    if not files:
        raise FileNotFoundError(f"no agent_data_dt*.feather under {scene_dir}")
    return files[0]


def load_index_scene_tracks(
    cache_root: Path,
    row: pd.Series,
    heading_source: str,
    heading_window_frames: int,
    heading_min_displacement: float,
    heading_col: str = "bbox_heading",
    agent_ids=None,
) -> Tuple[pd.DataFrame, str, Path]:
    scene_dir, scene_name = resolve_index_scene_dir(cache_root, row)
    agent_file = resolve_agent_data_file(scene_dir)
    scene_df = pd.read_feather(agent_file)
    required = {"agent_id", "scene_ts", "x", "y", "heading"}
    missing = sorted(required - set(scene_df.columns))
    if missing:
        raise ValueError(f"{agent_file} missing required columns: {missing}")

    scene_df = scene_df.copy()
    scene_df["agent_id_str"] = scene_df["agent_id"].astype(str)
    scene_df["scene_ts"] = scene_df["scene_ts"].astype(int)
    if agent_ids is not None:
        agent_ids = {str(agent_id) for agent_id in agent_ids}
        scene_df = scene_df[scene_df["agent_id_str"].isin(agent_ids)].copy()
    scene_df = add_motion_heading(
        scene_df,
        output_col=heading_col,
        source=heading_source,
        window_frames=heading_window_frames,
        min_displacement_m=heading_min_displacement,
    )
    return scene_df, scene_name, agent_file
