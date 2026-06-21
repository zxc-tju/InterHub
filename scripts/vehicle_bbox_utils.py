import numpy as np
import pandas as pd


def add_motion_heading(
    scene_df: pd.DataFrame,
    output_col: str = "bbox_heading",
    source: str = "raw",
    window_frames: int = 3,
    min_displacement_m: float = 0.2,
) -> pd.DataFrame:
    """Add a heading column for vehicle boxes.

    By default the box is aligned to the heading already stored in the cache.
    The trajectory option estimates heading from nearby observed positions and falls
    back to raw heading for very slow frames.
    """
    if source not in {"trajectory", "raw"}:
        raise ValueError(f"unsupported heading source: {source}")

    result = scene_df.copy()
    if "agent_id_str" not in result.columns:
        result["agent_id_str"] = result["agent_id"].astype(str)

    raw_heading = result["heading"].astype(float)
    result[output_col] = raw_heading
    result[f"{output_col}_method"] = "raw"
    result[f"{output_col}_displacement_m"] = 0.0
    if source == "raw":
        return result

    window_frames = max(1, int(window_frames))
    min_displacement_m = float(min_displacement_m)

    for _, group in result.groupby("agent_id_str", sort=False):
        sorted_group = group.sort_values("scene_ts")
        indices = sorted_group.index.to_numpy()
        x = sorted_group["x"].to_numpy(dtype=float)
        y = sorted_group["y"].to_numpy(dtype=float)
        n = len(sorted_group)
        if n < 2:
            continue

        headings = result.loc[indices, output_col].to_numpy(dtype=float)
        methods = np.asarray(["raw"] * n, dtype=object)
        displacements = np.zeros(n, dtype=float)

        for i in range(n):
            lo = max(0, i - window_frames)
            hi = min(n - 1, i + window_frames)
            if hi == lo:
                continue
            dx = x[hi] - x[lo]
            dy = y[hi] - y[lo]
            displacement = float(np.hypot(dx, dy))
            displacements[i] = displacement
            if displacement >= min_displacement_m and np.isfinite(dx) and np.isfinite(dy):
                headings[i] = float(np.arctan2(dy, dx))
                methods[i] = "trajectory"

        result.loc[indices, output_col] = headings
        result.loc[indices, f"{output_col}_method"] = methods
        result.loc[indices, f"{output_col}_displacement_m"] = displacements

    return result
