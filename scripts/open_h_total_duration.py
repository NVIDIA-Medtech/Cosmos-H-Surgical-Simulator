#!/usr/bin/env python3
"""
Compute total duration of LeRobot-formatted datasets by summing video durations,
using imageio (with optional ffprobe fallback), with multiprocessing for speed.

Usage:
  # Compute duration for specific dataset roots:
  python3 dataset_total_duration.py \
    /CMR_Versius/cholecystectomy_480p \
    /CMR_Versius/hysterectomy_480p \
    --per-root

  # Auto-discover LeRobot datasets in a directory:
  python3 dataset_total_duration.py /CMR_Versius --discover

  # Get detailed per-episode breakdown:
  python3 dataset_total_duration.py /SutureBot --detailed

Notes:
- LeRobot format expects: <dataset_root>/videos/, <dataset_root>/meta/, and optionally <dataset_root>/data/
- Supports videos under videos/ in various formats: .mp4, .avi, .mov, .webm, .mkv
- Duration estimation prefers metadata (nframes/fps, then duration).
- Falls back to ffprobe if imageio fails (requires ffprobe in PATH).
- Frame counting is used as a last resort (can be slow).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Video extensions to look for
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".MP4", ".AVI", ".MOV"}


def hms_from_seconds(total_seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    if not (total_seconds >= 0.0) or math.isinf(total_seconds) or math.isnan(total_seconds):
        total_seconds = 0.0
    secs = int(round(total_seconds))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def iter_videos(dataset_root: Path) -> List[Path]:
    """Find all video files under dataset_root/videos/."""
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        return []
    
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(videos_dir.rglob(f"*{ext}"))
    
    return sorted(p for p in videos if p.is_file())


def is_lerobot_root(p: Path, strict: bool = False) -> bool:
    """
    Check if a path looks like a LeRobot dataset root.
    
    Args:
        p: Path to check
        strict: If True, require data/ directory as well (some datasets only have videos/ + meta/)
    
    Returns:
        True if path appears to be a LeRobot dataset root
    """
    if not p.is_dir():
        return False
    
    # Minimum requirements: videos/ and meta/
    has_videos = (p / "videos").exists()
    has_meta = (p / "meta").exists()
    
    if strict:
        has_data = (p / "data").exists()
        return has_videos and has_meta and has_data
    
    return has_videos and has_meta


def discover_lerobot_roots(base_path: Path, max_depth: int = 3) -> List[Path]:
    """
    Recursively discover LeRobot dataset roots under a base path.
    
    Args:
        base_path: Starting directory to search
        max_depth: Maximum directory depth to search
    
    Returns:
        List of discovered LeRobot dataset root paths
    """
    discovered = []
    
    def _search(current: Path, depth: int):
        if depth > max_depth:
            return
        
        if is_lerobot_root(current):
            discovered.append(current)
            return  # Don't search inside a dataset
        
        if not current.is_dir():
            return
        
        try:
            for child in current.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    _search(child, depth + 1)
        except PermissionError:
            pass
    
    _search(base_path, 0)
    return sorted(discovered)


def video_duration_ffprobe(video_path: str) -> Optional[float]:
    """
    Get video duration using ffprobe (external tool).
    Returns None if ffprobe is not available or fails.
    """
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return None
    
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return None
        
        data = json.loads(result.stdout)
        
        # Try format duration first
        if "format" in data and "duration" in data["format"]:
            return float(data["format"]["duration"])
        
        # Try stream duration
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and "duration" in stream:
                return float(stream["duration"])
        
        return None
    except Exception:
        return None


def _is_valid_duration(dur: float) -> bool:
    """Check if a duration value is valid (finite, positive, reasonable)."""
    if dur is None:
        return False
    if not isinstance(dur, (int, float)):
        return False
    if math.isinf(dur) or math.isnan(dur):
        return False
    if dur <= 0:
        return False
    # Sanity check: reject durations longer than 24 hours for a single video
    if dur > 86400:
        return False
    return True


def video_duration_imageio(video_path_str: str) -> float:
    """
    Get video duration using imageio.
    Falls back to ffprobe, then frame counting if necessary.
    
    Args:
        video_path_str: String path to video file (pickleable for multiprocessing)
    
    Returns:
        Duration in seconds
    
    Raises:
        RuntimeError: If duration cannot be determined
    """
    video_path = Path(video_path_str)
    
    # Try imageio first (faster for most cases)
    try:
        import imageio.v3 as iio
        
        # Try with FFMPEG plugin explicitly
        try:
            meta = iio.immeta(video_path, plugin="FFMPEG")
        except Exception:
            meta = iio.immeta(video_path)
        
        fps = meta.get("fps", None)
        
        # If we have valid fps, try to compute duration
        if fps is not None and fps > 0 and not math.isinf(fps):
            # Try nframes first
            nframes = meta.get("nframes", None)
            if isinstance(nframes, (int, float)) and nframes > 0 and not math.isinf(nframes):
                computed_dur = float(nframes) / float(fps)
                if _is_valid_duration(computed_dur):
                    return computed_dur
            
            # Try duration directly
            dur = meta.get("duration", None)
            if _is_valid_duration(dur):
                return float(dur)
    except Exception:
        pass
    
    # Fallback: try ffprobe
    ffprobe_dur = video_duration_ffprobe(video_path_str)
    if _is_valid_duration(ffprobe_dur):
        return ffprobe_dur
    
    # Last resort: count frames (slow)
    try:
        import imageio.v3 as iio
        
        # Get fps first
        try:
            meta = iio.immeta(video_path, plugin="FFMPEG")
        except Exception:
            meta = iio.immeta(video_path)
        
        fps = meta.get("fps", 30.0)  # Default to 30 fps if unknown
        if fps is None or fps <= 0 or math.isinf(fps):
            fps = 30.0
        
        # Count frames
        count = 0
        try:
            for _ in iio.imiter(video_path, plugin="FFMPEG"):
                count += 1
        except Exception:
            for _ in iio.imiter(video_path):
                count += 1
        
        if count > 0:
            computed_dur = float(count) / float(fps)
            if _is_valid_duration(computed_dur):
                return computed_dur
    except Exception:
        pass
    
    raise RuntimeError(f"Could not determine duration for {video_path}")


def sum_durations_multiproc(
    videos: Iterable[Path], 
    workers: int,
    show_progress: bool = False,
) -> Tuple[float, int, List[str], Dict[Path, float]]:
    """
    Sum durations of video files using multiprocessing.
    
    Args:
        videos: Iterable of video file paths
        workers: Number of worker processes
        show_progress: Whether to show a progress bar
    
    Returns:
        Tuple of (total_duration, num_files, errors, per_file_durations)
    """
    video_list = list(videos)
    if not video_list:
        return 0.0, 0, [], {}
    
    total = 0.0
    errors: List[str] = []
    durations: Dict[Path, float] = {}
    
    # Use spawn for safety in mixed environments (containers/HPC)
    ctx = mp.get_context("spawn")
    
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            fut_to_path = {ex.submit(video_duration_imageio, str(p)): p for p in video_list}
            
            completed = 0
            for fut in as_completed(fut_to_path):
                p = fut_to_path[fut]
                completed += 1
                
                if show_progress:
                    print(f"\r  Processing: {completed}/{len(video_list)} videos", end="", flush=True)
                
                try:
                    dur = fut.result()
                    # Final safety check for valid duration
                    if _is_valid_duration(dur):
                        total += dur
                        durations[p] = dur
                    else:
                        errors.append(f"{p}: invalid duration value ({dur})")
                except Exception as e:
                    errors.append(f"{p}: {e}")
            
            if show_progress:
                print()  # Newline after progress
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        raise
    
    return total, len(video_list), errors, durations


def get_dataset_info(dataset_root: Path) -> Dict:
    """
    Get additional info about a LeRobot dataset.
    
    Returns:
        Dictionary with dataset metadata
    """
    info = {
        "has_videos": (dataset_root / "videos").exists(),
        "has_meta": (dataset_root / "meta").exists(),
        "has_data": (dataset_root / "data").exists(),
        "num_episodes": None,
        "total_frames": None,
    }
    
    # Try to read episode info from meta/episodes.jsonl or similar
    episodes_file = dataset_root / "meta" / "episodes.jsonl"
    if episodes_file.exists():
        try:
            with open(episodes_file, "r") as f:
                episodes = [json.loads(line) for line in f if line.strip()]
                info["num_episodes"] = len(episodes)
                
                # Sum total frames if available
                total_frames = 0
                for ep in episodes:
                    if "length" in ep:
                        total_frames += ep["length"]
                if total_frames > 0:
                    info["total_frames"] = total_frames
        except Exception:
            pass
    
    # Fallback: count video files in videos/
    if info["num_episodes"] is None:
        videos = iter_videos(dataset_root)
        # Estimate episodes from video directory structure
        episode_dirs = set()
        for v in videos:
            # Videos are typically in videos/observation.images.main/episode_000000.mp4
            # or videos/chunk-000/episode_000000/...
            rel_path = v.relative_to(dataset_root / "videos")
            if len(rel_path.parts) >= 1:
                episode_dirs.add(rel_path.parts[0] if "episode" in str(rel_path.parts[0]) else rel_path.parent.name)
    
    return info


def _get_duration_from_metadata(dataset_path: Path) -> Optional[Tuple[float, int, int, float]]:
    """Compute dataset duration from metadata only (no video decoding).

    Reads episodes.jsonl for frame counts and info.json for FPS.
    Returns (duration_seconds, n_episodes, total_frames, fps) or None if metadata is missing.
    """
    info_path = dataset_path / "meta" / "info.json"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"

    if not info_path.exists() or not episodes_path.exists():
        return None

    try:
        with open(info_path, "r") as f:
            info = json.load(f)
        fps = info.get("fps", None)
        if fps is None or fps <= 0:
            fps = 30.0  # safe default

        with open(episodes_path, "r") as f:
            episodes = [json.loads(line) for line in f if line.strip()]

        n_episodes = len(episodes)
        total_frames = sum(ep.get("length", 0) for ep in episodes)
        duration_s = total_frames / fps

        return duration_s, n_episodes, total_frames, fps
    except Exception:
        return None


def run_open_h(args) -> int:
    """Compute duration for all datasets in OPEN_H_DATASET_SPECS.

    Uses metadata-only approach (episodes.jsonl + info.json) for speed and
    robustness — no video decoding, no ffmpeg hangs.
    """
    from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.embodiment_tags import EmbodimentTag
    from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import OPEN_H_DATASET_SPECS

    # Deduplicate specs by path
    seen: set[str] = set()
    specs: list[dict] = []
    for spec in OPEN_H_DATASET_SPECS:
        p = spec["path"]
        if p in seen:
            continue
        seen.add(p)
        emb = spec["embodiment"]
        if isinstance(emb, EmbodimentTag):
            emb = emb.value
        specs.append({"path": p, "embodiment": emb, "mix_ratio": spec.get("mix_ratio", 1.0)})

    print("=" * 100)
    print("OPEN-H COLLECTIVE — TOTAL VIDEO DURATION (metadata-based, no video decoding)")
    print("=" * 100)
    print(f"Datasets in OPEN_H_DATASET_SPECS: {len(specs)}")
    print("=" * 100)

    # Per-embodiment aggregation
    per_embodiment: dict[str, dict] = {}
    grand_total_seconds = 0.0
    grand_total_frames = 0
    grand_total_episodes = 0
    skipped = []

    for i, spec in enumerate(specs, 1):
        dp = Path(spec["path"])
        emb = spec["embodiment"]
        ratio = spec["mix_ratio"]

        if not dp.exists():
            skipped.append(f"{emb}/{dp.name} (path missing)")
            continue

        result = _get_duration_from_metadata(dp)
        if result is None:
            skipped.append(f"{emb}/{dp.name} (metadata missing)")
            continue

        dur_s, n_episodes, total_frames, fps = result
        hms = hms_from_seconds(dur_s)

        print(f"  [{i:2d}/{len(specs)}] [{emb:<24s}] {dp.name:<40s}  "
              f"{hms:>10s}  ({n_episodes:,} eps, {total_frames:,} frames, {fps:.0f}Hz)")

        grand_total_seconds += dur_s
        grand_total_frames += total_frames
        grand_total_episodes += n_episodes

        # Aggregate by embodiment
        if emb not in per_embodiment:
            per_embodiment[emb] = {"seconds": 0.0, "frames": 0, "episodes": 0, "datasets": 0, "mix_ratio": 0.0}
        per_embodiment[emb]["seconds"] += dur_s
        per_embodiment[emb]["frames"] += total_frames
        per_embodiment[emb]["episodes"] += n_episodes
        per_embodiment[emb]["datasets"] += 1
        per_embodiment[emb]["mix_ratio"] += ratio

    # Print summary
    print(f"\n{'='*100}")
    print("PER-EMBODIMENT SUMMARY")
    print(f"{'='*100}")
    print(f"{'Embodiment':<26s} {'Datasets':>8s} {'Episodes':>10s} {'Frames':>14s} {'Duration':>12s} {'Mix Ratio':>10s}")
    print(f"{'-'*100}")

    for emb in sorted(per_embodiment.keys()):
        s = per_embodiment[emb]
        print(f"{emb:<26s} {s['datasets']:>8d} {s['episodes']:>10,d} {s['frames']:>14,d} "
              f"{hms_from_seconds(s['seconds']):>12s} {s['mix_ratio']:>10.3f}")

    print(f"{'-'*100}")
    print(f"{'TOTAL':<26s} {sum(s['datasets'] for s in per_embodiment.values()):>8d} "
          f"{grand_total_episodes:>10,d} {grand_total_frames:>14,d} "
          f"{hms_from_seconds(grand_total_seconds):>12s} "
          f"{sum(s['mix_ratio'] for s in per_embodiment.values()):>10.3f}")
    print(f"{'='*100}")

    if skipped:
        print(f"\nSkipped {len(skipped)} dataset(s):")
        for s in skipped:
            print(f"  - {s}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute total video duration for LeRobot dataset roots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute duration for ALL Open-H datasets (from OPEN_H_DATASET_SPECS):
  python open_h_total_duration.py --open-h

  # Basic usage with multiple dataset roots:
  python open_h_total_duration.py /path/to/dataset1 /path/to/dataset2 --per-root
  
  # Auto-discover LeRobot datasets in a directory:
  python open_h_total_duration.py /data/datasets --discover
  
  # Get detailed output with per-episode breakdown:
  python open_h_total_duration.py /path/to/dataset --detailed --per-root
        """,
    )
    parser.add_argument(
        "dataset_roots",
        nargs="*",
        help="One or more LeRobot dataset root paths (or parent directories with --discover). "
             "Not required when using --open-h.",
    )
    parser.add_argument(
        "--open-h",
        action="store_true",
        help="Compute duration for ALL datasets in OPEN_H_DATASET_SPECS (including CMR). "
             "No dataset_roots arguments needed.",
    )
    parser.add_argument(
        "--per-root",
        action="store_true",
        help="Print total duration per dataset root (one line each).",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Auto-discover LeRobot datasets under the given paths.",
    )
    parser.add_argument(
        "--discover-depth",
        type=int,
        default=3,
        help="Maximum depth for --discover (default: 3).",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-video duration breakdown.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(4, (os.cpu_count() or 8)),
        help="Number of worker processes (default: max(4, cpu_count)).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any dataset root is missing expected dirs or any read error occurs.",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Require data/ directory when validating LeRobot roots (stricter check).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress warnings and progress output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )

    args = parser.parse_args()

    # Dispatch to Open-H mode if requested
    if args.open_h:
        return run_open_h(args)

    if not args.dataset_roots:
        parser.error("dataset_roots are required (or use --open-h)")

    # Collect all roots to process
    roots: List[Path] = []
    
    for path_str in args.dataset_roots:
        p = Path(path_str).resolve()
        
        if not p.exists():
            if not args.quiet:
                print(f"[WARN] Path does not exist: {p}", file=sys.stderr)
            if args.strict:
                return 1
            continue
        
        if args.discover:
            discovered = discover_lerobot_roots(p, max_depth=args.discover_depth)
            if discovered:
                if not args.quiet:
                    print(f"Discovered {len(discovered)} LeRobot dataset(s) under {p}", file=sys.stderr)
                roots.extend(discovered)
            elif is_lerobot_root(p, strict=args.require_data):
                roots.append(p)
            else:
                if not args.quiet:
                    print(f"[WARN] No LeRobot datasets found under: {p}", file=sys.stderr)
        else:
            roots.append(p)
    
    if not roots:
        print("No dataset roots to process.", file=sys.stderr)
        return 1
    
    # Remove duplicates while preserving order
    seen = set()
    unique_roots = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            unique_roots.append(r)
    roots = unique_roots
    
    grand_total = 0.0
    any_error = False
    results = []

    for root in roots:
        # Validate LeRobot structure
        if not is_lerobot_root(root, strict=args.require_data):
            expected = "videos/, meta/, and data/" if args.require_data else "videos/ and meta/"
            if not args.quiet:
                print(
                    f"[WARN] {root} does not look like a LeRobot root (expected {expected}).",
                    file=sys.stderr,
                )
            if args.strict:
                any_error = True
                continue

        videos = iter_videos(root)
        
        if not videos:
            if not args.quiet:
                print(f"[WARN] No video files found in {root / 'videos'}", file=sys.stderr)
            results.append({
                "root": str(root),
                "duration_seconds": 0.0,
                "duration_hms": "00:00:00",
                "num_videos": 0,
                "errors": [],
            })
            continue
        
        show_progress = not args.quiet and not args.json
        total_s, n_files, errors, durations = sum_durations_multiproc(
            videos, workers=args.workers, show_progress=show_progress
        )
        grand_total += total_s

        if errors:
            any_error = True
            if not args.quiet:
                print(f"[WARN] {root}: {len(errors)} file(s) failed to read.", file=sys.stderr)
                for line in errors[:5]:
                    print(f"  {line}", file=sys.stderr)
                if len(errors) > 5:
                    print(f"  ... ({len(errors) - 5} more)", file=sys.stderr)

        result = {
            "root": str(root),
            "duration_seconds": total_s,
            "duration_hms": hms_from_seconds(total_s),
            "num_videos": n_files,
            "errors": errors[:10] if errors else [],
        }
        
        if args.detailed:
            result["videos"] = {
                str(p.relative_to(root)): {"duration_seconds": d, "duration_hms": hms_from_seconds(d)}
                for p, d in sorted(durations.items())
            }
        
        results.append(result)

        if args.per_root and not args.json:
            info = get_dataset_info(root)
            extra = ""
            if info.get("num_episodes"):
                extra += f", ~{info['num_episodes']} episodes"
            print(f"{root}\t{hms_from_seconds(total_s)}\t({n_files} videos{extra})")
            
            if args.detailed:
                for p, d in sorted(durations.items()):
                    rel = p.relative_to(root)
                    print(f"  {rel}\t{hms_from_seconds(d)}")

    # Output final results
    if args.json:
        output = {
            "datasets": results,
            "total_duration_seconds": grand_total,
            "total_duration_hms": hms_from_seconds(grand_total),
            "total_datasets": len(results),
            "total_videos": sum(r["num_videos"] for r in results),
        }
        print(json.dumps(output, indent=2))
    else:
        if len(roots) > 1 or not args.per_root:
            total_videos = sum(r["num_videos"] for r in results)
            print(f"TOTAL\t{hms_from_seconds(grand_total)}\t({len(results)} datasets, {total_videos} videos)")

    if args.strict and any_error:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

