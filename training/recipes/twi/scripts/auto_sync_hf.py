"""Auto-push converted 24k FLAC audio to Hugging Face every 20 minutes.

Once verified on Hugging Face:
- Keeps the converted file in data/twi/audio_vc/ locally.
- Deletes the original un-converted file in data/twi/audio/ to save disk space.
"""

import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "ghanaopenai/ghana-podcast-twi-female-vc"
INTERVAL_SEC = 20 * 60  # 20 minutes


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sync_once(api: HfApi, vc_dir: Path, orig_dir: Path, state_file: Path) -> int:
    synced = set()
    if state_file.exists():
        try:
            synced = set(json.loads(state_file.read_text()))
        except Exception:
            synced = set()

    vc_files = sorted(vc_dir.glob("*.flac"))
    to_sync = [f for f in vc_files if f.name not in synced and f.stat().st_size > 1024]

    if not to_sync:
        log(f"No new files to sync. Total already synced: {len(synced):,}")
        return 0

    log(f"Found {len(to_sync):,} newly converted files to push to HF...")
    newly_verified = 0
    bytes_freed = 0

    for i, f in enumerate(to_sync, 1):
        try:
            # Upload file
            api.upload_file(
                path_or_fileobj=str(f.resolve()),
                path_in_repo=f"audio/{f.name}",
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Add {f.name}",
            )
            # Verify file exists on remote
            exists = api.file_exists(
                repo_id=REPO_ID, filename=f"audio/{f.name}", repo_type="dataset"
            )
            if not exists:
                log(f"WARNING: Verification failed for {f.name}, skipping deletion.")
                continue

            synced.add(f.name)
            newly_verified += 1

            # Delete original un-converted file if it exists
            orig_file = orig_dir / f.name
            if orig_file.exists():
                sz = orig_file.stat().st_size
                orig_file.unlink()
                bytes_freed += sz

            if i % 5 == 0 or i == len(to_sync):
                log(
                    f"  Pushed & verified {i}/{len(to_sync)}: {f.name} (freed {bytes_freed / 1e6:.1f} MB original audio)"
                )

        except Exception as e:
            log(f"ERROR uploading {f.name}: {e}")

    # Save state
    state_file.write_text(json.dumps(sorted(list(synced))))
    log(
        f"Batch complete: {newly_verified:,} files pushed & verified. Freed {bytes_freed / 1e6:.1f} MB from original audio directory."
    )

    # Upload manifests if present
    manifests_dir = Path("data/twi/manifests_500h")
    if manifests_dir.exists():
        for mf in manifests_dir.glob("*.jsonl"):
            try:
                api.upload_file(
                    path_or_fileobj=str(mf.resolve()),
                    path_in_repo=f"manifests/{mf.name}",
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Update manifest {mf.name}",
                )
            except Exception as e:
                log(f"Error uploading manifest {mf.name}: {e}")

    return newly_verified


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    vc_dir = Path("data/twi/audio_vc").resolve()
    orig_dir = Path("data/twi/audio").resolve()
    state_file = Path("data/twi/synced_to_hf.json").resolve()

    vc_dir.mkdir(parents=True, exist_ok=True)
    log(f"Starting HF auto-sync daemon for {REPO_ID} (interval: {INTERVAL_SEC / 60:.0f} min)...")

    while True:
        if args.once:
            sync_once(api, vc_dir, orig_dir, state_file)
            log("Single sync pass complete!")
            break
        try:
            sync_once(api, vc_dir, orig_dir, state_file)
        except Exception as e:
            log(f"Unexpected error in sync loop: {e}")

        # Check if conversion process is still running
        p = os.popen("pgrep -af 'convert_dataset_vc' | grep -v pgrep").read().strip()
        if not p:
            log("Voice conversion process is no longer running. Running final sync pass...")
            try:
                sync_once(api, vc_dir, orig_dir, state_file)
            except Exception:
                pass
            log("Auto-sync daemon completed successfully!")
            break

        log(f"Sleeping {INTERVAL_SEC / 60:.0f} minutes until next sync...")
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
