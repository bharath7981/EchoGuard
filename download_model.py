"""
Download pre-trained EchoGuard model weights from GitHub Releases.
Run this once after cloning the repo to skip training.

Usage:
    python download_model.py
"""

import os
import sys
import urllib.request

# ============================================================
# CONFIGURATION — Update REPO_OWNER after pushing to GitHub
# ============================================================
REPO_OWNER = "YOUR_GITHUB_USERNAME"  # ← Change this to your GitHub username
REPO_NAME = "EchoGuard"
TAG = "v1.0"

MODELS = {
    "rawboost_ssl_best_model.pt": {
        "description": "Best Model (wav2vec2 + MLP + RawBoost) — ASVspoof EER: 3.54%, In-the-Wild EER: 26.01%",
        "required": True,
    },
    "best_model.pt": {
        "description": "Phase 1 CNN Baseline — ASVspoof EER: 6.16%",
        "required": False,
    },
    "ssl_best_model.pt": {
        "description": "Model A (Clean SSL) — ASVspoof EER: 2.72%, In-the-Wild EER: 26.94%",
        "required": False,
    },
    "clean_ocsoftmax_best_model.pt": {
        "description": "Model C (OC-Softmax) — ASVspoof EER: 3.15%, In-the-Wild EER: 31.19%",
        "required": False,
    },
    "rawboost_ocsoftmax_best_model.pt": {
        "description": "Model D (RawBoost + OC-Softmax) — ASVspoof EER: 3.49%, In-the-Wild EER: 34.38%",
        "required": False,
    },
}

CHECKPOINT_DIR = "checkpoints"


def download_file(url: str, dest: str, description: str):
    """Download a file with progress bar."""
    print(f"\n📥 Downloading: {os.path.basename(dest)}")
    print(f"   {description}")
    print(f"   URL: {url}")

    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                bar_len = 40
                filled = int(bar_len * percent / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stdout.write(f"\r   [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest, reporthook=reporthook)
        print(f"\n   ✅ Saved to {dest}")
        return True
    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  EchoGuard — Pre-trained Model Downloader")
    print("=" * 60)

    if REPO_OWNER == "YOUR_GITHUB_USERNAME":
        print("\n⚠️  Please edit download_model.py and set REPO_OWNER to your GitHub username.")
        print("   Then create a GitHub Release tagged 'v1.0' and upload the .pt files.")
        print("\n   Alternatively, download manually and place in checkpoints/")
        sys.exit(1)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Download required model first
    print("\n📦 Downloading required model checkpoint...")
    for filename, info in MODELS.items():
        if not info["required"]:
            continue
        dest = os.path.join(CHECKPOINT_DIR, filename)
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"\n✅ {filename} already exists ({size_mb:.1f} MB) — skipping")
            continue
        url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{TAG}/{filename}"
        success = download_file(url, dest, info["description"])
        if not success:
            print(f"\n❌ Failed to download required model. Cannot proceed.")
            print(f"   Please download manually from:")
            print(f"   https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/tag/{TAG}")
            sys.exit(1)

    # Ask about optional models
    print("\n" + "-" * 60)
    print("Optional: Download all ablation study models? (Model A/C/D + CNN Baseline)")
    choice = input("Download all models? [y/N]: ").strip().lower()

    if choice == "y":
        for filename, info in MODELS.items():
            if info["required"]:
                continue
            dest = os.path.join(CHECKPOINT_DIR, filename)
            if os.path.exists(dest):
                size_mb = os.path.getsize(dest) / (1024 * 1024)
                print(f"\n✅ {filename} already exists ({size_mb:.1f} MB) — skipping")
                continue
            url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{TAG}/{filename}"
            download_file(url, dest, info["description"])

    print("\n" + "=" * 60)
    print("✅ Model download complete!")
    print("   Run the app with: python run_server.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
