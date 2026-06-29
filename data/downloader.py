"""
NeuroSWAYML — Dataset Downloader
=================================
Downloads all public datasets required for every NeuroSWAYML analysis domain.

Usage
-----
    python data/downloader.py                    # download everything
    python data/downloader.py --domain neuro     # gaitpdb + gaitndd only
    python data/downloader.py --domain elderly   # URFD  (~240 MB, auto)
    python data/downloader.py --domain intox     # HBEDB only
    python data/downloader.py --domain congen    # GaitRec instructions only
    python data/downloader.py --check            # check what's present

Dataset access levels
---------------------
• gaitpdb   (neuro)    — Open, PhysioNet wget
• gaitndd   (neuro)    — Open, PhysioNet wget
• URFD      (elderly)  — Open, auto-downloaded via urllib (~240 MB)
• ltmm      (elderly)  — Open, PhysioNet wget  [optional, ~20 GB]
• hbedb     (intox)    — Open, PhysioNet wget
• GaitRec   (congen)   — Open, figshare (manual download — large file 2.3 GB)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# ── Root paths ─────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent.parent   # NeuroSWAYML/
_DATA_ROOT   = _HERE / "data" / "physionet"
_GAITREC_DIR = _HERE / "data" / "gaitrec"
_URFD_DIR    = _HERE / "data" / "urfd"

# ── PhysioNet dataset catalogue ────────────────────────────────────────────
PHYSIONET_DATASETS = {
    "neuro": [
        {
            "slug":  "gaitpdb",
            "name":  "Gait in Parkinson's Disease (gaitpdb)",
            "url":   "https://physionet.org/files/gaitpdb/1.0.0/",
            "dest":  _DATA_ROOT / "gaitpdb" / "gait-in-parkinsons-disease-1.0.0",
            "check": "Ga01Co01_01.txt",
        },
        {
            "slug":  "gaitndd",
            "name":  "Gait in Neurodegenerative Diseases (gaitndd)",
            "url":   "https://physionet.org/files/gaitndd/1.0.0/",
            "dest":  _DATA_ROOT / "gaitndd",
            "check": "als1.ts",
        },
    ],
    "elderly": [
        {
            "slug":  "ltmm",
            "name":  "Long Term Movement Monitoring Database (LTMM)",
            "url":   "https://physionet.org/files/ltmm/1.0.0/",
            "dest":  _DATA_ROOT / "ltmm",
            "check": "LTMM_metadata.csv",
            # Each subject file is a 3-day recording (~300 MB).
            # default_max_subjects limits the download to a manageable subset.
            "default_max_subjects": 30,
        },
    ],
    "intox": [
        {
            "slug":  "hbedb",
            "name":  "Human Balance Evaluation Database (HBEDB)",
            "url":   "https://physionet.org/files/hbedb/1.0.0/",
            "dest":  _DATA_ROOT / "hbedb",
            "check": None,     # first CSV file in dir will do
        },
    ],
    # GaitRec is too large (2.3 GB) for automated wget — instructions only.
    "congen": [],
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _check_present(entry: dict) -> bool:
    dest: Path = entry["dest"]
    if not dest.exists():
        return False
    check = entry.get("check")
    if check:
        return (dest / check).exists() or any(dest.glob(f"**/{check}"))
    return any(dest.glob("*"))


def _wget_dataset(entry: dict):
    dest: Path = entry["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "-r", "-N", "-c", "-np",
        "--reject", "index.html*",
        "-P", str(dest),
        entry["url"],
    ]
    print(f"\n  Downloading: {entry['name']}")
    print(f"  URL        : {entry['url']}")
    print(f"  Destination: {dest}")
    print(f"  Command    : {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        # wget not installed — try urllib fallback (single file index)
        print("  [WARNING] wget not found. Attempting urllib download of file list…")
        _urllib_fallback(entry)
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] wget exited with code {e.returncode}.")


def _urllib_fallback(entry: dict):
    """Download index page + all linked .txt / .ts / .csv via urllib."""
    import urllib.request
    import urllib.parse
    import re

    base_url = entry["url"]
    dest: Path = entry["dest"]
    dest.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(base_url, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [ERROR] Cannot fetch file list: {e}")
        return

    hrefs = re.findall(r'href="([^"]+\.(?:txt|ts|csv|dat|gz))"', html)
    total = len(hrefs)
    print(f"  Found {total} files to download…")
    for i, href in enumerate(hrefs, 1):
        fname = href.split("/")[-1]
        furl  = urllib.parse.urljoin(base_url, href)
        fpath = dest / fname
        if fpath.exists():
            continue
        try:
            print(f"  [{i:3d}/{total}] {fname}", end="\r", flush=True)
            urllib.request.urlretrieve(furl, str(fpath))
        except Exception as e:
            print(f"  [WARN] {fname}: {e}")
    print()


def _urfd_download_and_extract():
    """
    Download URFD falls.zip + adls.zip and extract them.
    Delegates to URFDLoader which handles progress printing.
    """
    sys.path.insert(0, str(_HERE))
    try:
        from data.loaders.urfd_loader import URFDLoader
    except ImportError:
        # Fall back to inline download if loader not importable
        import urllib.request
        _URFD_DIR.mkdir(parents=True, exist_ok=True)
        base = "http://fenix.ur.edu.pl/~mkepski/ds/data"
        items = (
            [(f"fall-{i:02d}-cam0-rgb.zip", f"{base}/fall-{i:02d}-cam0-rgb.zip") for i in range(1, 31)]
            + [(f"adl-{i:02d}-cam0-rgb.zip",  f"{base}/adl-{i:02d}-cam0-rgb.zip")  for i in range(1, 41)]
        )
        for name, url in items:
            dest = _URFD_DIR / name
            if not dest.exists():
                print(f"  Downloading {name} …", end="\r", flush=True)
                urllib.request.urlretrieve(url, str(dest))
        print()
        return

    loader = URFDLoader(data_dir=str(_URFD_DIR))
    loader.download(verbose=True)
    loader.extract(verbose=True)


def _print_gaitrec_instructions():
    print("""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║            GaitRec — Congenital Domain Dataset                       ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║  Dataset      : GaitRec v1 (Horst et al., 2021)                     ║
  ║  Subjects     : 2,084  (healthy + orthopaedic / congenital)          ║
  ║  Size         : ~2.3 GB                                              ║
  ║  Licence      : CC BY 4.0                                            ║
  ║                                                                      ║
  ║  Step 1 — Visit                                                      ║
  ║    https://figshare.com/articles/dataset/                            ║
  ║    GaitRec_A_large-scale_ground_reaction_force_dataset_of_           ║
  ║    healthy_and_impaired_gait/13598962                                ║
  ║                                                                      ║
  ║  Step 2 — Click "Download all"  → GaitRec.zip                       ║
  ║                                                                      ║
  ║  Step 3 — Extract to:                                                ║
  ║    data/gaitrec/                                                     ║
  ║                                                                      ║
  ║  Step 4 — Run training:                                              ║
  ║    python training/train_congenital.py                              ║
  ╚══════════════════════════════════════════════════════════════════════╝
""")


def _print_intox_extra():
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Optional: Kaiserslautern Intoxication IMU Dataset                   │
  │  Paper : Muaaz & Mayrhofer, "An Analysis of Different Approaches    │
  │          to Gait-based Intoxication Detection Using Smartphones"    │
  │  (IEEE Sensors 2015, IEEE DataPort / contact lab for data)          │
  │                                                                     │
  │  If you obtain the dataset, place CSV files in:                     │
  │    data/intoxication/                                               │
  │  Format: time, acc_x, acc_y, acc_z, label   (label 0/1/2)          │
  └─────────────────────────────────────────────────────────────────────┘
""")


# ── Status check ───────────────────────────────────────────────────────────

def check_all():
    print("\n  NeuroSWAYML — Dataset Status\n  " + "─" * 60)
    all_domains = {
        "neuro": PHYSIONET_DATASETS["neuro"],
        "elderly": PHYSIONET_DATASETS["elderly"],
        "intox": PHYSIONET_DATASETS["intox"],
        "congen": [],
    }
    for domain, entries in all_domains.items():
        if domain == "congen":
            ok = _GAITREC_DIR.exists() and any(_GAITREC_DIR.glob("**/*.csv"))
            status = "✓ Found" if ok else "✗ Missing"
            print(f"  [{domain:8s}] GaitRec          {status}")
        elif domain == "elderly":
            # URFD (primary)
            urfd_cache = _URFD_DIR / "features_cache.npz"
            urfd_seqs  = list(_URFD_DIR.rglob("fall-*")) + list(_URFD_DIR.rglob("adl-*"))
            if urfd_cache.exists():
                urfd_status = "✓ Processed (cache ready)"
            elif urfd_seqs:
                urfd_status = "✓ Extracted (not yet processed)"
            elif (_URFD_DIR / "urfd-falls.zip").exists():
                urfd_status = "└ Downloaded (not extracted)"
            else:
                urfd_status = "✗ Missing"
            print(f"  [{domain:8s}] URFD (primary)   {urfd_status}")
            # LTMM (optional)
            for e in entries:
                ok     = _check_present(e)
                status = "✓ Found" if ok else "✗ Missing (optional)"
                print(f"  [{domain:8s}] {e['name'][:35]:35s} {status}")
        else:
            for e in entries:
                ok     = _check_present(e)
                status = "✓ Found" if ok else "✗ Missing"
                print(f"  [{domain:8s}] {e['name'][:35]:35s} {status}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────

def _ltmm_targeted_download(entry: dict, max_subjects: int):
    """
    Download LTMM metadata CSV first, then up to *max_subjects* subject .txt files.
    This avoids pulling the full ~20 GB dataset when only a subset is needed.
    """
    import urllib.request, urllib.parse, re

    base_url = entry["url"]
    dest: Path = entry["dest"]
    dest.mkdir(parents=True, exist_ok=True)

    # ── 1. Fetch directory listing ─────────────────────────────────────────
    print(f"  Fetching LTMM file index from {base_url} …")
    try:
        with urllib.request.urlopen(base_url, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [ERROR] Cannot reach PhysioNet: {e}")
        print("  → Check your internet connection or try later.")
        return

    all_hrefs = re.findall(r'href="([^"]+)"', html)

    # ── 2. Always download metadata first ─────────────────────────────────
    meta_hrefs = [h for h in all_hrefs if "metadata" in h.lower() or h.endswith(".csv")]
    data_hrefs = [h for h in all_hrefs if h.endswith(".txt") or h.endswith(".dat")]

    to_download = meta_hrefs + data_hrefs[:max_subjects]
    total = len(to_download)
    print(f"  Downloading metadata + {min(max_subjects, len(data_hrefs))} of "
          f"{len(data_hrefs)} subject files  (~{min(max_subjects, len(data_hrefs)) * 300} MB estimated)")

    for i, href in enumerate(to_download, 1):
        fname = href.split("/")[-1]
        furl  = urllib.parse.urljoin(base_url, href) if not href.startswith("http") else href
        fpath = dest / fname
        if fpath.exists():
            print(f"  [{i:3d}/{total}] already exists: {fname}")
            continue
        try:
            print(f"  [{i:3d}/{total}] Downloading {fname} …", end="", flush=True)
            urllib.request.urlretrieve(furl, str(fpath))
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f" {size_mb:.1f} MB")
        except Exception as e:
            print(f" FAILED: {e}")

    print(f"  [LTMM] Done. {len(list(dest.glob('*.txt')))} subject files in {dest}")


def main():
    parser = argparse.ArgumentParser(
        description="NeuroSWAYML Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--domain", choices=["neuro", "elderly", "intox", "congen", "all"],
        default="all", help="Which domain's datasets to download (default: all)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only check dataset presence, do not download",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=None,
        help="For LTMM: max subject files to download (default: 30). "
             "Use 71 for the full dataset (~20 GB).",
    )
    args = parser.parse_args()

    if args.check:
        check_all()
        return

    domains = (
        ["neuro", "elderly", "intox", "congen"]
        if args.domain == "all"
        else [args.domain]
    )

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"  Domain: {domain.upper()}")
        print(f"{'='*60}")

        if domain == "congen":
            _print_gaitrec_instructions()
            continue

        if domain == "elderly":
            # Primary: URFD (~240 MB, video-based, zero domain gap)
            urfd_cache = _URFD_DIR / "features_cache.npz"
            urfd_seqs  = list(_URFD_DIR.rglob("fall-*")) + list(_URFD_DIR.rglob("adl-*"))
            if urfd_cache.exists() or urfd_seqs:
                print(f"  ✓ URFD already present in {_URFD_DIR}")
                print(f"    (run  python training/train_elderly.py  to train)")
            else:
                print(f"\n  Downloading URFD dataset → {_URFD_DIR}")
                print(f"  ~240 MB total (urfd-falls.zip ~50 MB + urfd-adls.zip ~190 MB)")
                _urfd_download_and_extract()
            # Optional: LTMM (much larger, skip by default)
            print("\n  NOTE: LTMM (~20 GB) is optional. To download:")
            print("        python data/downloader.py --domain elderly --include-ltmm")
            continue

        if domain == "intox":
            _print_intox_extra()

        for entry in PHYSIONET_DATASETS.get(domain, []):
            if _check_present(entry):
                print(f"  ✓ Already present: {entry['name']}")
                continue

            # LTMM: use targeted download to avoid pulling ~20 GB
            if entry["slug"] == "ltmm":
                n = args.max_subjects or entry.get("default_max_subjects", 30)
                print(f"\n  NOTE: Each LTMM subject file is ~300 MB (3-day recording).")
                print(f"  Downloading {n} of 71 subjects. Use --max-subjects 71 for all.")
                _ltmm_targeted_download(entry, max_subjects=n)
            else:
                _wget_dataset(entry)

    print("\n  Download complete. Run  python training/train_all.py  to train all models.\n")


if __name__ == "__main__":
    main()
