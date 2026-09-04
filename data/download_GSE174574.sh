#!/usr/bin/env bash
# ============================================================================
# Download the raw 10x matrices for GSE174574 from NCBI GEO.
# ============================================================================
# Run from the repository root:
#     bash data/download_GSE174574.sh
#
# Downloads ~250 MB. Requires curl (or wget) and tar.
#
# This script deliberately stops after unpacking. It does NOT move files into
# StrokeData/<sample>/ automatically, because GEO filenames are chosen by the
# submitter and the mapping from GSM accession to sham1/sham2/sham3/mcao1/
# mcao2/mcao3 must be read off the GEO page. Guessing it would silently change
# which animals form the held-out test set.
#
# See data/README.md for the required layout and the expected cell counts.
# ============================================================================

set -euo pipefail

ACC="GSE174574"
RAW_DIR="data/raw"
TARBALL="${RAW_DIR}/${ACC}_RAW.tar"
URL="https://www.ncbi.nlm.nih.gov/geo/download/?acc=${ACC}&format=file"

if [ ! -f "DP_Diff_Test_Train.R" ]; then
  echo "ERROR: run this from the repository root (DP_Diff_Test_Train.R not found here)." >&2
  exit 1
fi

mkdir -p "$RAW_DIR"

if [ -f "$TARBALL" ]; then
  echo "Archive already present: $TARBALL"
else
  echo "Downloading ${ACC} (~250 MB) ..."
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$TARBALL" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --show-progress -O "$TARBALL" "$URL"
  else
    echo "ERROR: neither curl nor wget is available." >&2
    exit 1
  fi
fi

echo
echo "Unpacking ..."
tar -xf "$TARBALL" -C "$RAW_DIR"

echo
echo "============================================================"
echo "Files unpacked into ${RAW_DIR}:"
echo "============================================================"
ls -la "$RAW_DIR" | grep -v '\.tar$' || true

cat <<'EOF'

============================================================
NEXT STEP - manual, on purpose
============================================================
1. Open the GEO page and note which GSM accession is which animal:
     https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174574

2. For each of the six samples, move its three files into the matching
   folder and strip the GSM prefix. Read10X() requires these exact names:

     StrokeData/<sample>/barcodes.tsv.gz
     StrokeData/<sample>/features.tsv.gz
     StrokeData/<sample>/matrix.mtx.gz

   where <sample> is one of: sham1 sham2 sham3 mcao1 mcao2 mcao3

   Example:
     mv data/raw/GSMxxxxxxx_sham1_barcodes.tsv.gz StrokeData/sham1/barcodes.tsv.gz

   If GEO ships "genes.tsv.gz" instead of "features.tsv.gz", rename it to
   features.tsv.gz - that is the Cell Ranger v2 vs v3 naming difference.

3. Sanity-check before running the pipeline:
     zcat StrokeData/sham1/barcodes.tsv.gz | wc -l    # expect 8771

4. Then run:
     Rscript DP_Diff_Test_Train.R

IMPORTANT: sham3 and mcao3 are the held-out test set. Assigning the
replicates differently will change your results relative to the paper.
============================================================
EOF
