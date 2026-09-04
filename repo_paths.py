"""
Repository path resolution
==========================
Small helper so every script finds its input data regardless of where the
repository is checked out.

The original analysis was run partly on Google Colab, where the data lived
under /content/drive/MyDrive/. Those locations are kept as fallbacks, but the
repository root is searched first so a plain `git clone` works out of the box.

Usage:
    from repo_paths import data_path, REPO_ROOT
    train = pd.read_csv(data_path('stroke_pca_train.csv'))
"""

from pathlib import Path

# Directory containing this file == repository root
REPO_ROOT = Path(__file__).resolve().parent

# Searched in order; first hit wins.
SEARCH_PATHS = [
    REPO_ROOT,
    Path("/content/Phase4_RevisedModels"),
    Path("/content/drive/MyDrive/stroke/Phase4_RevisedModels"),
    Path("/content/drive/MyDrive/stroke_project/Phase4_RevisedModels"),
]


def data_path(filename):
    """Return an absolute path to `filename`, searching known data locations.

    Raises FileNotFoundError with actionable guidance if the file is absent,
    which is the common case for the two large PCA matrices that are not
    distributed with the repository (see data/README.md).
    """
    for base in SEARCH_PATHS:
        candidate = base / filename
        if candidate.exists():
            return str(candidate)

    searched = "\n".join("  - %s" % (b / filename) for b in SEARCH_PATHS)
    raise FileNotFoundError(
        "Could not find '%s'.\n"
        "Searched:\n%s\n\n"
        "If this is 'stroke_pca_train.csv' or 'stroke_pca_test.csv', these are\n"
        "intermediate files that are NOT distributed with the repository "
        "because of their size.\n"
        "Regenerate them by running the R preprocessing step first:\n"
        "    Rscript DP_Diff_Test_Train.R\n"
        "See data/README.md for how to obtain the raw GSE174574 input."
        % (filename, searched)
    )


def data_dir():
    """Return the directory that holds the PCA/label CSVs."""
    for base in SEARCH_PATHS:
        if (base / "stroke_labels_train.csv").exists():
            return str(base)
    return str(REPO_ROOT)
