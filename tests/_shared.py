"""Shared sample-loading helpers for the feature-embedding and isoforest-leaf-embedding
notebooks, so both notebooks compare the same PSMs.

Mirrors the feature set the SUOD_model.pkl (grove_forest_new_features) was trained on --
see ../iforest_config.yaml -> iforest_features, and how features are scored per side in
../../../../../../src/python/train_suod.py (predict_search_set / build_test_set).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GROVE_FOREST_DIR = Path(
    "/cmnfs/proj/denovo_fdr/results/proteometools/results/grovems/"
    "run.19082026.grovems.6547926/rescoring/grove_forest_new_features"
)
RESULTS_DIR = GROVE_FOREST_DIR / "results"
MODEL_PATH = GROVE_FOREST_DIR / "model" / "SUOD_model.pkl"
EXPECTED_CSV = Path(
    "/cmnfs/proj/denovo_fdr/results/proteometools/metadata/expected_peptides/all_expected_sequences.csv"
)

FEATURES = [
    "abs_rt_diff",
    "fraction_not_observed_but_predicted",
    "fraction_not_observed_but_predicted_vs_predicted",
    "fraction_observed",
    "fraction_observed_and_predicted",
    "fraction_observed_and_predicted_vs_predicted",
    "spectral_angle",
]


def _il(s: pd.Series) -> pd.Series:
    """I -> L on a Series. I and L are isomers, so they must not count as a mismatch --
    same collapse task08_auc_new_features.ipynb uses."""
    return s.str.replace("I", "L", regex=False)


def _pool_of(path: Path) -> str:
    """Pool name from a results filename, e.g.
    01625b_GA1-TUM_first_pool_1_01_01-3xHCD-1h-R1.parquet -> TUM_first_pool_1_01_01 --
    same convention task08_auc_new_features.ipynb uses."""
    return path.name.removesuffix(".parquet").split("-")[1]


def load_expected_peptides() -> dict[str, set[str]]:
    """Pool -> set of expected (I/L-collapsed) sequences, from the ProteomeTools synthesis
    metadata -- same ground truth task08_auc_new_features.ipynb scores AUC against."""
    exp = pd.read_csv(EXPECTED_CSV)
    return exp.assign(Sequence=_il(exp["Sequence"])).groupby("FolderParent")["Sequence"].agg(set).to_dict()


def compute_training_cutoff() -> float:
    """Recompute the exact percolator_score_database cutoff that decided which PSMs went
    into the deployed SUOD model's training set -- grovems/iforest/iforest.py's
    select_training_candidates + build_training_set, training_source="percolator_percentile"
    (the setting ../iforest_config.yaml uses): candidates = shared, Label_database == 1,
    percolator_score_database > 0; cutoff = the 70th percentile of *that* population;
    train_set = candidates > cutoff. The cutoff is computed globally in production (not per
    file), so this does a full-corpus scan too -- only 3 light columns, but all ~269 files.
    """
    files = sorted(RESULTS_DIR.glob("*.parquet"))
    scores = []
    for f in files:
        df = pd.read_parquet(f, columns=["_merge", "Label_database", "percolator_score_database"])
        mask = (df["_merge"] == "shared") & (df["Label_database"] == 1) & (df["percolator_score_database"] > 0)
        scores.append(df.loc[mask, "percolator_score_database"].to_numpy())
    return float(np.percentile(np.concatenate(scores), 70))


def load_psm_sample(
    n_files: int = 20,
    n_rows: int = 20_000,
    seed: int = 0,
    training_cutoff: float | None = None,
) -> pd.DataFrame:
    """Sample PSMs scored by both sides of the grove_forest_new_features SUOD model.

    Each row is one (SpecId, side) observation -- a shared PSM contributes both a
    "database" row and a "denovo" row, using that side's own feature values and
    ISO_scores, exactly as train_suod.py scores each side independently.

    Also attaches `expected`: whether *that side's own* peptide call is in its raw file's
    pool of synthesized ProteomeTools sequences -- the same ground truth
    task08_auc_new_features.ipynb scores AUC against, here just carried per-PSM for coloring.

    If `training_cutoff` is given (from `compute_training_cutoff()`), also attaches
    `in_training_data`: whether this exact row's feature values were part of the ~2.1M rows
    the deployed forest was actually fit on. Only ever True on `side == "database"` rows --
    the model was trained on database-side feature values only (see `compute_training_cutoff`
    docstring), then applied to both sides at inference, so no `denovo`-side row was ever
    "seen" during training even when its `SpecId` does qualify on the database side.
    """
    rng = np.random.default_rng(seed)
    expected_peptides = load_expected_peptides()

    files = sorted(RESULTS_DIR.glob("*.parquet"))
    chosen_files = rng.choice(files, size=min(n_files, len(files)), replace=False)

    side_frames = []
    for side in ("database", "denovo"):
        cols = [
            "_merge",
            "SpecId",
            "chimeric",
            "sequence_match",
            f"Label_{side}",
            f"ISO_scores_{side}",
            f"SEQUENCE_{side}",
            "percolator_score_database",
        ] + [f"{feat}_{side}" for feat in FEATURES]
        if side == "denovo":
            cols.append("SCORE_denovo")  # Casanovo's own score -- only meaningful on this side
        only_label = f"{side}_only"

        frames = []
        for f in chosen_files:
            df = pd.read_parquet(f, columns=cols)
            df = df[df["_merge"].isin(["shared", only_label])]
            pool = _pool_of(f)
            df["expected"] = _il(df[f"SEQUENCE_{side}"]).isin(expected_peptides.get(pool, set()))
            frames.append(df)

        side_df = pd.concat(frames, ignore_index=True)
        side_df = side_df.dropna(subset=[f"{feat}_{side}" for feat in FEATURES])
        side_df = side_df.rename(
            columns={f"{feat}_{side}": feat for feat in FEATURES}
            | {f"Label_{side}": "Label", f"ISO_scores_{side}": "ISO_scores"}
        )
        if side == "denovo":
            side_df = side_df.rename(columns={"SCORE_denovo": "casanovo_score"})
        if training_cutoff is not None:
            side_df["in_training_data"] = (
                (side == "database")
                & (side_df["_merge"] == "shared")
                & (side_df["Label"] == 1)
                & (side_df["percolator_score_database"] > training_cutoff)
            )
        side_df = side_df.drop(columns=[f"SEQUENCE_{side}", "percolator_score_database"])
        side_df["side"] = side
        side_frames.append(side_df)

    combined = pd.concat(side_frames, ignore_index=True)

    # `_merge == "shared"` only means the scan was identified by *both* searches -- it does
    # NOT mean they called the same peptide (see combine_results.py: sequence_match =
    # modified_sequence_match & precursor_charge_match). For a shared scan where the two
    # searches disagree, the "database" row and "denovo" row below describe two different
    # peptides, not two scoring views of one identification. id_status makes that explicit.
    combined["id_status"] = np.select(
        [
            (combined["_merge"] == "shared") & combined["sequence_match"],
            (combined["_merge"] == "shared") & ~combined["sequence_match"],
        ],
        ["shared, same peptide", "shared, different peptide"],
        default=combined["_merge"].astype(str),
    )

    n_rows = min(n_rows, len(combined))
    sample_idx = rng.choice(len(combined), size=n_rows, replace=False)
    sample = combined.iloc[sample_idx].reset_index(drop=True)
    return sample


def trusted_mask(df: pd.DataFrame) -> np.ndarray:
    """Unsupervised "normal" population: the two searches agree on the peptide, and it's a
    target hit, not a decoy. No expected-peptide ground truth involved, so this generalizes
    to real (non-ProteomeTools) data -- unlike the SUOD training population (percolator
    percentile cut on the database side only), this is defined per side so it's usable for
    `denovo_only` corrections too.
    """
    return (df["id_status"] == "shared, same peptide").to_numpy() & (df["Label"].to_numpy() == 1)


def rank_blend(*score_arrays: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    """Average percentile rank across one or more score arrays (higher = better in each).
    Rank-based so arrays on different scales combine without manual normalization, and the
    blend can't be gamed by one score's arbitrary magnitude.
    """
    from scipy.stats import rankdata

    weights = weights or [1.0] * len(score_arrays)
    n = len(score_arrays[0])
    blended = np.zeros(n)
    for arr, w in zip(score_arrays, weights):
        blended += w * (rankdata(arr) - 1) / (n - 1)
    return blended / sum(weights)


def knn_trust_distance(X_query: np.ndarray, X_trusted: np.ndarray, k: int = 15) -> np.ndarray:
    """Mean distance from each query point to its k nearest neighbors in the trusted
    reference population (standardized feature space). Large distance = far from anything
    the model was told is a confident, corroborated identification.
    """
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(X_trusted))
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X_trusted)
    dist, _ = nn.kneighbors(X_query)
    return dist.mean(axis=1)


def rank_disagreement(iso_scores: np.ndarray, density_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """iso_rank, density_rank, disagreement (= density_rank - iso_rank) -- both inputs
    percentile-ranked first so "large positive disagreement" means the same thing across the
    whole range, regardless of either score's native scale. Split out from
    `disagreement_correction` so a threshold can be picked by actually looking at this
    distribution first (e.g. its 95th percentile), instead of guessing a fixed cutoff.
    """
    iso_rank = rank_blend(iso_scores)
    density_rank = rank_blend(density_score)
    return iso_rank, density_rank, density_rank - iso_rank


def disagreement_correction(
    iso_scores: np.ndarray,
    density_score: np.ndarray,
    threshold: float,
    strength: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric, disagreement-triggered correction -- deliberately *not* a symmetric blend.

    A plain symmetric blend (average `iso_scores` with `density_score` everywhere) has a
    real failure mode here: ISO_scores itself correlates with local position (see the raw
    feature embedding's "colored by ISO_scores" panel -- it's a smooth gradient, not noise),
    so a low-scoring PSM's neighbors tend to *also* score low. A plain kNN/density signal
    picks up on that regional correlation and comes out low too, even for a PSM that's
    genuinely mis-scored relative to its neighbors -- so blending it in everywhere barely
    moves anything, and, worse, still nudges down the "both low, but that's real"
    majority (denovo_only / shared-different-peptide) that shouldn't be touched at all.
    (Measured: iso_rank and density_rank correlate at ~0.95 on this data -- exactly this
    effect, and why a fixed disagreement threshold has to be picked relative to the observed
    distribution, not guessed -- see `rank_disagreement`.)

    This flags a PSM for correction only where the two signals actively *disagree* -- local
    density says trustworthy, but the isolation score says anomalous -- and leaves every
    other PSM's score untouched, including ones where both signals agree the PSM is bad.
    Only ever pushes the score up, never down.

    Returns (corrected, disagreement, flagged). corrected == iso_rank wherever
    flagged is False.
    """
    iso_rank, _, disagreement = rank_disagreement(iso_scores, density_score)
    flagged = disagreement > threshold

    corrected = iso_rank.copy()
    corrected[flagged] = np.clip(iso_rank[flagged] + strength * disagreement[flagged], 0, 1)
    return corrected, disagreement, flagged


def local_disagreement_correction(
    if_scores: np.ndarray, umap_emb: np.ndarray, k: int = 15, w: float = 0.5, z_threshold: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local z-score correction, computed directly in embedding space: for each point, find
    its k nearest neighbors in `umap_emb`, then compare its own ISO score against those
    neighbors' own ISO scores (mean/std) -- no separate "trusted reference" population
    needed, unlike `disagreement_correction`/`rank_disagreement` above.

    Corrects in **both** directions, always toward the local mean: a point whose score is a
    low outlier relative to its immediate neighborhood (z < -z_threshold) gets pushed *up*;
    a point whose score is a high outlier (z > z_threshold -- a spuriously high score sitting
    among otherwise low-scoring neighbors) gets pulled *down* just as readily. Untouched
    wherever `|z| <= z_threshold`, scaled by `w` otherwise.

    Returns (corrected, local_mean, z).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(umap_emb)
    _, idx = nn.kneighbors(umap_emb)
    neighbor_idx = idx[:, 1:]  # drop self

    local_mean = if_scores[neighbor_idx].mean(axis=1)
    local_std = if_scores[neighbor_idx].std(axis=1)

    # z-score: how far is this point's score from its local neighborhood
    z = (if_scores - local_mean) / (local_std + 1e-6)

    # correct whenever the point disagrees strongly with its neighbors, in either direction --
    # w * (local_mean - if_scores) is positive (pushes up) when if_scores < local_mean, and
    # negative (pulls down) when if_scores > local_mean, so one expression covers both.
    correction = np.where(np.abs(z) > z_threshold, w * (local_mean - if_scores), 0)

    return np.clip(if_scores + correction, 0, 1), local_mean, z


def local_regression_correction(
    if_scores: np.ndarray, umap_emb: np.ndarray, k: int = 30, bandwidth: float | None = None, w: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-weighted local regression, in the same spirit as `local_disagreement_correction`
    but smoother and threshold-free. Instead of a hard k-NN mean/std, each of the k nearest
    neighbors gets a Gaussian weight by distance (`bandwidth`, default the median neighbor
    distance) -- closer neighbors count more -- giving a Nadaraya-Watson-style local
    expectation `local_pred` for what this PSM's score "should" look like given its
    neighborhood.

    Bidirectional, like `local_disagreement_correction`, but continuous rather than
    threshold-triggered: every point moves toward `local_pred` by a fraction `w` of the gap,
    in whichever direction that gap points -- up when `local_pred > if_scores`, down when
    `local_pred < if_scores` -- with no on/off boundary (bigger gap -> bigger nudge, tiny gap
    -> tiny nudge, nothing is ever fully "untouched" except where the gap is exactly zero).

    Returns (corrected, local_pred).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(umap_emb)
    dist, idx = nn.kneighbors(umap_emb)
    dist, idx = dist[:, 1:], idx[:, 1:]  # drop self

    bandwidth = bandwidth or np.median(dist)
    weights = np.exp(-(dist**2) / (2 * bandwidth**2))
    weights /= weights.sum(axis=1, keepdims=True)

    local_pred = (weights * if_scores[idx]).sum(axis=1)  # kernel-weighted local expectation

    gap = local_pred - if_scores
    correction = w * gap  # bidirectional: pulls toward local_pred either way

    return np.clip(if_scores + correction, 0, 1), local_pred


def linear_umap_correction(
    if_scores: np.ndarray, umap_emb: np.ndarray, z_threshold: float = 1.0, w: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The simplest version of this idea: one global linear regression of ISO_scores on the
    two UMAP coordinates (`linear_pred = b0 + b1*UMAP1 + b2*UMAP2`), instead of a per-point
    local neighborhood (`local_disagreement_correction`/`local_regression_correction` above).
    A point's residual (`if_scores - linear_pred`) is z-scored against the residual
    population as a whole; only points more than `z_threshold` residual-std away from the
    fitted plane get corrected, pulled toward `linear_pred` by a fraction `w` of the
    residual, in whichever direction -- exactly mirroring `local_disagreement_correction`'s
    rule, just with one global fit standing in for a per-point local neighborhood.

    Much coarser than the local methods: since ISO_scores is not actually linear in UMAP1/2
    (see the "colored by ISO_scores" panel -- it's a curved gradient, not a plane), a lot of
    the residual here reflects that curvature, not genuine local anomalies. Check the
    returned r_squared -- a low value means "far from the fit" is picking up curvature more
    than outliers.

    Returns (corrected, linear_pred, z, r_squared).
    """
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(umap_emb, if_scores)
    linear_pred = reg.predict(umap_emb)
    r_squared = reg.score(umap_emb, if_scores)

    residual = if_scores - linear_pred
    z = residual / (residual.std() + 1e-6)

    correction = np.where(np.abs(z) > z_threshold, w * (linear_pred - if_scores), 0)

    return np.clip(if_scores + correction, 0, 1), linear_pred, z, r_squared


def plot_local_correction_diagnostic(
    if_scores: np.ndarray,
    local_mean: np.ndarray,
    z: np.ndarray,
    z_threshold: float,
    suptitle: str,
    pred_label: str = "local_mean (k nearest neighbors' own ISO_scores, UMAP space)",
    z_label: str = "z  ((ISO_scores - local_mean) / local_std)",
):
    """Same two-panel style as plot_disagreement_diagnostic, for any z-score-threshold-based
    correction (local_disagreement_correction, linear_umap_correction, ...): (1) each point's
    own ISO score vs its "prediction" (whatever `local_mean` actually is -- a local
    neighborhood mean, a global linear fit, etc.), flagged points highlighted and colored by
    correction direction -- "corrected up" points should sit above the diagonal (self low,
    prediction high), "corrected down" points below it; (2) the z distribution with both
    thresholds marked. `pred_label`/`z_label` customize the axis text for whichever
    "prediction" this actually is.
    """
    import matplotlib.pyplot as plt

    flagged_up = z < -z_threshold
    flagged_down = z > z_threshold
    flagged = flagged_up | flagged_down
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.scatter(if_scores[~flagged], local_mean[~flagged], s=4, alpha=0.3, color="#888", label="not flagged")
    ax1.scatter(
        if_scores[flagged_up], local_mean[flagged_up], s=6, alpha=0.7, color="#d62728", label="flagged (corrected up)"
    )
    ax1.scatter(
        if_scores[flagged_down], local_mean[flagged_down], s=6, alpha=0.7, color="#1f77b4",
        label="flagged (corrected down)",
    )
    ax1.plot([0, 1], [0, 1], color="black", lw=1, ls="--", zorder=0)
    ax1.set_xlabel("ISO_scores (own value)")
    ax1.set_ylabel(pred_label)
    ax1.set_title("Where flagged points sit", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")

    ax2.hist(z, bins=100, color="#4c72b0")
    ax2.axvline(-z_threshold, color="#d62728", lw=1.5, ls="--", label=f"z threshold = ±{z_threshold}")
    ax2.axvline(z_threshold, color="#d62728", lw=1.5, ls="--")
    ax2.set_xlabel(z_label)
    ax2.set_ylabel("count")
    ax2.set_title(
        f"{flagged.sum():,} / {len(z):,} flagged ({100 * flagged.mean():.1f}%): "
        f"{flagged_up.sum():,} up, {flagged_down.sum():,} down",
        fontsize=10,
    )
    ax2.legend(fontsize=8)

    fig.suptitle(suptitle, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_regression_correction_diagnostic(if_scores: np.ndarray, local_pred: np.ndarray, suptitle: str):
    """Two-panel diagnostic for local_regression_correction: (1) each point's own ISO score vs
    its kernel-weighted local prediction, colored by correction direction -- above the
    diagonal (local_pred > if_scores) gets pulled up, below it gets pulled down; (2) the gap
    distribution -- no hard threshold here, so direction is a continuum, not a flagged/not
    split (every nonzero gap gets *some* correction, scaled by its own size).
    """
    import matplotlib.pyplot as plt

    gap = local_pred - if_scores
    up = gap > 0
    down = gap < 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.scatter(if_scores[up], local_pred[up], s=5, alpha=0.4, color="#d62728", label="corrected up (local_pred > ISO_scores)")
    ax1.scatter(if_scores[down], local_pred[down], s=5, alpha=0.4, color="#1f77b4", label="corrected down (local_pred < ISO_scores)")
    ax1.plot([0, 1], [0, 1], color="black", lw=1, ls="--", zorder=0)
    ax1.set_xlabel("ISO_scores (own value)")
    ax1.set_ylabel("local_pred (kernel-weighted local expectation, UMAP space)")
    ax1.set_title("Correction direction", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")

    ax2.hist(gap, bins=100, color="#4c72b0")
    ax2.axvline(0, color="black", lw=1.5, ls="--", label="gap = 0 (no correction)")
    ax2.set_xlabel("gap  (local_pred - ISO_scores)")
    ax2.set_ylabel("count")
    ax2.set_title(
        f"{up.sum():,} up ({100 * up.mean():.1f}%), {down.sum():,} down ({100 * down.mean():.1f}%)", fontsize=10
    )
    ax2.legend(fontsize=8)

    fig.suptitle(suptitle, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def tree_leaves(base_detectors, X: np.ndarray) -> np.ndarray:
    """(n_samples, n_trees) leaf index per tree, across every base IForest detector's every
    tree -- same extraction as isoforest_leaf_embedding_pca_umap.ipynb.
    """
    leaf_columns = []
    for base in base_detectors:
        iso = base.detector_
        for tree, feat_idx in zip(iso.estimators_, iso.estimators_features_):
            leaf_columns.append(tree.apply(X[:, feat_idx]))
    return np.column_stack(leaf_columns)


def leaf_trust_similarity_streaming(
    base_detectors, X_query: np.ndarray, X_trusted: np.ndarray, chunk_size: int = 500_000
) -> np.ndarray:
    """Same result as leaf_trust_similarity(tree_leaves(base_detectors, X_query), ...), but
    never materializes the full (n_samples, n_trees) leaf matrix -- at full task08 scale
    (~11.8M scans x 800 trees) that would be ~38GB of int32. Instead accumulates the running
    similarity sum one tree at a time, and processes X_query in chunks so at most one
    (chunk_size,) leaf-index array is live per tree.
    """
    n = X_query.shape[0]
    similarity = np.zeros(n)
    n_trusted = X_trusted.shape[0]
    n_trees = 0

    for base in base_detectors:
        iso = base.detector_
        for tree, feat_idx in zip(iso.estimators_, iso.estimators_features_):
            n_trees += 1
            trusted_leaves = tree.apply(X_trusted[:, feat_idx])
            freq = np.bincount(trusted_leaves) / n_trusted
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                leaf_ids = tree.apply(X_query[start:end, feat_idx])
                in_range = leaf_ids < len(freq)
                chunk_sim = np.zeros(end - start)
                chunk_sim[in_range] = freq[leaf_ids[in_range]]
                similarity[start:end] += chunk_sim

    return similarity / n_trees


def leaf_trust_similarity(query_leaves: np.ndarray, trusted_leaves: np.ndarray) -> np.ndarray:
    """Isolation-kernel similarity to the trusted population: for each tree, what fraction of
    the trusted population landed in the same leaf as this query point; averaged over trees.
    High = this PSM keeps ending up in leaves the forest also uses for confident, corroborated
    identifications -- the direct tree-structure analog of `knn_trust_distance`.
    """
    n_trees = query_leaves.shape[1]
    n_trusted = trusted_leaves.shape[0]
    similarity = np.zeros(query_leaves.shape[0])
    for t in range(n_trees):
        freq = np.bincount(trusted_leaves[:, t]) / n_trusted
        leaf_ids = query_leaves[:, t]
        in_range = leaf_ids < len(freq)
        similarity[in_range] += freq[leaf_ids[in_range]]
    return similarity / n_trees


def color_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Convenience dict of columns worth coloring scatter plots by."""
    cols = {
        "ISO_scores": df["ISO_scores"],
        "id_status": df["id_status"],
        "side": df["side"],
        "Label": df["Label"],
    }
    if "expected" in df.columns:
        cols["expected"] = df["expected"]
    if "in_training_data" in df.columns:
        cols["in_training_data"] = df["in_training_data"]
    return cols


# For a categorical coloring, which value (if present) should be drawn last so it sits on
# top instead of being buried under a more numerous category.
Z_ORDER_ON_TOP = {"Label": -1, "expected": True, "in_training_data": True}


def _plot_colored_grid(
    coords_by_method: dict[str, np.ndarray],
    colorings: dict[str, pd.Series],
    suptitle: str,
    continuous_names: set[str] = frozenset(),
):
    """Shared scatter-grid core: one row per entry in `colorings`, one column per entry in
    `coords_by_method`. A row renders as a continuous viridis scatter if its name is in
    `continuous_names`, otherwise as a categorical tab10 scatter (one series per category,
    z-ordered via Z_ORDER_ON_TOP so the more interesting value isn't buried).
    """
    import matplotlib.pyplot as plt

    methods = list(coords_by_method)
    fig, axes = plt.subplots(
        len(colorings), len(methods), figsize=(5.5 * len(methods), 4.2 * len(colorings)), squeeze=False
    )

    for row, (color_name, values) in enumerate(colorings.items()):
        is_categorical = color_name not in continuous_names
        if is_categorical:
            categories = values.astype("category")
            codes = categories.cat.codes
            cmap = plt.get_cmap("tab10")
            # draw order = z-order (later = on top); color stays tied to the category's
            # natural code so it's stable regardless of draw order.
            cats = list(categories.cat.categories)
            draw_order = cats
            top = Z_ORDER_ON_TOP.get(color_name)
            if top in cats:
                draw_order = [c for c in cats if c != top] + [top]
            code_of = {label: code for code, label in enumerate(cats)}
        for col, method in enumerate(methods):
            ax = axes[row, col]
            xy = coords_by_method[method]
            if is_categorical:
                for label in draw_order:
                    code = code_of[label]
                    mask = codes == code
                    ax.scatter(xy[mask, 0], xy[mask, 1], s=4, alpha=0.5, color=cmap(code % 10), label=str(label))
                if col == len(methods) - 1:
                    ax.legend(fontsize=7, markerscale=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
            else:
                sc = ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.6, c=values, cmap="viridis")
                if col == len(methods) - 1:
                    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{method} -- colored by {color_name}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(suptitle, y=1.0, fontsize=13)
    fig.tight_layout()
    return fig


def plot_embedding_grid(coords_by_method: dict[str, np.ndarray], df: pd.DataFrame, suptitle: str):
    """coords_by_method: {"PCA": (n,2) array, "UMAP": (n,2) array}. One row per status column
    from `color_columns` (ISO_scores continuous, the rest categorical), one column per method.
    """
    return _plot_colored_grid(coords_by_method, color_columns(df), suptitle, continuous_names={"ISO_scores"})


def plot_feature_grid(coords_by_method: dict[str, np.ndarray], feature_values: dict[str, np.ndarray], suptitle: str):
    """Same layout as plot_embedding_grid, but one row per entry in `feature_values` instead
    of per status column -- every row continuous (viridis), showing that feature's own raw
    value (not standardized) across the embedding. Lets you read PCA/UMAP structure back
    against each input column directly, complementing plot_pca_loadings' PC1/PC2-only view.
    Pass e.g. `{f: df[f] for f in FEATURES}` for the raw columns, or
    `{name: X_engineered[:, i] for i, name in enumerate(ENGINEERED_FEATURE_NAMES)}` for
    engineered/synthetic columns that aren't in `df`.
    """
    return _plot_colored_grid(coords_by_method, feature_values, suptitle, continuous_names=set(feature_values))


def plot_score_correction(df: pd.DataFrame, raw_col: str, corrected_col: str, suptitle: str):
    """Raw vs corrected score distribution, one panel per `side`, lines split by `id_status`
    -- same style as distribution.ipynb's ISO_scores histogram, so before/after is a direct
    visual comparison against the existing pipeline plots.
    """
    import matplotlib.pyplot as plt

    sides = list(df["side"].unique())
    fig, axes = plt.subplots(len(sides), 2, figsize=(11, 4.2 * len(sides)), squeeze=False)

    for row, side in enumerate(sides):
        side_df = df[df["side"] == side]
        for col, (score_col, label) in enumerate([(raw_col, "raw ISO_scores"), (corrected_col, "corrected")]):
            ax = axes[row, col]
            for status, group in side_df.groupby("id_status", observed=True):
                ax.hist(
                    group[score_col].dropna(), bins=80, range=(0, 1), density=True,
                    histtype="step", lw=1.6, label=f"{status} (n={len(group):,})",
                )
            ax.set_xlim(0, 1)
            ax.set_xlabel(f"{label} (0=bad, 1=good)")
            ax.set_ylabel("density")
            ax.set_title(f"{side} -- {label}", fontsize=10)
            if row == 0 and col == 1:
                ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.suptitle(suptitle, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_disagreement_diagnostic(
    iso_rank: np.ndarray, density_rank: np.ndarray, disagreement: np.ndarray, threshold: float, suptitle: str
):
    """Two panels: (1) density_rank vs iso_rank, flagged points (disagreement > threshold)
    highlighted -- shows the flagged region sits above the diagonal (density confident, IF
    not), not just scattered anywhere with a low IF score; (2) the disagreement distribution
    itself with the threshold marked, so the cutoff's severity is visible, not just a number.
    """
    import matplotlib.pyplot as plt

    flagged = disagreement > threshold
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.scatter(iso_rank[~flagged], density_rank[~flagged], s=4, alpha=0.3, color="#888", label="not flagged")
    ax1.scatter(iso_rank[flagged], density_rank[flagged], s=6, alpha=0.7, color="#d62728", label="flagged (corrected)")
    ax1.plot([0, 1], [0, 1], color="black", lw=1, ls="--", zorder=0)
    ax1.set_xlabel("iso_rank (isolation score, percentile)")
    ax1.set_ylabel("density_rank (trust-reference closeness, percentile)")
    ax1.set_title("Where flagged points sit", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")

    ax2.hist(disagreement, bins=100, color="#4c72b0")
    ax2.axvline(threshold, color="#d62728", lw=1.5, ls="--", label=f"threshold = {threshold:.3f}")
    ax2.set_xlabel("disagreement  (density_rank - iso_rank)")
    ax2.set_ylabel("count")
    ax2.set_title(f"{flagged.sum():,} / {len(disagreement):,} flagged ({100 * flagged.mean():.1f}%)", fontsize=10)
    ax2.legend(fontsize=8)

    fig.suptitle(suptitle, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_pca_loadings(pca, feature_names: list[str], title: str):
    """Bar chart of each input feature's weight on PC1 and PC2 -- makes explicit which of
    the named columns actually drive the two PCA axes plotted above (PCA itself doesn't
    label its axes by feature, so this is the only way to see it).
    """
    import matplotlib.pyplot as plt

    loadings = pca.components_[:2]  # (2, n_features)
    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(feature_names)), 3.5))
    ax.bar(x - width / 2, loadings[0], width, label="PC1")
    ax.bar(x + width / 2, loadings[1], width, label="PC2")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("loading")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
