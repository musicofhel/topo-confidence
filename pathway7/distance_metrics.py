"""Non-Euclidean persistent homology backends for topo-confidence.

Three distance-metric alternatives to the Euclidean Vietoris-Rips PH used in
the original winning_features.py pipeline. All return the same
``dict[int, np.ndarray]`` diagram format as ``compute_ph()`` so they can be
fed directly into ``features_from_diagrams()``.

References
----------
- Effective resistance: Damrich et al. NeurIPS 2024 (arXiv:2311.03087)
- Cosine PH: standard, see also Fay et al. 2505.20435 "Holes in Latent Space"
- DTM: GUDHI weighted-Rips, Anai et al. 2020
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph
from ripser import ripser


# ---------------------------------------------------------------------------
# Cosine-distance PH
# ---------------------------------------------------------------------------

def cosine_ph(
    points: np.ndarray,
    maxdim: int = 1,
    thresh: float | None = None,
) -> dict[int, np.ndarray]:
    """Persistent homology using cosine distance.

    Cosine distance captures directional structure that Euclidean PH misses
    when hidden-state norms vary widely across tokens.

    Parameters
    ----------
    points : (n, d) array
    maxdim : max homology dimension (0 = H0 only, 1 = H0+H1)
    thresh : optional filtration threshold

    Returns
    -------
    dict mapping dimension -> (n_features, 2) birth-death array
    """
    if len(points) < 3:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}

    D = squareform(pdist(points, metric="cosine"))
    # Handle NaN from zero-norm vectors
    np.nan_to_num(D, copy=False, nan=0.0)

    kwargs: dict = {"maxdim": maxdim, "distance_matrix": True}
    if thresh is not None:
        kwargs["thresh"] = thresh

    result = ripser(D, **kwargs)
    return {d: result["dgms"][d] for d in range(maxdim + 1)}


# ---------------------------------------------------------------------------
# Effective-resistance PH (Damrich et al. NeurIPS 2024)
# ---------------------------------------------------------------------------

def effective_resistance_ph(
    points: np.ndarray,
    k: int = 30,
    n_eigs: int | None = None,
    maxdim: int = 1,
    thresh: float | None = None,
) -> dict[int, np.ndarray]:
    """Persistent homology via effective-resistance embedding on a kNN graph.

    Addresses Euclidean distance concentration in high dimensions by computing
    the normalized-Laplacian spectral embedding (Proposition 6.1 of Damrich
    et al.) and running Vietoris-Rips on the embedding distances.

    Parameters
    ----------
    points : (n, d) array  — raw hidden states (no PCA needed)
    k : kNN graph connectivity (clamped to n-1 if n is small)
    n_eigs : number of Laplacian eigenvectors to keep (default: min(50, n-2))
    maxdim : max homology dimension
    thresh : optional filtration threshold

    Returns
    -------
    dict mapping dimension -> (n_features, 2) birth-death array
    """
    n = len(points)
    if n < 3:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}

    k_eff = min(k, n - 1)
    if n_eigs is None:
        n_eigs = min(50, n - 2)
    else:
        n_eigs = min(n_eigs, n - 2)

    if n_eigs < 1:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}

    # (a) Symmetric kNN graph (connectivity mode = binary adjacency)
    A = kneighbors_graph(points, n_neighbors=k_eff, mode="connectivity", include_self=False)
    A = A.maximum(A.T)  # symmetrize

    # (b) Normalized Laplacian + degree vector
    L_sym, degrees = csgraph.laplacian(A, normed=True, return_diag=True)

    # (c) Smallest eigenvectors (SM = smallest magnitude)
    # Request n_eigs+1 to include the trivial zero eigenvector
    n_request = min(n_eigs + 1, n - 1)
    try:
        mu, U = eigsh(L_sym, k=n_request, which="SM")
    except Exception:
        # eigsh can fail on very small or disconnected graphs
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}

    # Drop the trivial (near-zero) eigenvector
    nontrivial = mu > 1e-8
    mu = mu[nontrivial]
    U = U[:, nontrivial]

    if len(mu) == 0:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}

    # (d) Effective-resistance embedding (Proposition 6.1)
    #     e_i = ((1 - mu_k) / sqrt(mu_k)) * u_{k,i}  /  sqrt(d_i)
    scale = (1.0 - mu) / np.sqrt(mu)  # (n_eigs,)
    deg_sqrt_inv = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))  # (n,)
    E = (U * scale[None, :]) * deg_sqrt_inv[:, None]  # (n, n_eigs)

    # (e) Euclidean distance on embedding → ripser
    D = squareform(pdist(E, metric="euclidean"))

    kwargs: dict = {"maxdim": maxdim, "distance_matrix": True}
    if thresh is not None:
        kwargs["thresh"] = thresh

    result = ripser(D, **kwargs)
    return {d: result["dgms"][d] for d in range(maxdim + 1)}


# ---------------------------------------------------------------------------
# Diffusion-distance PH (runner-up in Damrich et al.)
# ---------------------------------------------------------------------------

def diffusion_ph(
    points: np.ndarray,
    k: int = 15,
    t: int = 8,
    n_eigs: int | None = None,
    maxdim: int = 1,
    thresh: float | None = None,
) -> dict[int, np.ndarray]:
    """Persistent homology via diffusion-distance embedding.

    Damrich et al.'s second-ranked method: same Laplacian eigenvectors but
    with diffusion kernel scaling (1-mu)^t instead of (1-mu)/sqrt(mu).

    Parameters
    ----------
    points : (n, d) array
    k : kNN graph connectivity
    t : diffusion time
    n_eigs : eigenvectors to keep
    maxdim : max homology dimension
    thresh : filtration threshold

    Returns
    -------
    dict mapping dimension -> (n_features, 2) birth-death array
    """
    n = len(points)
    if n < 3:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}

    k_eff = min(k, n - 1)
    if n_eigs is None:
        n_eigs = min(50, n - 2)
    else:
        n_eigs = min(n_eigs, n - 2)

    if n_eigs < 1:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}

    A = kneighbors_graph(points, n_neighbors=k_eff, mode="connectivity", include_self=False)
    A = A.maximum(A.T)

    L_sym, degrees = csgraph.laplacian(A, normed=True, return_diag=True)

    n_request = min(n_eigs + 1, n - 1)
    try:
        mu, U = eigsh(L_sym, k=n_request, which="SM")
    except Exception:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}

    nontrivial = mu > 1e-8
    mu = mu[nontrivial]
    U = U[:, nontrivial]

    if len(mu) == 0:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}

    # Diffusion scaling: (1-mu)^t
    scale = np.power(1.0 - mu, t)
    deg_sqrt_inv = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    E = (U * scale[None, :]) * deg_sqrt_inv[:, None]

    D = squareform(pdist(E, metric="euclidean"))

    kwargs: dict = {"maxdim": maxdim, "distance_matrix": True}
    if thresh is not None:
        kwargs["thresh"] = thresh

    result = ripser(D, **kwargs)
    return {d: result["dgms"][d] for d in range(maxdim + 1)}


# ---------------------------------------------------------------------------
# DTM-Rips PH (GUDHI, optional)
# ---------------------------------------------------------------------------

def dtm_ph(
    points: np.ndarray,
    k: int = 15,
    maxdim: int = 1,
    max_filtration: float = float("inf"),
) -> dict[int, np.ndarray]:
    """Persistent homology with Distance-to-Measure filtration (outlier-robust).

    Uses GUDHI's DTMRipsComplex. Falls back to Euclidean Rips if GUDHI is not
    installed (GUDHI is optional for the core pipeline).

    Parameters
    ----------
    points : (n, d) array
    k : DTM neighborhood size
    maxdim : max homology dimension
    max_filtration : filtration cutoff

    Returns
    -------
    dict mapping dimension -> (n_features, 2) birth-death array
    """
    if len(points) < 3:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}

    try:
        from gudhi.dtm_rips_complex import DTMRipsComplex
    except ImportError:
        # Fallback: standard Euclidean Rips via ripser
        result = ripser(points, maxdim=maxdim)
        return {d: result["dgms"][d] for d in range(maxdim + 1)}

    k_eff = min(k, len(points) - 1)
    dtm_rips = DTMRipsComplex(points=points, k=k_eff, q=2, max_filtration=max_filtration)
    st = dtm_rips.create_simplex_tree(max_dimension=maxdim + 1)
    st.compute_persistence()

    diagrams: dict[int, np.ndarray] = {}
    for d in range(maxdim + 1):
        pairs = st.persistence_intervals_in_dimension(d)
        if len(pairs) > 0:
            diagrams[d] = np.array(pairs)
        else:
            diagrams[d] = np.empty((0, 2))
    return diagrams


# ---------------------------------------------------------------------------
# Standard Euclidean PH (for benchmark comparisons)
# ---------------------------------------------------------------------------

def euclidean_ph(
    points: np.ndarray,
    maxdim: int = 1,
) -> dict[int, np.ndarray]:
    """Standard Euclidean Vietoris-Rips persistent homology.

    This is the same computation as winning_features.py:compute_ph() but
    returns the same dict interface as the non-Euclidean backends.
    """
    if len(points) < 3:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}
    result = ripser(points, maxdim=maxdim)
    return {d: result["dgms"][d] for d in range(maxdim + 1)}


# ---------------------------------------------------------------------------
# Feature extraction from persistence diagrams
# ---------------------------------------------------------------------------

def persistence_entropy(lifetimes: np.ndarray) -> float:
    """Shannon entropy of normalized persistence lifetimes."""
    if len(lifetimes) == 0 or lifetimes.sum() <= 0:
        return 0.0
    p = lifetimes / lifetimes.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def features_from_diagrams(
    diagrams: dict[int, np.ndarray],
    prefix: str = "",
) -> tuple[np.ndarray, list[str]]:
    """Extract scalar features from persistence diagrams.

    Returns 8 features per diagram set:
    - H0: total_persistence, max_lifetime, entropy, n_features
    - H1: total_persistence, max_lifetime, entropy, n_features

    Parameters
    ----------
    diagrams : output of any *_ph() function
    prefix : name prefix for feature names (e.g. "effres_", "cosine_")

    Returns
    -------
    (8,) feature array and list of 8 feature names
    """
    feats = np.zeros(8, dtype=np.float64)
    names = []

    for dim_idx, dim in enumerate([0, 1]):
        dgm = diagrams.get(dim, np.empty((0, 2)))
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.empty((0, 2))
        life = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])

        base = dim_idx * 4
        feats[base + 0] = life.sum() if len(life) > 0 else 0.0
        feats[base + 1] = life.max() if len(life) > 0 else 0.0
        feats[base + 2] = persistence_entropy(life)
        feats[base + 3] = float(len(life))

        dim_name = f"H{dim}"
        names.extend([
            f"{prefix}{dim_name}_total_persistence",
            f"{prefix}{dim_name}_max_lifetime",
            f"{prefix}{dim_name}_entropy",
            f"{prefix}{dim_name}_n_features",
        ])

    return feats, names


# ---------------------------------------------------------------------------
# Graph-level features from the kNN Laplacian
# ---------------------------------------------------------------------------

def graph_spectral_features(
    points: np.ndarray,
    k: int = 30,
) -> tuple[np.ndarray, list[str]]:
    """Spectral gap and algebraic connectivity from the kNN graph Laplacian.

    Parameters
    ----------
    points : (n, d) array
    k : kNN connectivity

    Returns
    -------
    (2,) feature array and list of 2 feature names
    """
    n = len(points)
    feats = np.zeros(2, dtype=np.float64)
    names = ["spectral_gap", "algebraic_connectivity"]

    if n < 4:
        return feats, names

    k_eff = min(k, n - 1)
    A = kneighbors_graph(points, n_neighbors=k_eff, mode="connectivity", include_self=False)
    A = A.maximum(A.T)
    L_sym, _ = csgraph.laplacian(A, normed=True, return_diag=True)

    try:
        vals, _ = eigsh(L_sym, k=min(3, n - 1), which="SM")
        vals = np.sort(vals)
        # Spectral gap: lambda_2 - lambda_1 (lambda_1 ~ 0 for connected graph)
        if len(vals) >= 2:
            feats[0] = float(vals[1] - vals[0])
        # Algebraic connectivity: lambda_2 (Fiedler value)
        if len(vals) >= 2:
            feats[1] = float(vals[1])
    except Exception:
        pass

    return feats, names


# ---------------------------------------------------------------------------
# Convenience: compute all non-Euclidean features for one point cloud
# ---------------------------------------------------------------------------

def compute_all_noneuclid_features(
    points: np.ndarray,
    k_effres: int = 30,
    k_diffusion: int = 15,
    t_diffusion: int = 8,
    maxdim: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Compute all non-Euclidean PH + spectral features for a single point cloud.

    Returns ~18 features:
    - 8 from effective-resistance PH (H0+H1 × 4 stats)
    - 8 from cosine PH
    - 2 from graph spectral properties

    Parameters
    ----------
    points : (n, d) array — subsampled token hidden states
    k_effres : kNN for effective resistance
    k_diffusion : kNN for diffusion (unused here but available)
    t_diffusion : diffusion time (unused here but available)
    maxdim : max homology dimension

    Returns
    -------
    feature array and feature names
    """
    all_feats = []
    all_names = []

    # Effective-resistance PH
    er_dgms = effective_resistance_ph(points, k=k_effres, maxdim=maxdim)
    er_feats, er_names = features_from_diagrams(er_dgms, prefix="effres_")
    all_feats.append(er_feats)
    all_names.extend(er_names)

    # Cosine PH
    cos_dgms = cosine_ph(points, maxdim=maxdim)
    cos_feats, cos_names = features_from_diagrams(cos_dgms, prefix="cosine_")
    all_feats.append(cos_feats)
    all_names.extend(cos_names)

    # Graph spectral features
    spec_feats, spec_names = graph_spectral_features(points, k=k_effres)
    all_feats.append(spec_feats)
    all_names.extend(spec_names)

    return np.concatenate(all_feats), all_names


def compute_all_ph_features(
    points: np.ndarray,
    k_effres: int = 30,
    maxdim: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Compute Euclidean + non-Euclidean PH + spectral features (26 total).

    Returns features in order:
    - [0:8]   Euclidean PH (H0+H1 × 4 stats)
    - [8:16]  Effective-resistance PH
    - [16:24] Cosine PH
    - [24:26] Graph spectral (spectral_gap, algebraic_connectivity)

    Parameters
    ----------
    points : (n, d) array — PCA-reduced, subsampled token hidden states
    k_effres : kNN for effective resistance and spectral features
    maxdim : max homology dimension

    Returns
    -------
    (26,) feature array and list of 26 feature names
    """
    all_feats = []
    all_names = []

    # Euclidean PH (8 features)
    euc_dgms = euclidean_ph(points, maxdim=maxdim)
    euc_feats, euc_names = features_from_diagrams(euc_dgms, prefix="euclid_")
    all_feats.append(euc_feats)
    all_names.extend(euc_names)

    # Non-Euclidean (18 features)
    ne_feats, ne_names = compute_all_noneuclid_features(
        points, k_effres=k_effres, maxdim=maxdim
    )
    all_feats.append(ne_feats)
    all_names.extend(ne_names)

    return np.concatenate(all_feats), all_names
