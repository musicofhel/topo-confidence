"""Zigzag persistence across transformer layers.

Implements the Gardinazzi et al. ICML 2025 recipe:
  1. Per-layer kNN graph (k=4) on subsampled token hidden states
  2. Intersection complex between consecutive layers
  3. Zigzag persistence via Dionysus2
  4. Vectorize: persistent entropy, total H1 persistence, n long-lived cycles

Requires dionysus (source build) and pyfzz. These are only available in the
Pathway 7 Docker image — this module will raise ImportError on systems without
Dionysus installed.

References:
  - Gardinazzi et al. 2410.11042 (ICML 2025)
  - github.com/RitAreaSciencePark/ZigZagLLMs
  - Samaga et al. 2601.01552 "HalluZig" (EACL 2026)
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import kneighbors_graph
from itertools import combinations


def _knn_edges(X: np.ndarray, k: int = 4) -> set[tuple[int, int]]:
    """Build symmetric kNN edge set from point cloud."""
    n = len(X)
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return set()

    A = kneighbors_graph(X, n_neighbors=k_eff, mode="connectivity", include_self=False)
    A = A.maximum(A.T)  # symmetrize

    # Extract edges
    coo = A.tocoo()
    edges = set()
    for i, j in zip(coo.row, coo.col):
        if i < j:
            edges.add((i, j))
    return edges


def _flag_complex(edges: set[tuple[int, int]], vertices: set[int], max_dim: int = 2) -> list[tuple]:
    """Build flag (clique) complex from edge set up to max_dim.

    A flag complex includes all simplices whose 1-skeleton is in the graph.
    """
    simplices = []

    # 0-simplices (vertices)
    for v in sorted(vertices):
        simplices.append((v,))

    # 1-simplices (edges)
    for e in sorted(edges):
        simplices.append(e)

    if max_dim < 2:
        return simplices

    # Build adjacency for clique enumeration
    adj: dict[int, set[int]] = {v: set() for v in vertices}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    # 2-simplices (triangles)
    triangles = set()
    for i, j in edges:
        common = adj[i] & adj[j]
        for k in common:
            tri = tuple(sorted([i, j, k]))
            triangles.add(tri)
    simplices.extend(sorted(triangles))

    if max_dim < 3:
        return simplices

    # 3-simplices (tetrahedra)
    for tri in triangles:
        a, b, c = tri
        common = adj[a] & adj[b] & adj[c]
        for d in common:
            tet = tuple(sorted([a, b, c, d]))
            simplices.append(tet)

    return simplices


def compute_zigzag_diagrams(
    layer_clouds: list[np.ndarray],
    k: int = 4,
    max_simplex_dim: int = 2,
) -> list[np.ndarray]:
    """Compute zigzag persistence across transformer layers.

    Parameters
    ----------
    layer_clouds : list of (n_tokens, hidden_dim) arrays, one per layer
    k : kNN connectivity per layer
    max_simplex_dim : max simplex dimension (tracking homology up to dim-1)

    Returns
    -------
    List of (n_features, 2) birth-death arrays, one per homology dimension.
    Birth/death values are layer indices (even = model layer, odd = intersection).
    """
    try:
        import dionysus as d
    except ImportError:
        raise ImportError(
            "Dionysus2 is required for zigzag persistence. "
            "Install via: pip install dionysus==2.0.10 (requires Boost + C++14). "
            "This is pre-built in the Pathway 7 Docker image."
        )

    n_layers = len(layer_clouds)
    if n_layers < 2:
        return [np.empty((0, 2))]

    n_points = len(layer_clouds[0])
    vertices = set(range(n_points))

    # Build per-layer kNN edge sets
    layer_edges = []
    for X in layer_clouds:
        edges = _knn_edges(X, k=k)
        layer_edges.append(edges)

    # Build zigzag filtration:
    # K_L0 ⊇ K_int(0,1) ⊆ K_L1 ⊇ K_int(1,2) ⊆ K_L2 ⊇ ...
    # Even times = model layer, odd times = intersection

    # Collect all simplices across all layers + intersections
    all_simplices_set: set[tuple] = set()
    simplex_times: dict[tuple, list[float]] = {}

    for layer_idx in range(n_layers):
        # Layer complex
        time_val = 2 * layer_idx  # even
        layer_complex = _flag_complex(layer_edges[layer_idx], vertices, max_simplex_dim)
        for s in layer_complex:
            all_simplices_set.add(s)
            if s not in simplex_times:
                simplex_times[s] = []
            simplex_times[s].append(time_val)

        # Intersection complex (between layer_idx and layer_idx+1)
        if layer_idx < n_layers - 1:
            time_val_int = 2 * layer_idx + 1  # odd
            int_edges = layer_edges[layer_idx] & layer_edges[layer_idx + 1]
            int_complex = _flag_complex(int_edges, vertices, max_simplex_dim)
            for s in int_complex:
                all_simplices_set.add(s)
                if s not in simplex_times:
                    simplex_times[s] = []
                simplex_times[s].append(time_val_int)

    # Build Dionysus filtration
    simplex_list = sorted(all_simplices_set, key=lambda s: (len(s), s))
    f = d.Filtration([list(s) for s in simplex_list])

    # Build times: for each simplex, list of time points where it's present
    times = []
    for s in simplex_list:
        t = sorted(simplex_times.get(s, []))
        times.append(t)

    # Run zigzag persistence
    zz, dgms, cells = d.zigzag_homology_persistence(f, times, prime=2)

    # Convert to numpy arrays
    result = []
    for dim in range(max_simplex_dim):
        pairs = []
        for pt in dgms[dim] if dim < len(dgms) else []:
            if pt.death != float("inf"):
                pairs.append([pt.birth, pt.death])
        if pairs:
            result.append(np.array(pairs))
        else:
            result.append(np.empty((0, 2)))

    return result


def persistence_entropy(lifetimes: np.ndarray) -> float:
    """Shannon entropy of normalized persistence lifetimes."""
    if len(lifetimes) == 0 or lifetimes.sum() <= 0:
        return 0.0
    p = lifetimes / lifetimes.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def zigzag_features(
    layer_clouds: list[np.ndarray],
    k: int = 4,
    max_simplex_dim: int = 2,
    long_lived_threshold: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Extract scalar features from zigzag persistence diagrams.

    Parameters
    ----------
    layer_clouds : list of (n_tokens, hidden_dim) per layer
    k : kNN connectivity
    max_simplex_dim : max simplex dimension for flag complex
    long_lived_threshold : min layer span for "long-lived" cycle count

    Returns
    -------
    (~6,) feature array and list of feature names
    """
    dgms = compute_zigzag_diagrams(layer_clouds, k=k, max_simplex_dim=max_simplex_dim)

    features = []
    names = []

    for dim_idx, dim_name in enumerate(["H0", "H1"]):
        dgm = dgms[dim_idx] if dim_idx < len(dgms) else np.empty((0, 2))
        life = dgm[:, 1] - dgm[:, 0] if len(dgm) > 0 else np.array([])

        features.append(life.sum() if len(life) > 0 else 0.0)
        names.append(f"zz_{dim_name}_total_persistence")

        features.append(persistence_entropy(life))
        names.append(f"zz_{dim_name}_entropy")

        n_long = int(np.sum(life > long_lived_threshold)) if len(life) > 0 else 0
        features.append(float(n_long))
        names.append(f"zz_{dim_name}_n_long_lived")

    return np.array(features, dtype=np.float64), names
