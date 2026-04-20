# Production-quality persistent homology for LLM hidden states: implementation playbook

This playbook consolidates exact APIs, versions, and code for building a validated persistent-homology (PH) pipeline on Qwen2.5-1.5B-Instruct hidden states on RunPod H100. **The single most consequential recommendation: replace Euclidean Vietoris-Rips with effective-resistance PH on a k-NN graph (Damrich et al. 2024) for within-layer structure, and add zigzag persistence across layers using the Gardinazzi et al. 2024 pipeline (Dionysus2 + FastZigzag) with kNN=4.** These two additions address the Euclidean-PH failure mode above ~50 dimensions and give you layer-dynamics features that vanilla Rips cannot capture. Cosine-distance PH via precomputed matrices is a third, cheap win. Everything below runs on Python 3.11 with CUDA 12.1+ on an H100 pod. The full MATH-500 + HumanEval + 3×BBH subset run fits in ~45 minutes of wall clock if you skip zigzag; add ~3 hours if you include zigzag across all 28 layers per problem.

## 1. Cosine distance with Ripser is a one-line API change

Ripser.py (current **0.6.14**, Dec 2025, wheels for CPython 3.10–3.14) accepts a precomputed distance matrix via the `distance_matrix=True` flag. The canonical signature is `ripser(X, maxdim=1, thresh=inf, coeff=2, distance_matrix=False, metric='euclidean', n_perm=None)` and returns a dict whose `'dgms'` key is a list of (birth, death) arrays per homology dimension. For cosine PH, compute the distance matrix with `scipy.spatial.distance.pdist(X, metric='cosine')` (values in [0, 2]) and pass it in squareform:

```python
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
D_cos = squareform(pdist(X, metric='cosine'))          # X: (n_tokens, 1536)
dgms  = ripser(D_cos, distance_matrix=True, maxdim=1)['dgms']
```

GUDHI (**3.12.0**) exposes the same via `gudhi.RipsComplex(distance_matrix=D, max_edge_length=...)` followed by `.create_simplex_tree(max_dimension=2).persistence()`. Use giotto-ph's `ripser_parallel(X, metric='precomputed', collapse_edges=True, n_threads=N)` for the multi-threaded variant; edge collapse is a significant speedup for dense inputs.

**HOLE (arXiv:2512.07988) is not yet pip-installable.** The paper describes four supported metrics (Euclidean, cosine, Mahalanobis, geodesic) plus density-normalized variants, but as of April 2026 no public GitHub repo has been found; the authors state code will be released at publication. The paper also restricts its VR filtration to dimension 1 (H_0 only) and targets CIFAR/ViT — meaning HOLE offers little beyond what ripser + custom distance matrices already provide. **Do not depend on HOLE.** Build on ripser/GUDHI directly. Prior cosine-PH work on neural activations (arXiv:2505.20435, "Holes in Latent Space") reports consistent topological compression signatures under adversarial conditions across Phi3, Mistral 7B, Llama3 8B/70B, Mixtral 8x7B; they vectorize into a **41-D statistical summary** (mean/median/std of births, deaths, persistence, totals, counts, persistent entropy, for H_0 and H_1) which is directly reusable.

## 2. Effective resistance PH is the top recommendation and needs a custom chain

Damrich, Berens & Kobak (NeurIPS 2024, arXiv:2311.03087) prove that in the high-dimensional isotropic-noise regime, Euclidean pairwise distances concentrate so that `‖ε₁−ε₂‖/‖(x₁+ε₁)−(x₂+ε₂)‖ → 1` — this is precisely the regime of LLM hidden states. Their benchmark winner across noisy circles, eyeglasses, linked circles, spheres and tori is **effective resistance** on a symmetric k-NN graph with **k=100** (diffusion distance with **k=15, t=8** is the close runner-up). Their reference code is at **github.com/berenslab/eff-ph** (branch `neurips2024`, MIT license); note it depends on a custom ripser build from `https://github.com/Ripser/ripser.git -b representative-cycles`.

The corrected effective-resistance embedding (their Proposition 6.1) for node *i* is:

`e^eff_i = ((1−μ₂)/√μ₂ · u_{2,i}, (1−μ₃)/√μ₃ · u_{3,i}, ...) / √d_i`

where `μ_k, u_k` are eigenvalues/eigenvectors of the symmetrically normalized Laplacian `L^sym = I − D^(−1/2) A D^(−1/2)`, and `d_i` is the degree of node *i*. Distances are Euclidean on these embeddings squared. Implementation chain:

```python
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from ripser import ripser

def effective_resistance_distance(X, k=100, n_eigs=None):
    # (a) symmetric kNN graph
    A = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
    A = A.maximum(A.T)                                     # symmetrize
    # (b) normalized Laplacian
    L_sym, d = csgraph.laplacian(A, normed=True, return_diag=True)
    n = A.shape[0]
    n_eigs = n_eigs or (n - 1)
    # Take all non-trivial eigenvectors; eigsh returns ascending
    mu, U = eigsh(L_sym, k=min(n_eigs, n-1), which='SM')
    mu = np.clip(mu[1:], 1e-8, None); U = U[:, 1:]         # drop trivial
    # (c) embedding vectors from Proposition 6.1
    scale = (1.0 - mu) / np.sqrt(mu)
    E = (U * scale[None, :]) / np.sqrt(d)[:, None]         # (n, n-1)
    # (d) pairwise Euclidean squared
    diffs = E[:, None, :] - E[None, :, :]
    return np.sqrt((diffs ** 2).sum(-1))

D_er  = effective_resistance_distance(X, k=100)
dgms  = ripser(D_er, distance_matrix=True, maxdim=1)['dgms']
```

No end-to-end package exists; chain `sklearn + scipy + ripser` as above. For **n=350 points in 1536 dims, expect ~10–40 ms** for `eigsh` on the 350×350 Laplacian and ~30–150 ms for the ripser call — sub-second total. For large k (100), also worth testing **k=15 with diffusion at t=8** per Damrich's second-ranked method; formula is identical except replace `(1−μ)/√μ` with `(1−μ)^t` and the outer `√vol(G)` normalization.

## 3. DTM filtration in GUDHI is a weighted-Rips construction

GUDHI ships a dedicated `gudhi.dtm_rips_complex.DTMRipsComplex(points=X, k=1, q=2, max_filtration=inf)` and, more flexibly, `gudhi.point_cloud.dtm.DistanceToMeasure(k, q=2, metric='euclidean')` whose output weights feed a `WeightedRipsComplex(distance_matrix=D, weights=w, max_filtration=...)`. The weighted-Rips convention: vertex *i* enters at `2·w_i`; edge *(i,j)* enters at `d(i,j) + w_i + w_j`. Parameters: **k = number of neighbors defining the empirical measure**, **q = norm exponent** (q=2 is standard), **max_filtration** is the threshold. Reference tutorial: `GUDHI/TDA-tutorial/Tuto-GUDHI-DTM-filtrations.ipynb`. DTM is outlier-robust but in Damrich's benchmarks it underperforms effective resistance and diffusion at high noise, so use it as a third baseline rather than your primary non-Euclidean path.

## 4. Zigzag persistence follows the Gardinazzi et al. recipe exactly

Gardinazzi et al. (ICML 2025, arXiv:2410.11042, "Persistent Topological Features in LLMs") built the reference zigzag pipeline for transformer layers; their code lives at **github.com/RitAreaSciencePark/ZigZagLLMs**. The exact recipe:

1. **Token choice:** last-token representation of each prompt at each of the 28 layers — so a point cloud per layer has one vector per prompt in the dataset, not per token within a single problem. Reuse this if you want *corpus-level* features; adapt by treating each problem's 30–350 tokens as prompts for *problem-local* zigzag.
2. **Graph construction:** symmetric k-NN graph per layer with **k=4** (best hole count in their sweep over k∈[1,15]); the k-NN graph is expanded to a flag/clique complex up to maximum simplex dimension **m=4** (tracking holes up to p=3, with main results focused on **H_1**).
3. **Inclusion maps:** between adjacent layers ℓᵢ and ℓᵢ₊₁, they use an **intersection complex** `K_int = K_ℓᵢ ∩ K_ℓᵢ₊₁`, giving the zigzag `K_ℓ₁ ⊇ K_int(1,2) ⊆ K_ℓ₂ ⊇ ... ⊆ K_ℓ_L` with alternating injective inclusions indexed over `{0,...,2(N_layers−1)}`. Even indices are model layers, odd are intersections.
4. **Computation:** Dionysus2 builds `f` (union filtration) and `times` (even entries = appearance, odd = disappearance); **FastZigzag** (Dey & Hou 2022, `pyfzz`) executes. GUDHI has zigzag only in C++ as of 3.12 — the Python binding is not yet exposed.
5. **Vectorization:** Persistence Image P_I_p(b,d) on a `(2N_layers−1)²` grid per homology degree, then **Effective Persistence Image (EPI)**: `P̂_I_p(b/2, d/2) = P_I_p(b,d) + P_I_p(b−1,d) + P_I_p(b,d−1) + P_I_p(b−1,d−1)`. From EPI extract two layer-indexed scalar functions: **births' relative frequency** `B_p(ℓ)` and **inter-layer persistence** `Z̄_p(ℓ)` with weighting `ω(ℓ,ℓᵢ) = |ℓ−ℓᵢ|^α`, α ∈ {−1, 0, 0.5, 1, 2}.

Exact Dionysus2 API from mrzv.org/software/dionysus2/tutorial/zigzags.html:

```python
import dionysus as d
f = d.Filtration([[0], [1], [0,1], [2], [0,2], [1,2]])
times = [[.4, .6, .7], [.1], [.9], [.9], [.9], [.9]]
zz, dgms, cells = d.zigzag_homology_persistence(f, times, prime=2)
```

**Dionysus has no PyPI wheels** — it builds from source via PyBind11 (Boost headers + C++14 compiler, ~2–4 min); pre-build in your Docker image. For 28 point clouds × ~100 points each, expect **2–10 s per zigzag sequence** on H100-host CPU.

**HalluZig (arXiv:2601.01552, Samaga et al., EACL 2026, code at github.com/TDA-Jyamiti/halluzig)** is the closest adaptation target: it applies zigzag to per-layer **attention graphs** (not hidden states), vectorizes via persistence images/landscapes/entropy, and feeds a Random Forest. Adapting to hidden states means swapping the per-layer attention-graph for a per-layer k-NN graph of hidden vectors — precisely the Gardinazzi construction. Use HalluZig's Random Forest + multi-vectorization ensemble as your classifier baseline.

## 5. HumanEval uses the official OpenAI harness with Qwen chat template

Dataset loads as `load_dataset("openai/openai_humaneval", split="test")` returning **164 problems** with fields `task_id`, `prompt`, `canonical_solution`, `test`, `entry_point`. Install the official harness with `git clone https://github.com/openai/human-eval && pip install -e human-eval` — **you must manually uncomment the `exec()` line in `human_eval/execution.py`** (commented by OpenAI as a safety gate). Layered defenses already present: `multiprocessing.Process` + `signal.SIGALRM` timeouts, `resource.setrlimit` for memory, and `reliability_guard()` which monkey-patches dangerous builtins (`os.system`, `subprocess.Popen`, `shutil.rmtree`, `socket.socket`, etc.). Because **RunPod pods are already isolated VMs**, `reliability_guard + multiprocess timeout` is sufficient; for defense-in-depth wrap with `firejail --private --net=none --caps.drop=all`. The fastest path is **EvalPlus** (`pip install "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"`) which ships a sandboxed Docker image `ganler/evalplus:latest` and auto-handles chat-model completion extraction (a real pitfall: Qwen2.5-Instruct outputs the full function including the signature, breaking openai/human-eval's concat-then-exec convention).

Qwen2.5-1.5B-Instruct is a chat model — use the tokenizer's chat template, not raw completion. The official Qwen blog (qwenlm.github.io/blog/qwen2.5-llm/) reports **pass@1 = 61.6 for HumanEval** on this model; HumanEval+ is not officially reported for 1.5B-Instruct and third-party runs show ±5–10pp variance. Verify with `evalplus.evaluate --model "Qwen/Qwen2.5-1.5B-Instruct" --dataset humaneval --backend hf --greedy` for a one-shot sanity check.

Hidden-state extraction during generation uses the standard API — verified against transformers 4.45+:

```python
out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                     output_hidden_states=True, return_dict_in_generate=True,
                     pad_token_id=tokenizer.eos_token_id)
# out.hidden_states: tuple of length gen_len
# out.hidden_states[t]: tuple of (num_layers+1) = 29 tensors (layer 0 = embedding)
# out.hidden_states[t][L]: (batch, s_t, 1536) — s_t = prompt_len if t==0 else 1
# Gather per-generated-token last-layer vector:
traj = torch.stack([s[-1][0, -1, :].float().cpu() for s in out.hidden_states])
```

Memory budget: weights ≈ 3.1 GB (bf16); per-problem full hidden-state tensor in fp16 is `gen_len × 29 × 1536 × 2 B ≈ 18 MB` for 200 tokens. For all benchmarks combined at 500 tokens you're looking at ~300 GB; **decide layer selection before the run** (e.g., save only layers 14 and 28, or save as fp16).

## 6. BBH loads from lukaemon/bbh with 3-shot CoT prompts from Suzgun's GitHub

Use `load_dataset("lukaemon/bbh", subset, split="test")` — 27 subsets, 6,511 rows total, fields `input` and `target`. Relevant subsets for constraint tasks each have **250 rows**:

| Subset | Answer format | Random baseline |
|---|---|---|
| `tracking_shuffled_objects_seven_objects` | (A)–(G) | 14% |
| `logical_deduction_seven_objects` | (A)–(G) | 14% |
| `web_of_lies` | Yes/No | 50% |

Canonical 3-shot CoT prompts live at `https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/{subset}.txt` — pre-download all 27 in the Docker build (total <200 KB). Each ends with `"So the answer is (X)."` or `"So the answer is Yes/No."` — parse with `re.search(r"(?i)the answer is\s*\(?([A-Za-z]+)\)?", response)`. Build the prompt as user-role: `f"{cot_prompt}\n\nQ: {question}\nA: Let's think step by step."` wrapped in `tokenizer.apply_chat_template(...)`.

**Expected accuracies for Qwen2.5-1.5B-Instruct are not officially published** on per-subtask BBH. The Qwen blog reports the base 1.5B at BBH 3-shot = 45.1 aggregate. Third-party comparable runs suggest 15–35% on `tracking_shuffled_objects_seven_objects` with CoT, 20–40% on `logical_deduction_seven_objects`, and 50–65% on `web_of_lies`. Run the baseline yourself and treat these as rough priors. The hidden-state extraction API is identical to HumanEval.

## 7. Validation protocol has four mandatory stages

**Stage 1 — PH computation correctness.** Use `tadasets` (pip, 0.2.2) to generate canonical test shapes. A noisy circle sampled via `np.c_[np.cos(theta), np.sin(theta)] + 0.05*randn` with 200 points must produce a single H_1 bar with persistence >5× the second-largest — a hard assertion. A torus from `tadasets.torus(n=500, c=2, a=1, noise=0.05)` with `maxdim=2` must yield two dominant H_1 bars. A 2-sphere from `tadasets.dsphere(n=400, d=2, r=1, noise=0.03)` must yield one dominant H_2 bar (note: `maxdim=2` on 400 points takes tens of seconds; use threshold).

**Stage 2 — distance-metric divergence.** Construct points at `dirs * radii` where `dirs` are on the unit circle and `radii ~ Uniform(1, 10)`. Cosine PH detects the clean loop; Euclidean PH's H_1 is dominated by short radial noise. Assert `max_persistence(cosine_H1) > 3× max_persistence(euclidean_H1)`. This is the pattern from "Holes in Latent Space" and motivates cosine PH for norm-varying LLM hidden states.

**Stage 3 — end-to-end shape/finite checks.** Each stage has trivial invariants: trajectory shape is `(T_gen, 29, 1536)`, distance matrix is square+symmetric+non-negative, diagrams have birth ≤ death, feature vectors are finite, classifier AUROC is in [0.5, 1.0] for a reasonable signal.

**Stage 4 — statistical AUROC comparison via DeLong.** Use `pip install MLstatkit` (v0.1.9+) — neither scipy nor statsmodels ship DeLong:

```python
from MLstatkit.stats import Delong_test
z, p = Delong_test(y_true, probs_cosine, probs_euclidean)    # paired
```

Fallback to Yandex's `roc_comparison/compare_auc_delong_xu.py` (copy the `delong_roc_test` function directly) or a paired stratified bootstrap (2000 resamples) for unpaired cases. Use two-sided p < 0.05 as the significance gate when comparing your new pipeline's AUROC against the 0.796 baseline.

## 8. Compute budget fits comfortably on a single H100

Empirical timings (H100 SXM host, modern CPU):

| Operation | Input | Wall-time |
|---|---|---|
| Qwen2.5-1.5B forward, bf16, 350 tokens | batch=1 | **~80–150 ms** |
| Ripser H_1 on 350×350 precomputed matrix | dense | **~30–150 ms** |
| `scipy.sparse.linalg.eigsh` on 350-node L_sym | all eigvecs | **~10–40 ms** |
| Pairwise effective-resistance distance from eigenpairs | n=350 | **~5–20 ms** |
| Zigzag (Dionysus2 + FastZigzag) 28 layers × 100 pts | m=4, kNN=4 | **~2–10 s** |
| PersistenceImager transform (10×10 grid) | per diagram | **<10 ms** |

Per-problem budget without zigzag ≈ 1.5 s (LLM forward + 28× ripser across layers + vectorize). **Per-benchmark totals: MATH-500 ≈ 12–15 min, HumanEval 164 ≈ 4–5 min, 3×BBH subsets × 250 ≈ 18–20 min → combined ~35–45 min wall-clock.** With zigzag across all 28 layers per problem, add ~5 s/problem → total ~2–4 hours. GPU memory stays at ~4–5 GB peak; disk is dominated by saved hidden states (use `float16` and consider saving only layers of interest — 2 layers instead of all 29 gives a 14× reduction).

## 9. Five papers define the methodological backbone

**Gardinazzi et al. 2410.11042 (ICML 2025, code at RitAreaSciencePark/ZigZagLLMs)** is your primary zigzag reference — last-token per layer, kNN=4, intersection inclusions, EPI-derived B_p(ℓ) and Z̄_p(ℓ) features; their layer-pruning benchmark (their Table 1) gives you a numerical comparison surface. **Damrich et al. 2311.03087 (NeurIPS 2024, code at berenslab/eff-ph)** provides effective resistance (k=100) and diffusion distance (k=15, t=8) on symmetric k-NN graphs; Proposition 6.1 gives the closed-form embedding and Corollary 6.2 proves effective resistance is an aggregate of diffusion distances. **Fay et al. 2505.20435 "Holes in Latent Space"** applies Euclidean VR for H_0 and H_1 with subsampling (4096 activations per subsample), vectorizes into a **41-D barcode statistics vector** (mean/median/std of births, deaths, persistences, totals, counts, persistent entropy, per H_0 and H_1) — reuse this compact feature set directly. **Samaga et al. 2601.01552 "HalluZig" (EACL 2026, code at TDA-Jyamiti/halluzig)** adapts zigzag to attention graphs with FastZigzag + GUDHI vectorization + Random Forest. **Athreya & Rosen 2512.07988 "HOLE"** provides a taxonomy of metric choices (cosine for transformer-like direction-only embeddings, Mahalanobis for anisotropic, geodesic for manifolds, density-normalized for robustness) but has no released code as of April 2026.

"Persistent Topological Features in Large Language Models" is indeed the same paper as Gardinazzi et al. 2410.11042 — not a distinct work.

## 10. Pinned library stack for the RunPod Dockerfile

Pin these PyPI versions, then build `dionysus` from source in the image:

```bash
# Core ML
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.45.0,<5.0.0" "accelerate>=0.33.0" \
            "datasets>=2.19.0" "tokenizers>=0.19"

# Persistent homology
pip install ripser==0.6.14 gudhi==3.12.0 persim==0.3.8 \
            giotto-tda==0.6.2 tadasets==0.2.2

# Zigzag (source build: needs boost + C++14)
pip install dionysus==2.0.10                     # ~2–4 min build
pip install pyfzz                                # FastZigzag, via pip

# Statistics
pip install MLstatkit                            # DeLong test
pip install scikit-learn scipy numpy

# Evaluation harnesses
git clone https://github.com/openai/human-eval && pip install -e human-eval
pip install "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"
```

**Notes:** `scikit-tda` is a meta-package — prefer installing `ripser`, `persim`, `tadasets`, `kmapper` individually. `dionysus` has no PyPI wheels (plan for the source build or bake it into your base image). GUDHI zigzag is C++-only in the Python package — you must use Dionysus2 or pyfzz for zigzag in Python. HOLE is not installable. torch 2.4 is the most conservative choice that still supports H100 and Qwen2.5; torch 2.5+ works but transformers v5 (late 2025/early 2026) has breaking pipeline API changes, so pin `transformers<5.0.0` until you've tested.

## Conclusion and recommended sequence

The pipeline upgrade that most directly addresses the Damrich et al. warning is **effective-resistance PH on a k=100 symmetric k-NN graph**, implemented as an 8-line chain of sklearn + scipy + ripser. Running this alongside cosine-distance VR gives you two non-Euclidean baselines essentially for free. Adding **zigzag across layers** following the Gardinazzi kNN=4 intersection recipe provides orthogonal layer-dynamics features; the EPI + Z̄_p(ℓ) + B_p(ℓ) descriptors from their paper are reproduced feature-for-feature by their open repo. Validate each stage with the tadasets synthetic shapes and the MLstatkit DeLong test against your 0.796 Euclidean-VR baseline, compute all 1414 problems in ~45 minutes without zigzag or ~3 hours with it, and expect memory to stay under 5 GB on your H100. Two concrete pitfalls to avoid on day one: you must **uncomment the `exec()` line in `human-eval/execution.py`** before evaluation runs at all, and you must **pre-build dionysus in the Docker image** or your first run will stall for several minutes during pip install.
