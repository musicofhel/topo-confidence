# Handoff — topo-confidence (2026-06-13)

## SUMMARY

- **Accomplished this session:**
  - **confgate publicly released on GitHub** → https://github.com/musicofhel/confgate
    - Standalone public repo, MIT licensed, fresh git history (no private monorepo internals).
    - Tagged release **v0.1.0** with `topo_confgate-0.1.0` wheel + sdist attached.
    - 8 discovery topics, README carries the pinned v6/v7/v8 evidence + bake-off.
    - Verified: 8/8 pytest, clean `python -m build`, `twine check` PASSED, fresh-venv install smoke test OK, pinned data JSONs bundled in the wheel.
  - SPEC v8 confirmed complete/committed from prior session (commit `560f433` on branch `max-depth-retriage-2026-04-28`). Nothing reopened.

- **Decisions:**
  - confgate extracted as a **standalone repo** (`~/confgate`), not a monorepo subdir — it's an installable package and the research repo is private. `~/confgate` is now canonical; `~/topo-confidence/confgate` is the original in-repo copy and has **diverged** in the `name`/URL fields (expected for an extraction).
  - PyPI distribution name = **`topo-confgate`** (user choice). Bare `confgate` is owned by an unrelated package (Sri Harsha, "Confidence-gated decisions for LLM agent outputs", already 0.1.1). **Import name stays `confgate`** (`pip install topo-confgate` → `import confgate`).
  - Key rotation comment dropped per user ("forget the runkey comment, i dont care").

- **Current state:**
  - GitHub release fully live and self-consistent (assets + notes both say `topo-confgate`).
  - **PyPI upload is the only open release step** — built & twine-checked at `~/confgate/dist/`, blocked solely on a PyPI token (none in env/.pypirc/keyring; intentionally not requested through chat). User runs:
    `! cd ~/confgate && UV_PUBLISH_TOKEN=pypi-YOUR_TOKEN uv publish`
    (or `TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-... twine upload dist/*`).

- **Blockers:** PyPI token (user-side, optional — GitHub release already installable via git/release URL).

- **Next steps (NEW EXPERIMENTS — the actual focus):**
  **Query the link-forge research DB for high-ROI research directions**, then propose the next experiment(s). Concretely:
  1. Use the link-forge MCP tools to mine the literature DB for under-explored, high-leverage directions adjacent to the project's live story:
     - `mcp__link-forge__forge_ask` — natural-language questions over the DB (e.g. "highest-ROI directions for improving small-LLM correctness via cheap inference-time signals or base/target swaps").
     - `mcp__link-forge__forge_search`, `forge_recent`, `forge_related`, `forge_concepts`, `forge_find_tools` — to triangulate clusters and recent work.
     - `mcp__link-forge__scout_discover` / `scout_similar` — GitHub-side discovery.
  2. Cross-check against the project's own queue + novelty graph BEFORE proposing anything (mandatory per memory `feedback-topo-confidence-novelty-check`):
     - `cat ~/topo-confidence/NEXT_EXPERIMENTS.md`
     - `cd ~/topo-confidence/research-graph && python query.py status-report` and `python query.py novelty "<claim>"`.
  3. Anchor proposals to **v8's reframing of where the headroom is** (see Notes) — the ceiling moves by *swapping the base/target*, NOT by probes/introspection (v7 closed) or gate-curated distillation (v8 H-R refuted). So high-ROI directions should NOT re-litigate probe-vs-gate or curation; look for *new levers* on the rescue-density / base-quality / escalation-target axes, or genuinely new surfaces from the literature.
  4. Produce a ranked shortlist (ROI + novelty + cost) and let the user pick before any GPU spend. Honor: **no local GPU ever**, **H100 SXM only on RunPod**, **pods have no network volume** (pull JSONs before stop), **claims invariant never regresses**.

- **Notes / load-bearing context:**
  - **Project goal: improve small models via hidden-state understanding, NOT papers** (`feedback-topo-confidence-goal`).
  - **v8 verdict (the frame for "what's next"):** *swap the base/target, don't curate data.*
    - H-N CONFIRMED: escalation target = Qwen2.5-Math-7B-Instruct, cascade +4.20pp (0.648→0.690), rescues 41/121.
    - H-P CONFIRMED (cheap headline): off-the-shelf Qwen2.5-Math-1.5B-Instruct 0.740 @ ¼ budget (+9.2pp) — the base is the real lever. R1-Distill REFUTED (length intrinsic).
    - H-Q CONFIRMED: free gate (length+logprob) generalizes to all new bases (OOF 0.863/0.950/0.936).
    - H-O INCONCLUSIVE (Mathstral 15/121). **H-R REFUTED**: gate-curated distillation is a free-rider (gate 0.494 < unfiltered 0.500 < skyline 0.504; gate = step-length proxy; MATH-only SFT forgets BBH).
    - Binding constraint = **rescue density** (47% of 1.5B failures are unrescuable by the 7B) — a prime target for a new direction.
  - v7 already closed: no probe beats the free gate at matched cost; marginal compute should buy *escalation, not introspection*; cross-domain certs need k=32 true labels; cross-scale certs work zero-label.
  - Memory files to lean on: `topo-confidence-spec-v8`, `topo-confidence-spec-v7`, `topo-confidence-spec-v4` (v6), `feedback-topo-confidence-novelty-check`, `feedback-no-local-gpu-even-rescoring`, `runpod-preferences`, `link-forge-research-graph-bridge`.
  - link-forge research DB is the designated literature-sweep surface (Neo4j `bolt://localhost:7687`, separate from the topo research-graph at `:7688`).
  - North-star metric: matched-cost MATH-500 accuracy at the v7 operating point (budget 2220.7 token-FLOPs/problem; cost unit 1 = one 1.5B token, 7B-class = 4.7×; oracle 0.758).
