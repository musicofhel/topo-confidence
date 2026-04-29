# Edu-Fetch PDF Triage — 2026-04-27

**Status:** All 51 PDFs ingested into link-forge (`:Link` nodes with `file://` URLs).

**Gap identified:** The research-graph hook (`link-forge/src/processor/research-graph-suggest.ts:121-125`) only fires on URLs containing an arxiv ID. PDFs with `file://` URLs are silently skipped — none of today's 51 PDFs were checked for relevance against `FINDINGS.md` / `HYPOTHESES.md`. To add them as `:Paper {status:"candidate"}` requires looking up arxiv IDs externally (publisher PDFs don't embed them).

---

## Tier A — Strong relevance (15): topology/PH on hidden states, ICL, steering, causal rep

### 1 s2.0 S0370157320302489 main
- **authors:** Federico Battiston, Giulia Cencetti, Iacopo Iacopini, Vito Latora +4
- **concepts:** hypergraphs, simplicial-complexes, higher-order-laplacians, simplicial-homology, combinatorial-laplacian, higher-order-kuramoto-model
- **hash:**   ·  **type:** research-paper

### s41598 023 28985 3
- **authors:** Ysanne Pritchard, Aikta Sharma, Claire Clarkin, Helen Ogden +2
- **concepts:** persistent-homology, signed-euclidean-distance-transform, persistence-statistics, persistence-diagrams, cubical-complex, second-harmonic-generation-imaging
- **hash:**   ·  **type:** research-paper

### Adversarially Trained Persistent Homology Based Graph Convolutional Network for Disease Identification Using Brain Connectivity
- **authors:** Chenyuan Bian, Nan Xia, Anmu Xie, Shan Cong +1
- **concepts:** persistent-homology, graph-convolutional-networks, adversarial-training, brain-connectomics, sparse-representation, ROI-pooling
- **hash:**   ·  **type:** research-paper

### 157 166
- **authors:** Vin de Silva, Gunnar Carlsson
- **concepts:** witness-complex, persistent-homology, simplicial-complex, delaunay-triangulation, cech-complex, rips-complex
- **hash:**   ·  **type:** research-paper

### 2023.findings emnlp.462
- **authors:** Cheng Qian, Chi Han, Yi R. Fung, Yujia Qin +2
- **concepts:** tool-creation, creator-framework, abstract-vs-concrete-reasoning, chain-of-thought, program-of-thought, tool-augmented-llms
- **hash:**   ·  **type:** research-paper

### 2023.findings emnlp.459
- **authors:** Guozheng Li, Peng Wang, Wenjun Ke
- **concepts:** summarize-and-ask-prompting, zero-shot-relation-extraction, chain-of-thought-prompting, in-context-learning, vanilla-prompting, qa4re
- **hash:**   ·  **type:** research-paper

### 2024.emnlp main.64
- **authors:** Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng +9
- **concepts:** in-context-learning, demonstration-selection, symbol-tuning, meta-icl, instruction-induction, induction-heads
- **hash:**   ·  **type:** research-paper

### Toward Causal Representation Learning
- **authors:** Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke +3
- **concepts:** causal-representation-learning, independent-causal-mechanisms, sparse-mechanism-shift-hypothesis, structural-causal-models, interventions, counterfactual-reasoning
- **hash:**   ·  **type:** research-paper

### Causal Deconfounding for Spurious Correlation in Domain Generalization
- **authors:** Bin Qin, Yi Li, Jiangmeng Li, Xuesong Wu +2
- **concepts:** structural-causal-model, backdoor-adjustment, do-operation, propensity-score-weighting, confounding-bias, invariant-risk-minimization
- **hash:**   ·  **type:** research-paper

### PHG Net Persistent Homology Guided Medical Image Classification
- **authors:** Yaopeng Peng, Hongxiao Wang, Milan Sonka, Danny Z. Chen
- **concepts:** persistent-homology, cubical-persistence, persistence-diagram, topological-data-analysis, pointnet, sub-level-set-filtration
- **hash:**   ·  **type:** research-paper

### s41060 022 00332 1
- **authors:** Satoru Watanabe, Hayato Yamana
- **concepts:** persistent-homology, topological-data-analysis, clique-complex, PHOM, NPHOM, filtration
- **hash:**   ·  **type:** research-paper

### Persistent Homology based Graph Convolution Network for Fine grained 3D Shape Segmentation
- **authors:** Chi-Chong Wong, Chi-Man Vong
- **concepts:** persistent-homology, topological-data-analysis, persistence-diagram-loss, graph-convolution-network, simplicial-complex, filtration
- **hash:**   ·  **type:** research-paper

### J Comput Chem   2018   Wu   TopP S  Persistent homology‐based multi‐task deep neural networks for simultaneous predictions
- **authors:** Kedi Wu, Zhixiong Zhao, Renxiao Wang, Guo-Wei Wei
- **concepts:** element-specific-persistent-homology, betti-numbers, barcodes, multi-task-deep-learning, qsar, qspr
- **hash:**   ·  **type:** research-paper

### quantitative toxicity prediction using topology based multitask deep neural networks
- **authors:** Kedi Wu, Guo-Wei Wei
- **concepts:** element-specific-persistent-homology, persistent-homology, element-specific-topological-descriptors, multitask-deep-neural-networks, qsar, betti-numbers
- **hash:**   ·  **type:** research-paper

### applsci 14 02074
- **authors:** Rajvardhan Patil, Venkat Gudivada
- **concepts:** transformer-architecture, self-attention, autoregressive-language-modeling, pretraining-objectives, transfer-learning, in-context-learning
- **hash:**   ·  **type:** research-paper

## Tier B — Peripheral relevance (10): broad LLM, generic topology, deconfounding

### 2022.acl long.62
- **authors:** Fangxiaoyu Feng, Yinfei Yang, Daniel Cer +2
- **concepts:** language-agnostic-bert-sentence-embedding, dual-encoder-architecture, translation-ranking-loss, additive-margin-softmax, masked-language-modeling
- **hash:** \n
### A Review on Large Language Models Architectures Applications Taxonomies Open Issues and Challenges
- **authors:** Mohaimenul Azam Khan Raiaan, Md. Saddam Hossain Mukta, Kaniz Fatema +6
- **concepts:** transformer-architecture, self-attention, pre-trained-language-models, gpt, bert
- **hash:** \n
### s11704 024 40231 1
- **authors:** Lei Wang, Chen Ma, Xueyang Feng +10
- **concepts:** llm-based-autonomous-agents, unified-agent-framework, profiling-module, memory-module, planning-module
- **hash:** \n
### 296916885 oa
- **authors:** Xu Yang, Kaihua Tang, Hanwang Zhang +1
- **concepts:** scene-graph-auto-encoder, scene-graphs, graph-convolutional-networks, multi-modal-gcn, language-inductive-bias
- **hash:** \n
### techrxiv.23589741.v1
- **authors:** Muhammad Usman Hadi, Qasem Al-Tashi, Rizwan Qureshi +8
- **concepts:** generative-pre-trained-transformers, statistical-language-models, neural-language-models, pre-trained-language-models, masked-language-modeling
- **hash:** \n
### 2021.06.16.448764v1.full
- **authors:** Oualid Benkarim, Casey Paquola, Bo-yong Park +10
- **concepts:** propensity-scores, population-stratification, distribution-drift, ignorability, default-mode-network
- **hash:** \n
### 1 s2.0 S0168169923001734 main
- **authors:** Zhongke Qu, Yang Zhang, Chao Hong +8
- **concepts:** graph-convolutional-networks, transformer-architecture, spatiotemporal-modeling, multi-output-forecasting, sensor-network-topology
- **hash:** \n
### bbad161
- **authors:** Weihe Dong, Qiang Yang, Jian Wang +4
- **concepts:** multi-modality-attributes-learning, molecular-transformer, heterogeneous-graph-convolutional-network, SMILES-tokenization, adaptive-weight-fusion
- **hash:** \n
### journal.pbio.3001627
- **authors:** Oualid Benkarim, Casey Paquola, Bo-yong Park +10
- **concepts:** propensity-scores, population-heterogeneity, cross-validation, default-mode-network, deconfounding
- **hash:** \n
### AIChE Journal   2018   Venkatasubramanian   The promise of artificial intelligence in chemical engineering  Is it here
- **authors:** Venkat Venkatasubramanian
- **concepts:** expert-systems, knowledge-based-systems, production-system-framework, process-systems-engineering, CONPHYDE
- **hash:** \n
## Tier C — Likely off-topic (26): cosmology, NMR, drug-protein, sensor nets, etc.

- 331499.331504 — _A.K. Jain, M.N. Murty +1_
- 34393VoR — _Sam Bond-Taylor, Adam Leach +2_
- s10462 023 10466 8 — _Shams Forruque Ahmed, Md. Sakib Bin Alam +7_
- q 2018 08 06 79 — _John Preskill_
- aa25830 15 — _Planck Collaboration, P. A. R. Ade +4_
- 19880001113 — _Stanley Osher, James A. Sethian_
- 978 3 319 55556 0 — _Isaac Pesenson, Quoc Thong Le Gia +4_
- 978 3 319 55550 8 — _Isaac Pesenson, Quoc Thong Le Gia +4_
- s12559 023 10179 8 — _Vikas Hassija, Vinay Chamola +8_
- J of Applied Econometrics   November 1996   Papke   Econometric methods for fractional response variables with an — _Leslie E. Papke, Jeffrey M. Wooldridge_
- Computational and Mathematical Methods in Medicine   2019   Kocbek   Maximizing Interpretability and Cost‐Effectiveness of — _Primoz Kocbek, Nino Fijacko +8_
- 2988450.2988454 — _Heng-Tze Cheng, Levent Koc +14_
- ACM computing surveys — _João Gama, Indrė Žliobaitė +3_
- 978 3 030 80519 7 — _Joseph F. Hair Jr., G. Tomas M. Hult +4_
- make 06 00071 v2 — _Stephen Fox, Vitor Fortes Rey_
- make 06 00004 — _Adnan Alagic, Natasa Zivic +5_
- s42256 024 00974 9 — _Samson J. Mataraso, Camilo A. Espinosa +18_
- 265 — _Tirumala Rao Gundala_
- 07872494 — _Mojtaba Forouzesh, Yam P. Siwakoti +3_
- applsci 11 04440 — _Youheng Tan, Xiaojun Jing_
- Addiction   2005   WEST   Time for a change  putting the Transtheoretical  Stages of Change  Model to rest — _Robert West_
- bmj.n160.full — _Matthew J Page, David Moher +24_
- ijcv voc09 — _Mark Everingham, Luc Van Gool +3_
- The CMS Collaboration 2008 J. Inst. 3 S08004 — _CMS Collaboration, S. Chatrchyan +3_
- PhysRev.73.679 — _N. Bloembergen, E. M. Purcell +1_
- s41586 024 08025 4 — _Sumanth Dathathri, Abigail See +22_

---

## What to do with this

1. **Quick win — Tier A only.** ~24 papers worth looking up arxiv IDs for. The PH-on-hidden-states papers are mostly methodologically adjacent to F-10 (PH = Gaussian null on residual streams) — they apply PH on _other_ objects (graph attention, brain connectivity, molecules) and don't directly contradict F-10. Most-likely-novel triggers: **Schölkopf causal rep learning** (could trigger experiments on whether L19 DoM is causal vs correlate — see existing P11-FE3 activation patching), **Dong ICL Survey** (induction heads), **SynthID Text** (steering methodology).
2. **Long-term fix — patch the bridge.** Modify  to also accept PDF uploads: extract arxiv ID via Semantic-Scholar title-match or DOI->arxiv lookup, then run the existing relevance check. Without this, every future edu-fetch batch re-creates the same gap.
3. **Already done:**  regenerated (24 experiments, 25 watchlist papers). The current top-of-queue (P8-FE1, P9-FE1, P10-FE1, P11-FE5, P3-FE1, P11-FE3 — all ROI 9-10) does not depend on today's PDFs being added; you can run any of them now.

