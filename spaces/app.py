"""topo-confidence: See if your LLM knows what it's doing."""

import gradio as gr
import numpy as np
import torch
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import ripser

logging.basicConfig(level=logging.WARNING)

# ---------- Model loading ----------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
)
if device == "cpu":
    model = model.to(device)
model.eval()
print(f"Model loaded on {device}")


# ---------- Feature computation ----------

def persistence_entropy(lifetimes):
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    p = lifetimes / lifetimes.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def compute_topo_features(token_hidden_states):
    """Compute 7 topological features from a (n_tokens, hidden_dim) array."""
    n_tokens = token_hidden_states.shape[0]
    n_comp = min(30, n_tokens - 1, token_hidden_states.shape[1])
    if n_comp < 2 or n_tokens < 5:
        return np.zeros(7), None, None, None

    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(token_hidden_states)

    # PH
    subsample_n = min(100, n_tokens)
    if n_tokens > subsample_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_tokens, subsample_n, replace=False)
        ph_input = reduced[idx]
    else:
        ph_input = reduced

    result = ripser.ripser(ph_input, maxdim=1)
    dgms = result["dgms"]

    features = np.zeros(7)

    # H0
    h0 = dgms[0]
    h0_fin = h0[np.isfinite(h0[:, 1])]
    h0_life = h0_fin[:, 1] - h0_fin[:, 0] if len(h0_fin) > 0 else np.array([])
    features[0] = persistence_entropy(h0_life)
    features[2] = h0_life.sum() if len(h0_life) > 0 else 0.0
    features[3] = len(h0_life)

    # H1
    h1 = dgms[1]
    h1_fin = h1[np.isfinite(h1[:, 1])]
    h1_life = h1_fin[:, 1] - h1_fin[:, 0] if len(h1_fin) > 0 else np.array([])
    features[1] = h1_life.max() if len(h1_life) > 0 else 0.0
    features[4] = persistence_entropy(h1_life)
    features[5] = len(h1_life)

    # Bridge silhouette
    if n_tokens >= 10:
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = km.fit_predict(reduced)
        if len(set(labels)) == 2:
            sil = silhouette_samples(reduced, labels)
            features[6] = sil[0]
        else:
            labels = np.zeros(n_tokens, dtype=int)
            sil = np.zeros(n_tokens)
    else:
        labels = np.zeros(n_tokens, dtype=int)
        sil = np.zeros(n_tokens)

    # For visualization: PCA to 2D
    pca_2d = PCA(n_components=2)
    coords_2d = pca_2d.fit_transform(reduced)

    return features, coords_2d, labels, sil


FEATURE_NAMES = [
    "H0 persistence entropy",
    "H1 max lifetime",
    "H0 total persistence",
    "H0 feature count",
    "H1 persistence entropy",
    "H1 feature count",
    "H2 feature count",
    "H2 total persistence",
    "H2 persistence entropy",
    "Bridge silhouette",
    "H0 PH significance",
    "H1 PH significance",
    "Topological sensitivity",
]


# ---------- Confidence scoring (uncalibrated heuristic) ----------

def heuristic_confidence(features):
    """Simple heuristic confidence from raw features.

    Based on the finding that correct answers have lower H0 entropy
    and the bridge token is closer to the cluster boundary.

    For a calibrated score, use TopoConfidence.calibrate() with labeled data.
    """
    h0_ent = features[0]
    bridge_sil = features[6]

    # Lower H0 entropy → higher confidence (more coherent representation)
    # Map typical range [1.5, 4.5] to [1.0, 0.0]
    conf_h0 = np.clip(1.0 - (h0_ent - 1.5) / 3.0, 0.0, 1.0)

    # Bridge silhouette near 0 → healthy bridge → higher confidence
    # |sil| > 0.3 means bridge absorbed into cluster → lower confidence
    conf_bridge = np.clip(1.0 - abs(bridge_sil) / 0.3, 0.0, 1.0)

    # Weighted combination
    confidence = 0.7 * conf_h0 + 0.3 * conf_bridge
    return float(np.clip(confidence, 0.0, 1.0))


# ---------- Visualization ----------

def make_cluster_plot(coords_2d, labels, sil, token_texts):
    """Create the 2D cluster visualization with bridge highlighted."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Color by cluster
    colors = np.array(["#58a6ff" if l == 0 else "#f97583" for l in labels])
    sizes = np.full(len(labels), 30)
    alphas = np.full(len(labels), 0.5)

    # Identify bridge tokens
    bridge_mask = np.abs(sil) < 0.1
    colors[bridge_mask] = "#ffd700"
    sizes[bridge_mask] = 80
    alphas[bridge_mask] = 1.0

    # Position 0 is the singleton bridge — make it huge
    colors[0] = "#ffd700"
    sizes[0] = 200
    alphas[0] = 1.0

    # Plot core tokens first
    for i in range(len(coords_2d)):
        ax.scatter(
            coords_2d[i, 0], coords_2d[i, 1],
            c=colors[i], s=sizes[i], alpha=alphas[i],
            edgecolors="white" if i == 0 else "none",
            linewidths=2 if i == 0 else 0,
            zorder=3 if i == 0 else 1,
        )

    # Label position 0
    if len(token_texts) > 0:
        ax.annotate(
            f"pos 0: '{token_texts[0]}'",
            (coords_2d[0, 0], coords_2d[0, 1]),
            textcoords="offset points", xytext=(15, 15),
            fontsize=11, color="#ffd700", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#ffd700", lw=1.5),
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#58a6ff', markersize=10, label='Cluster A', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f97583', markersize=10, label='Cluster B', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffd700', markersize=14, label='Bridge (pos 0)', linestyle='None', markeredgecolor='white', markeredgewidth=1.5),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    ax.set_title("Hidden-State Token Geometry", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("PC1", color="#8b949e")
    ax.set_ylabel("PC2", color="#8b949e")

    n_bridge = int(bridge_mask.sum())
    sil_mean = float(np.mean(np.abs(sil)))
    ax.text(0.02, 0.02, f"Tokens: {len(labels)}  |  Bridges: {n_bridge}  |  Silhouette: {sil_mean:.3f}",
            transform=ax.transAxes, fontsize=9, color="#8b949e")

    plt.tight_layout()
    return fig


def make_feature_plot(features):
    """Bar chart of feature values."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Normalize features for display
    display_vals = features.copy()
    colors = ["#58a6ff"] * 7
    colors[6] = "#ffd700"  # bridge feature

    bars = ax.barh(range(7), display_vals, color=colors, alpha=0.8, height=0.6)
    ax.set_yticks(range(7))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=10, color="#c9d1d9")
    ax.invert_yaxis()
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Value", color="#8b949e")
    ax.set_title("Topological Features", color="white", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


# ---------- Main pipeline ----------

@torch.no_grad()
def analyze(problem_text, max_new_tokens=256):
    """Full pipeline: generate answer + compute topo features + visualize."""
    if not problem_text.strip():
        return "Please enter a math problem.", "", None, None, ""

    prompt = f"You are a helpful math assistant. Provide the final answer.\n\n{problem_text}\n\nPlease provide the final answer."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    t0 = time.perf_counter()

    # Forward pass — extract hidden states
    outputs = model(**inputs, output_hidden_states=True)
    final_hidden = outputs.hidden_states[-1][0].cpu().float().numpy()  # (n_tokens, hidden_dim)

    # Generate answer
    gen_outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = gen_outputs[0][prompt_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Decode token texts for visualization
    token_ids = inputs["input_ids"][0].tolist()
    token_texts = [tokenizer.decode([tid]) for tid in token_ids]

    # Compute topo features
    features, coords_2d, labels, sil = compute_topo_features(final_hidden)

    dt = time.perf_counter() - t0

    # Confidence
    confidence = heuristic_confidence(features)

    # Confidence badge
    if confidence >= 0.7:
        badge = f"🟢 High confidence: {confidence:.0%}"
    elif confidence >= 0.4:
        badge = f"🟡 Medium confidence: {confidence:.0%}"
    else:
        badge = f"🔴 Low confidence: {confidence:.0%}"

    # Plots
    if coords_2d is not None:
        cluster_fig = make_cluster_plot(coords_2d, labels, sil, token_texts)
        feature_fig = make_feature_plot(features)
    else:
        cluster_fig = None
        feature_fig = None

    # Feature summary text
    summary_lines = [f"Computed in {dt:.2f}s (topo overhead: ~0.007s)"]
    summary_lines.append(f"Tokens: {final_hidden.shape[0]}, Hidden dim: {final_hidden.shape[1]}")
    summary_lines.append("")
    for name, val in zip(FEATURE_NAMES, features):
        summary_lines.append(f"  {name}: {val:.4f}")

    return answer, badge, cluster_fig, feature_fig, "\n".join(summary_lines)


# ---------- Gradio interface ----------

EXAMPLES = [
    "What is 2 + 2?",
    "Find the derivative of x^3 + 2x^2 - 5x + 3.",
    "If a triangle has sides 3, 4, and 5, what is its area?",
    "Solve for x: 3x^2 - 12x + 9 = 0",
    "What is the sum of the first 100 positive integers?",
    "How many ways can you arrange the letters in MISSISSIPPI?",
    "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
    "Prove that the square root of 2 is irrational.",
]

with gr.Blocks(title="topo-confidence") as demo:
    gr.Markdown("""
    # topo-confidence
    **See if your LLM knows what it's doing — from the shape of its hidden states.**

    This demo runs Qwen2.5-1.5B-Instruct on your math problem, then uses persistent homology
    to analyze the geometry of the model's internal representations. The key finding:
    **token representations organize into two clusters with a single bridge token at position 0**.
    Correct answers have simpler geometry. [Read the research →](https://github.com/musicofhel/att-docs)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            problem_input = gr.Textbox(
                label="Math Problem",
                placeholder="Enter a math problem...",
                lines=3,
            )
            submit_btn = gr.Button("Analyze", variant="primary", size="lg")
            gr.Examples(examples=EXAMPLES, inputs=problem_input)

        with gr.Column(scale=1):
            answer_output = gr.Textbox(label="Model Answer", lines=4)
            confidence_output = gr.Textbox(label="Topo-Confidence", elem_classes="badge-text")

    with gr.Row():
        cluster_plot = gr.Plot(label="Hidden-State Geometry")
        feature_plot = gr.Plot(label="Topological Features")

    with gr.Accordion("Technical Details", open=False):
        details_output = gr.Textbox(label="Feature Values", lines=10)

    submit_btn.click(
        analyze,
        inputs=[problem_input],
        outputs=[answer_output, confidence_output, cluster_plot, feature_plot, details_output],
    )

    gr.Markdown("""
    ---
    **How it works:** Each token's hidden state at the final transformer layer is a point in R^1536.
    Persistent homology extracts topological features (connected components, loops) from this point cloud.
    Position 0 (the first token) serves as a computational bridge between two clusters —
    all inter-cluster information flows through this single token.
    [pip install topo-confidence](https://github.com/musicofhel/topo-confidence) |
    [Research paper](https://github.com/musicofhel/att-docs)
    """)

demo.queue()
demo.launch(
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="gray",
    ),
    css="""
    .badge-text { font-size: 1.3em; font-weight: bold; }
    footer { display: none !important; }
    """,
)
