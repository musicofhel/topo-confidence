"""Command-line interface for topo-confidence."""

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def cmd_score(args):
    """Score one or more prompts."""
    from topo_confidence import TopoConfidence

    tc = TopoConfidence(model_name=args.model, device=args.device)

    if args.calibration:
        tc.load(args.calibration)

    prompts = args.prompts
    if args.file:
        with open(args.file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        print("No prompts provided. Use positional args or --file.", file=sys.stderr)
        sys.exit(1)

    if tc.calibrated:
        confidences = tc.predict_confidence(prompts)
        for prompt, conf in zip(prompts, confidences):
            print(f"{conf:.3f}\t{prompt[:80]}")
    else:
        # Raw feature mode
        result = tc.extractor.extract(prompts)
        for i, traj in enumerate(result["token_trajectories"]):
            features = tc.feature_extractor.extract_single(traj)
            feature_dict = dict(
                zip(tc.feature_extractor.feature_names, features.tolist())
            )
            print(json.dumps({"prompt": prompts[i][:80], "features": feature_dict}))


def cmd_explain(args):
    """Show feature-level breakdown for a single prompt."""
    from topo_confidence import TopoConfidence

    tc = TopoConfidence(model_name=args.model, device=args.device)

    if args.calibration:
        tc.load(args.calibration)
    else:
        print("--calibration required for explain mode.", file=sys.stderr)
        sys.exit(1)

    explanation = tc.explain(args.prompt)
    print(f"Confidence: {explanation['confidence']:.3f}")
    print(f"Top contributor: {explanation['top_contributor']}")
    print()
    for name, vals in explanation["features"].items():
        if vals["contribution"] > 0:
            bar = "+" * int(abs(vals["contribution"]) * 20)
        else:
            bar = "-" * int(abs(vals["contribution"]) * 20)
        print(
            f"  {name:30s}  raw={vals['raw_value']:8.4f}"
            f"  contribution={vals['contribution']:+.4f}  {bar}"
        )


def cmd_calibrate(args):
    """Calibrate on a JSONL file with 'prompt' and 'correct' fields."""
    import numpy as np

    from topo_confidence import TopoConfidence

    tc = TopoConfidence(model_name=args.model, device=args.device)

    prompts = []
    correct = []
    with open(args.data) as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row["prompt"])
            correct.append(int(row["correct"]))

    print(f"Calibrating on {len(prompts)} examples...")
    metrics = tc.calibrate(prompts, np.array(correct))
    print(f"AUROC: {metrics.get('auroc', 'N/A')}")

    tc.save(args.output)
    print(f"Saved calibrated model to {args.output}")


def cmd_selective(args):
    """Generate answers only when confident."""
    from topo_confidence import TopoConfidence

    tc = TopoConfidence(model_name=args.model, device=args.device)

    if args.calibration:
        tc.load(args.calibration)
    else:
        print("--calibration required for selective mode.", file=sys.stderr)
        sys.exit(1)

    prompts = []
    if args.file:
        with open(args.file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = args.prompts

    if not prompts:
        print("No prompts provided.", file=sys.stderr)
        sys.exit(1)

    results = tc.selective_predict(prompts, threshold=args.threshold)

    answered = 0
    for prompt, answer, conf in zip(
        prompts, results["answers"], results["confidences"]
    ):
        status = "ANSWER" if answer is not None else "SKIP"
        answered += 1 if answer is not None else 0
        print(f"[{status}] conf={conf:.3f} | {prompt[:60]}")
        if answer is not None:
            print(f"  -> {answer[:200]}")

    print(f"\nAnswered: {answered}/{len(prompts)} ({answered / len(prompts):.0%})")
    print(
        f"Expected accuracy on answered: {results['expected_accuracy_on_answered']:.1%}"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="topo-confidence",
        description="Topological confidence estimation for LLM outputs",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--calibration", "-c", default=None, help="Path to calibrated model (.pkl)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # score
    p_score = subparsers.add_parser("score", help="Score prompts")
    p_score.add_argument("prompts", nargs="*", help="Prompts to score")
    p_score.add_argument("--file", "-f", help="File with one prompt per line")
    p_score.set_defaults(func=cmd_score)

    # explain
    p_explain = subparsers.add_parser(
        "explain", help="Explain a single prediction"
    )
    p_explain.add_argument("prompt", help="Prompt to explain")
    p_explain.set_defaults(func=cmd_explain)

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Calibrate on labeled data")
    p_cal.add_argument(
        "data", help="JSONL file with 'prompt' and 'correct' fields"
    )
    p_cal.add_argument(
        "--output", "-o", default="calibrated.pkl", help="Output path"
    )
    p_cal.set_defaults(func=cmd_calibrate)

    # selective
    p_sel = subparsers.add_parser(
        "selective", help="Answer only when confident"
    )
    p_sel.add_argument("prompts", nargs="*")
    p_sel.add_argument("--file", "-f", help="File with one prompt per line")
    p_sel.add_argument("--threshold", "-t", type=float, default=0.7)
    p_sel.set_defaults(func=cmd_selective)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
