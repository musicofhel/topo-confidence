#!/usr/bin/env python3
"""Interactive demo of topo-confidence."""

from __future__ import annotations

import argparse
import logging

from topo_confidence import TopoConfidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Topo-confidence demo")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration", default=None,
                        help="Path to calibrated model (.pkl)")
    args = parser.parse_args()

    tc = TopoConfidence(model_name=args.model, device=args.device)

    if args.calibration:
        tc.load(args.calibration)
        logger.info("Loaded calibrated model from %s", args.calibration)
    else:
        # Quick self-calibration with toy examples
        logger.info("No calibration file provided. Running quick self-calibration...")
        toy_prompts = [
            "What is 2 + 2?",
            "What is 15 * 7?",
            "What is the capital of France?",
            "What is the square root of 144?",
            "Solve: 3x + 7 = 22",
            "What is the derivative of x^3?",
            "What is 1000 / 8?",
            "Name a prime number between 20 and 30.",
            "What is the integral of 1/x?",
            "How many sides does a hexagon have?",
        ]
        # Assume simple questions are correct for demo
        import numpy as np
        toy_correct = np.ones(len(toy_prompts), dtype=int)
        tc.calibrate(toy_prompts, toy_correct)
        logger.info("Self-calibrated on %d toy examples (demo only!)", len(toy_prompts))

    print("\n" + "=" * 60)
    print("topo-confidence interactive demo")
    print("Type a question/problem and press Enter.")
    print("Type 'quit' to exit, 'explain' for feature breakdown.")
    print("=" * 60 + "\n")

    last_prompt = None
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            break
        if prompt.lower() == "explain" and last_prompt:
            explanation = tc.explain(last_prompt)
            print(f"\nConfidence: {explanation['confidence']:.3f}")
            print(f"Top contributor: {explanation['top_contributor']}")
            print("\nFeature breakdown:")
            for name, vals in explanation["features"].items():
                print(f"  {name}: raw={vals['raw_value']:.4f}, "
                      f"contribution={vals['contribution']:+.4f}")
            print()
            continue

        last_prompt = prompt
        result = tc.selective_predict([prompt], threshold=0.0, max_new_tokens=256)
        conf = result["confidences"][0]
        answer = result["answers"][0]

        print(f"\nConfidence: {conf:.3f}", end="")
        if conf >= 0.7:
            print(" [HIGH]")
        elif conf >= 0.4:
            print(" [MEDIUM]")
        else:
            print(" [LOW]")
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
