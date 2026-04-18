#!/usr/bin/env python3
"""Step 0: Validate the corrected answer checker against known edge cases.

Tests extract_answer_v2, normalize_answer_v2, and check_correct_v2 with
hand-curated cases from MATH-500 known failures + synthetic edge cases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    extract_answer_v2,
    extract_boxed_balanced,
    check_correct_v2,
    normalize_answer_v2,
)

PASS = 0
FAIL = 0


def test(description: str, result: bool, expected: bool = True):
    global PASS, FAIL
    status = "PASS" if result == expected else "FAIL"
    if result != expected:
        FAIL += 1
        print(f"  [{status}] {description}  (got {result}, expected {expected})")
    else:
        PASS += 1
        print(f"  [{status}] {description}")


def test_eq(description: str, result: str, expected: str):
    global PASS, FAIL
    status = "PASS" if result == expected else "FAIL"
    if result != expected:
        FAIL += 1
        print(f"  [{status}] {description}  (got {result!r}, expected {expected!r})")
    else:
        PASS += 1
        print(f"  [{status}] {description}")


def main():
    print("=" * 70)
    print("CHECKER VALIDATION")
    print("=" * 70)

    # ---- Balanced brace extraction ----
    print("\n--- extract_boxed_balanced ---")

    test_eq(
        "Simple boxed integer",
        extract_boxed_balanced("The answer is \\boxed{42}."),
        "42",
    )
    test_eq(
        "Boxed fraction (nested braces)",
        extract_boxed_balanced("Therefore \\boxed{\\frac{14}{3}}"),
        "\\frac{14}{3}",
    )
    test_eq(
        "Boxed sqrt (nested braces)",
        extract_boxed_balanced("We get \\boxed{3\\sqrt{13}}."),
        "3\\sqrt{13}",
    )
    test_eq(
        "Boxed complex nested expression",
        extract_boxed_balanced("\\boxed{\\frac{3\\sqrt{3}}{4}}"),
        "\\frac{3\\sqrt{3}}{4}",
    )
    test_eq(
        "Multiple boxed: return LAST",
        extract_boxed_balanced("\\boxed{wrong} then \\boxed{42}"),
        "42",
    )
    test_eq(
        "Boxed with \\left/\\right",
        extract_boxed_balanced("\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}"),
        "\\left( 3, \\frac{\\pi}{2} \\right)",
    )
    test_eq(
        "Deeply nested",
        extract_boxed_balanced("\\boxed{\\frac{\\sqrt{2}}{\\sqrt{3}}}"),
        "\\frac{\\sqrt{2}}{\\sqrt{3}}",
    )
    test(
        "No boxed returns None",
        extract_boxed_balanced("There is no boxed answer") is None,
    )

    # ---- extract_answer_v2 ----
    print("\n--- extract_answer_v2 ---")

    test_eq(
        "Boxed takes priority over ####",
        extract_answer_v2("#### 5\nThe answer is \\boxed{42}"),
        "42",
    )
    test_eq(
        "Boxed with nested frac",
        extract_answer_v2("So \\boxed{\\frac{1}{2}} is the answer."),
        "\\frac{1}{2}",
    )
    test_eq(
        "#### fallback when no boxed",
        extract_answer_v2("#### 1234\n"),
        "1234",
    )
    test_eq(
        "Answer is pattern",
        extract_answer_v2("The answer is 7."),
        "7",
    )
    test_eq(
        "Last number fallback",
        extract_answer_v2("We compute 3 + 4 = 7"),
        "7",
    )

    # ---- check_correct_v2 ----
    print("\n--- check_correct_v2 (sympy path) ---")

    test(
        "\\frac{3}{4} == 0.75",
        check_correct_v2("\\boxed{\\frac{3}{4}}", "\\frac{3}{4}"),
    )
    # Model outputs 0.75, ground truth is \frac{3}{4}
    test(
        "0.75 == \\frac{3}{4} (float vs latex)",
        check_correct_v2("#### 0.75", "\\frac{3}{4}"),
    )
    test(
        "\\frac{14}{3} == \\frac{14}{3}",
        check_correct_v2("\\boxed{\\frac{14}{3}}", "\\frac{14}{3}"),
    )
    test(
        "\\frac{16}{3} != \\frac{14}{3}",
        check_correct_v2("\\boxed{\\frac{16}{3}}", "\\frac{14}{3}"),
        expected=False,
    )
    test(
        "3\\sqrt{13} == 3\\sqrt{13}",
        check_correct_v2("\\boxed{3\\sqrt{13}}", "3\\sqrt{13}"),
    )
    test(
        "\\sqrt{5} != 5",
        check_correct_v2("\\boxed{5}", "\\sqrt{5}"),
        expected=False,
    )
    test(
        "\\pi sympy comparison",
        check_correct_v2("\\boxed{\\pi}", "\\pi"),
    )
    test(
        "\\frac{1}{2} == 0.5",
        check_correct_v2("\\boxed{\\frac{1}{2}}", "0.5"),
    )

    print("\n--- check_correct_v2 (numeric path) ---")

    test(
        "Integer match: 42 == 42",
        check_correct_v2("#### 42", "42"),
    )
    test(
        "Float match: 3.14 == 3.14",
        check_correct_v2("#### 3.14", "3.14"),
    )
    test(
        "Comma removal: 1,234 == 1234",
        check_correct_v2("#### 1,234", "1234"),
    )
    test(
        "Negative: -7 == -7",
        check_correct_v2("\\boxed{-7}", "-7"),
    )

    print("\n--- check_correct_v2 (string path) ---")

    test(
        "String match: case insensitive",
        check_correct_v2("\\boxed{TRIANGLE}", "triangle"),
    )
    test(
        "String with \\text{}",
        check_correct_v2("\\boxed{\\text{yes}}", "yes"),
    )

    # ---- Known false negatives from MATH-500 audit ----
    print("\n--- Known MATH-500 false negatives ---")

    # idx=0: gt=\left( 3, \frac{\pi}{2} \right), predicted=(3, \frac{\pi
    # The old checker truncated due to lazy regex
    # With balanced braces, if model output \boxed{\left( 3, \frac{\pi}{2} \right)},
    # it should extract correctly
    test(
        "idx=0 style: boxed tuple with pi",
        check_correct_v2(
            "\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}",
            "\\left( 3, \\frac{\\pi}{2} \\right)",
        ),
    )

    # idx=8: gt=3\sqrt{13}, model output had \boxed{3\sqrt{13}}
    # Old: extracted "3\sqrt{13" (missing closing brace of sqrt)
    test(
        "idx=8 style: boxed sqrt",
        check_correct_v2("\\boxed{3\\sqrt{13}}", "3\\sqrt{13}"),
    )

    # idx=82: gt=\frac{3}{2}, model output had \boxed{\frac{3}{2}}
    # Old: extracted "\frac{3" (truncated)
    test(
        "idx=82 style: boxed fraction",
        check_correct_v2("\\boxed{\\frac{3}{2}}", "\\frac{3}{2}"),
    )

    # idx=82 variant: model outputs 1.5 for ground truth \frac{3}{2}
    test(
        "1.5 == \\frac{3}{2}",
        check_correct_v2("#### 1.5", "\\frac{3}{2}"),
    )

    # ---- GSM8K specific ----
    print("\n--- GSM8K edge cases ---")

    test(
        "GSM8K comma number",
        check_correct_v2("#### 12,345", "12345"),
    )
    test(
        "GSM8K negative",
        check_correct_v2("#### -42", "-42"),
    )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    print("=" * 70)

    if FAIL > 0:
        print("\nWARNING: Some tests failed. Review before proceeding.")
        sys.exit(1)
    else:
        print("\nAll tests passed. Checker is ready.")

    # Save results
    output_dir = Path(__file__).parent.parent / "phase0_relabel"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "checker_validation.txt", "w") as f:
        f.write(f"Checker validation: {PASS} passed, {FAIL} failed\n")


if __name__ == "__main__":
    main()
