"""
Recurrent-Depth Latent Reasoning scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Recurrent-Depth Latent Reasoning.

Story: a prelude embeds an addition problem, a shared core is looped with the
input re-injected every time from a random initial state, and a coda decodes.
Train with a random loop budget and truncated backprop, then scale test-time
compute in loops instead of tokens: sweep the budget, watch the latent state
converge, exit on a KL threshold, and add numbers with the model.
"""
import torch


def main() -> None:
    n_digits = 2
    torch.manual_seed(0)
    model = RecurrentDepthLM(VOCAB_SIZE, d=64, n_heads=4, n_prelude=1, n_core=2, n_coda=1, max_len=3 * n_digits + 3)
    n_params = sum(p.numel() for p in model.parameters())
    core_params = sum(p.numel() for p in model.core.parameters())
    print(f"recurrent-depth model: {n_params:,} parameters, {core_params:,} of them in the looped core")

    # ---- 1. Train with a random loop budget ----
    hist = train_recurrent(model, steps=600, n_digits=n_digits, batch_size=64, lr=2e-3, mean_loops=4, max_loops=12, k_backprop=4)
    counts = hist["loop_counts"]
    print(f"trained 600 steps; loop budgets seen: min {min(counts)}, mean {sum(counts) / len(counts):.1f}, max {max(counts)}")
    for i in (0, 99, 199, 299, 399, 499, 599):
        print(f"  step {i + 1:3d}: answer loss {hist['losses'][i]:.3f}  (loops {counts[i]})")

    # ---- 2. Test-time scaling in loops ----
    batch = make_addition_batch(256, n_digits, torch.Generator().manual_seed(123))
    curve = accuracy_vs_loops(model, batch, [1, 2, 4, 8, 16, 32])
    print("\nsequence accuracy vs loop budget (same seed, same 256 problems):")
    print("  " + "  ".join(f"{k:2d} loops: {v:.3f}" for k, v in curve.items()))
    print("  (toy scale: the curve should rise then flatten; exact values move with the seed)")

    # ---- 3. Convergence and adaptive exit ----
    x = batch[0][:8]
    print("\nrelative state change per loop:", state_convergence(model, x, n_loops=10))
    print("KL between successive predictions:", [f"{v:.2e}" for v in kl_between_loops(model, x, n_loops=10)])
    for thr in (1e-2, 1e-4):
        _, used, _ = adaptive_exit_forward(model, x, max_loops=32, kl_threshold=thr)
        print(f"  adaptive exit at KL < {thr:g}: stopped after {used} loops")

    # ---- 4. Add numbers ----
    print("\ngreedy addition with 12 loops:")
    for a, b in ((47, 38), (99, 1), (12, 34), (65, 79)):
        pred = greedy_add(model, a, b, n_digits, n_loops=12)
        print(f"  {a} + {b} = {pred}  ({'ok' if pred == a + b else 'wrong, truth ' + str(a + b)})")


if __name__ == "__main__":
    main()

