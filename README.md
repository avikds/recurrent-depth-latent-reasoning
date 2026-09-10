# Recurrent-Depth Latent Reasoning

Build the recurrent-depth language model of Geiping et al. (2025) at toy scale in PyTorch: a prelude that embeds the input, a core block stack that is looped with the input re-injected every iteration from a random initial state, and a coda that decodes. Train it on multi-digit addition with a random loop budget per step and truncated backpropagation, then scale test-time compute without emitting a single extra token: sweep the loop budget, watch the latent state converge, and exit adaptively when the KL divergence between successive loops drops below a threshold.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_addition_batch
- [x] **2.** tokens_to_str
- [x] **3.** RMSNorm
- [x] **4.** CausalSelfAttention
- [x] **5.** Block
- [x] **6.** Prelude
- [x] **7.** InputAdapter
- [x] **8.** Core
- [x] **9.** Coda
- [x] **10.** RecurrentDepthLM
- [x] **11.** sample_loop_count
- [x] **12.** truncated_recurrence
- [x] **13.** masked_lm_loss
- [x] **14.** train_step
- [x] **15.** train_recurrent
- [x] **16.** answer_accuracy
- [x] **17.** accuracy_vs_loops
- [x] **18.** state_convergence
- [x] **19.** kl_between_loops
- [x] **20.** adaptive_exit_forward
- [x] **21.** greedy_add

## Results

```
recurrent-depth model: 208,832 parameters, 107,456 of them in the looped core

trained 600 steps; loop budgets seen: min 1, mean 4.8, max 12
  step   1: answer loss 2.648  (loops 3)
  step 100: answer loss 1.158  (loops 12)
  step 200: answer loss 1.011  (loops 9)
  step 300: answer loss 0.522  (loops 2)
  step 400: answer loss 0.243  (loops 8)
  step 500: answer loss 0.111  (loops 2)
  step 600: answer loss 0.040  (loops 6)

sequence accuracy vs loop budget (same seed, same 256 problems):
   1 loops: 0.809   2 loops: 0.945   4 loops: 0.934   8 loops: 0.930  16 loops: 0.930  32 loops: 0.930
  (toy scale: the curve should rise then flatten; exact values move with the seed)

relative state change per loop: [0.3746, 0.1341, 0.0506, 0.0205, 0.0086, 0.0036, 0.0015, 0.0006, 0.0003]
KL between successive predictions: ['1.43e-01', '1.65e-02', '1.86e-03', '3.68e-04', '5.30e-05', '6.00e-06', '1.00e-06', '0.00e+00', '0.00e+00']
  adaptive exit at KL < 0.01: stopped after 4 loops
  adaptive exit at KL < 0.0001: stopped after 6 loops

greedy addition with 12 loops:
  47 + 38 = 85  (ok)
  99 + 1 = 100  (ok)
  12 + 34 = 46  (ok)
  65 + 79 = 144  (ok)
```
