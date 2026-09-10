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

---

Built on Deep-ML.
