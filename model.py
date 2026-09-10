"""
Recurrent-Depth Latent Reasoning

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_addition_batch
import torch

PLUS, EQ, VOCAB_SIZE = 10, 11, 12

def make_addition_batch(batch_size, n_digits, generator=None):
    # Draw a and b uniformly from [0, 10**n_digits).
    limit = 10 ** n_digits
    a = torch.randint(
        0,
        limit,
        (batch_size,),
        generator=generator,
        dtype=torch.int64,
    )
    b = torch.randint(
        0,
        limit,
        (batch_size,),
        generator=generator,
        dtype=torch.int64,
    )

    # Powers of 10 used to extract decimal digits from most significant
    # to least significant position.
    powers_a = 10 ** torch.arange(
        n_digits - 1, -1, -1, dtype=torch.int64
    )
    powers_sum = 10 ** torch.arange(
        n_digits, -1, -1, dtype=torch.int64
    )

    # Convert a and b to zero-padded digit sequences of length n_digits.
    a_digits = (a[:, None] // powers_a[None, :]) % 10
    b_digits = (b[:, None] // powers_a[None, :]) % 10

    # Convert a + b to a zero-padded digit sequence of length n_digits + 1.
    sum_digits = ((a + b)[:, None] // powers_sum[None, :]) % 10

    # Build:
    # digits(a) + [PLUS] + digits(b) + [EQ] + digits(a + b)
    seq = torch.cat(
        [
            a_digits,
            torch.full((batch_size, 1), PLUS, dtype=torch.int64),
            b_digits,
            torch.full((batch_size, 1), EQ, dtype=torch.int64),
            sum_digits,
        ],
        dim=1,
    ).to(torch.int64)

    # Next-token prediction setup.
    x = seq[:, :-1]
    y = seq[:, 1:]

    # Only the answer digits in y are included in the loss/evaluation mask.
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, -(n_digits + 1):] = True

    return x, y, mask

# Step 2 - tokens_to_str
def tokens_to_str(ids):
    # Accept either a PyTorch tensor or a Python list.
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()

    if not isinstance(ids, (list, tuple)):
        raise TypeError("ids must be a 1-D torch.Tensor, list, or tuple.")

    tokens = []

    for token in ids:
        token = int(token)

        if 0 <= token <= 9:
            tokens.append(str(token))
        elif token == PLUS:
            tokens.append("+")
        elif token == EQ:
            tokens.append("=")
        else:
            raise ValueError(f"Invalid token id: {token}")

    return "".join(tokens)

# Step 3 - RMSNorm
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

# Step 4 - CausalSelfAttention
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()

        if d % n_heads != 0:
            raise ValueError("d must be divisible by n_heads.")

        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, d = x.shape

        # Project to queries, keys, and values.
        qkv = self.qkv(x)

        # Split into Q, K, V: (B, T, 3, H, head_dim)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)

        # Rearrange to (3, B, H, T, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention scores: (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: each position can only attend to itself and earlier positions.
        causal_mask = torch.ones(
            T,
            T,
            device=x.device,
            dtype=torch.bool,
        ).tril()

        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Normalize attention scores.
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum of values: (B, H, T, head_dim)
        out = attn @ v

        # Merge heads: (B, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, d)

        # Final projection.
        return self.proj(out)

# Step 5 - Block
class Block(nn.Module):
    def __init__(self, d, n_heads, mlp_mult=4):
        super().__init__()

        self.norm1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_mult * d),
            nn.GELU(),
            nn.Linear(mlp_mult * d, d),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# Step 6 - Prelude
class Prelude(nn.Module):
    def __init__(self, vocab_size, d, n_heads, n_blocks, max_len):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads) for _ in range(n_blocks)]
        )

    def forward(self, idx):
        B, T = idx.shape

        # Position indices shared across the batch.
        pos = torch.arange(T, device=idx.device)

        # Token + positional embeddings.
        e = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]

        # Process through the transformer blocks in order.
        for block in self.blocks:
            e = block(e)

        return e

# Step 7 - InputAdapter
class InputAdapter(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.proj = nn.Linear(2 * d, d)

    def forward(self, e, s):
        return self.proj(torch.cat([e, s], dim=-1))

# Step 8 - Core
class Core(nn.Module):
    def __init__(self, d, n_heads, n_blocks):
        super().__init__()

        self.adapter = InputAdapter(d)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads) for _ in range(n_blocks)]
        )

    def forward(self, e, s):
        # Inject the prelude output together with the current latent state.
        s = self.adapter(e, s)

        # Process the adapted state through the core blocks.
        for block in self.blocks:
            s = block(s)

        return s

    def iterate(self, e, s0, n_loops, return_all=False):
        s = s0
        states = []

        for _ in range(n_loops):
            s = self.forward(e, s)

            if return_all:
                states.append(s)

        if return_all:
            return states

        return s

# Step 9 - Coda
class Coda(nn.Module):
    def __init__(self, d, vocab_size, n_heads, n_blocks):
        super().__init__()

        self.blocks = nn.ModuleList(
            [Block(d, n_heads) for _ in range(n_blocks)]
        )
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, s):
        for block in self.blocks:
            s = block(s)

        s = self.norm(s)
        logits = self.head(s)

        return logits

# Step 10 - RecurrentDepthLM
class RecurrentDepthLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d,
        n_heads,
        n_prelude,
        n_core,
        n_coda,
        max_len,
        init_std=1.0,
    ):
        super().__init__()

        self.prelude = Prelude(
            vocab_size=vocab_size,
            d=d,
            n_heads=n_heads,
            n_blocks=n_prelude,
            max_len=max_len,
        )
        self.core = Core(
            d=d,
            n_heads=n_heads,
            n_blocks=n_core,
        )
        self.coda = Coda(
            d=d,
            vocab_size=vocab_size,
            n_heads=n_heads,
            n_blocks=n_coda,
        )
        self.init_std = init_std

    def initial_state(self, e, generator=None):
        return torch.randn(
            e.shape,
            device=e.device,
            dtype=e.dtype,
            generator=generator,
        ) * self.init_std

    def forward(
        self,
        idx,
        n_loops,
        s0=None,
        generator=None,
        return_states=False,
    ):
        # Encode the input once through the prelude.
        e = self.prelude(idx)

        # Start from a supplied latent state or sample a fresh one.
        if s0 is None:
            s0 = self.initial_state(e, generator=generator)

        # Recurrently apply the core.
        states = self.core.iterate(
            e,
            s0,
            n_loops,
            return_all=return_states,
        )

        if return_states:
            # With return_all=True, states is a list containing one
            # latent state for each recurrent loop.
            s = states[-1]
            logits = self.coda(s)
            return logits, states

        # With return_all=False, iterate returns the final state directly.
        s = states
        logits = self.coda(s)

        return logits

# Step 11 - sample_loop_count
def sample_loop_count(generator, mean_loops=8, max_loops=32, sigma=0.5):
    # Sample z ~ N(0, 1).
    z = torch.randn(1, generator=generator)

    # Log-normal scale with mean 1.
    scale = torch.exp(sigma * z - (sigma ** 2) / 2)

    # Poisson rate.
    rate = mean_loops * scale

    # Sample Poisson(rate) and add 1.
    n = torch.poisson(rate, generator=generator) + 1

    # Clip to [1, max_loops] and return as a Python int.
    n = torch.clamp(n, min=1, max=max_loops)

    return int(n.item())

# Step 12 - truncated_recurrence
def truncated_recurrence(core, e, s0, n_loops, k_backprop):
    n_tracked = min(k_backprop, n_loops)
    n_warmup = n_loops - n_tracked

    s = s0

    # Run the warm-up iterations without building an autograd graph.
    with torch.no_grad():
        for _ in range(n_warmup):
            s = core(e, s)

    # Detach before starting the tracked portion of the recurrence.
    s = s.detach()

    # Run only the final n_tracked iterations with gradient tracking.
    for _ in range(n_tracked):
        s = core(e, s)

    return s, n_tracked

# Step 13 - masked_lm_loss
def masked_lm_loss(logits, y, mask):
    return F.cross_entropy(logits[mask], y[mask])

# Step 14 - train_step
def train_step(model, opt, batch, n_loops, k_backprop, generator=None):
    x, y, mask = batch

    e = model.prelude(x)
    s0 = model.initial_state(e, generator=generator)

    s, _ = truncated_recurrence(
        model.core,
        e,
        s0,
        n_loops,
        k_backprop,
    )

    logits = model.coda(s)
    loss = masked_lm_loss(logits, y, mask)

    opt.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    opt.step()

    return float(loss.item())

# Step 15 - train_recurrent
def train_recurrent(
    model,
    steps,
    n_digits,
    batch_size=64,
    lr=1e-3,
    seed=0,
    mean_loops=4,
    max_loops=12,
    k_backprop=4,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    generator = torch.Generator().manual_seed(seed)

    losses = []
    loop_counts = []

    for _ in range(steps):
        # Generate the training batch using the shared generator.
        batch = make_addition_batch(
            batch_size,
            n_digits,
            generator=generator,
        )

        # Sample the recurrent loop budget using the same generator.
        n_loops = sample_loop_count(
            generator,
            mean_loops=mean_loops,
            max_loops=max_loops,
        )

        # Perform one optimizer step. The same generator is also used
        # internally for the random initial latent state.
        loss = train_step(
            model,
            opt,
            batch,
            n_loops=n_loops,
            k_backprop=k_backprop,
            generator=generator,
        )

        losses.append(loss)
        loop_counts.append(n_loops)

    return {
        "losses": losses,
        "loop_counts": loop_counts,
    }

# Step 16 - answer_accuracy
def answer_accuracy(model, batch, n_loops, seed=0):
    x, y, mask = batch

    model.eval()

    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        logits = model(
            x,
            n_loops,
            generator=generator,
        )

        predictions = logits.argmax(dim=-1)

    # Correctness at each answer-digit position.
    correct = predictions.eq(y)

    masked_correct = correct[mask]

    # Fraction of all masked answer digits predicted correctly.
    digit_acc = masked_correct.float().mean().item()

    # Each example is correct only if every answer digit is correct.
    sequence_correct = correct.masked_fill(~mask, True).all(dim=1)

    sequence_acc = sequence_correct.float().mean().item()

    return {
        "digit_acc": round(digit_acc, 4),
        "sequence_acc": round(sequence_acc, 4),
    }

# Step 17 - accuracy_vs_loops
def accuracy_vs_loops(model, batch, budgets, seed=0):
    results = {}

    for budget in budgets:
        results[budget] = answer_accuracy(
            model,
            batch,
            n_loops=budget,
            seed=seed,
        )["sequence_acc"]

    return results

# Step 18 - state_convergence
def state_convergence(model, x, n_loops, seed=0):
    model.eval()

    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        _, states = model(
            x,
            n_loops=n_loops,
            generator=generator,
            return_states=True,
        )

    changes = []

    for i in range(1, len(states)):
        diff_norm = torch.linalg.norm(states[i] - states[i - 1])
        state_norm = torch.linalg.norm(states[i])

        relative_change = diff_norm / state_norm
        changes.append(round(float(relative_change.item()), 4))

    return changes

# Step 19 - kl_between_loops
def kl_between_loops(model, x, n_loops, seed=0):
    model.eval()

    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        _, states = model(
            x,
            n_loops=n_loops,
            generator=generator,
            return_states=True,
        )

        # Decode every latent state.
        log_probs = [
            torch.log_softmax(model.coda(state), dim=-1)
            for state in states
        ]

    kl_values = []

    for i in range(1, len(log_probs)):
        prev_log_probs = log_probs[i - 1]
        cur_log_probs = log_probs[i]

        # p_{i-1} = exp(log p_{i-1})
        prev_probs = prev_log_probs.exp()

        # KL(p_{i-1} || p_i), summed over vocabulary and
        # averaged over all positions in the batch.
        kl = (
            prev_probs
            * (prev_log_probs - cur_log_probs)
        ).sum(dim=-1).mean()

        kl_values.append(round(float(kl.item()), 6))

    return kl_values

# Step 20 - adaptive_exit_forward (not yet solved)
# TODO: implement

# Step 21 - greedy_add (not yet solved)
# TODO: implement

