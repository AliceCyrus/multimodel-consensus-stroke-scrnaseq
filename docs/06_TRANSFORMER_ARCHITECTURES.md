# Chapter 6: Transformer Architectures

## Deep Dive into Foundation Model-Inspired Approaches

---

## 6.1 What are Transformers?

### Origin Story

Transformers were invented in 2017 for language translation. The key paper: **"Attention is All You Need"** (Vaswani et al.)

### The Big Idea: Attention

Instead of processing words one-by-one (like older models), transformers look at ALL words at once and figure out which ones to "pay attention to."

```
Sentence: "The cat sat on the mat because it was tired"

What does "it" refer to?

Attention mechanism: "The cat sat on the mat because it was tired"
                      ───┬───                      ↑
                         └──────────────────────────┘
                         "it" attends strongly to "cat"
```

### Why Use Transformers for Single-Cell Data?

If transformers understand relationships between WORDS...
...maybe they can understand relationships between GENES or PCs?

---

## 6.2 The Original Foundation Models (Brief Overview)

### What Are Foundation Models?

Large models pre-trained on massive datasets, then fine-tuned for specific tasks.

| Model | Pre-training Data | Original Purpose |
|-------|-------------------|------------------|
| **scGPT** | 33M human cells | Cell type annotation, perturbation prediction |
| **scBERT** | 1M cells | Cell type annotation |
| **Geneformer** | 30M cells | Gene network understanding |

### Why We Say "Inspired By"

**Real Foundation Models:**
- Pre-trained on millions of cells
- Process individual genes as tokens
- Use gene-level attention

**Our Implementation:**
- Trained from scratch on our data
- Process PCA components as input
- Use transformer architecture but differently

**Honest Statement:** Our models use transformer architectures inspired by these foundation models, but are not identical implementations.

---

## 6.3 Our scGPT-Inspired Model

### Architecture

```
INPUT: 50-dim PCA vector
       ↓
┌────────────────────────────────────────────┐
│  LINEAR PROJECTION                          │
│  50 → 256 dimensions                        │
│  (input_projection)                         │
└────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────┐
│  ADD POSITIONAL ENCODING                    │
│  + learnable position parameter             │
└────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────┐
│  TRANSFORMER ENCODER (4 layers)             │
│  ┌──────────────────────────────────────┐  │
│  │  Multi-Head Attention (8 heads)       │  │
│  │  Feed-Forward Network (1024 hidden)   │  │
│  │  LayerNorm + Dropout (0.1)            │  │
│  └──────────────────────────────────────┘  │
│  (repeated 4 times)                         │
└────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────┐
│  CLASSIFICATION HEAD                        │
│  LayerNorm → Linear(256→128) → GELU        │
│  → Dropout → Linear(128→2)                 │
└────────────────────────────────────────────┘
       ↓
OUTPUT: [P(Control), P(Stroke)]
```

### Code Implementation

```python
class scGPTModel(nn.Module):
    def __init__(self, input_dim=50, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        # Project 50 PCs to 256 dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=8, 
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        # x: [batch, 50]
        x = self.input_projection(x)     # [batch, 256]
        x = x.unsqueeze(1)               # [batch, 1, 256] - single token
        x = x + self.pos_encoding        # Add position
        x = self.transformer(x)          # [batch, 1, 256]
        x = x.squeeze(1)                 # [batch, 256]
        return self.classifier(x)        # [batch, 2]
```

### Honest Assessment

**What This Model Is:**
- A deep neural network with transformer layers
- Uses LayerNorm, GELU, residual connections

**What This Model Is NOT:**
- Multi-token attention (we have only 1 token)
- Pre-trained on millions of cells

**Why It Still Works:**
- Deep non-linear transformations
- Strong regularization
- Good architecture design

---

## 6.4 Our Geneformer-Inspired Model

### What Makes Geneformer Special

Real Geneformer uses **rank-based tokenization**:
- Genes ordered by expression level
- Rank determines position, not absolute value

### Our Adaptation

```
INPUT: 50-dim PCA vector
       ↓
┌────────────────────────────────────────────┐
│  COMPUTE RANKS                              │
│  For each cell, rank PCs from lowest to    │
│  highest value (normalized to 0-1)         │
└────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────┐
│  DUAL EMBEDDING                             │
│  value_embedding: actual PC value → 256-dim│
│  rank_embedding: PC rank → 256-dim         │
│  Concatenate: 256 + 256 = 512-dim          │
└────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────┐
│  ADD LEARNED POSITION EMBEDDINGS            │
│  gene_positions: learnable [50, 512]       │
│  Each PC has its own position embedding    │
└────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────┐
│  TRANSFORMER (4 layers)                     │
│  Now processes 50 TOKENS (one per PC!)     │
│  Attention between PCs possible            │
└────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────┐
│  MEAN POOLING                               │
│  Average across 50 token positions         │
└────────────────────────────────────────────┘
       ↓
OUTPUT: Classification
```

### Code Implementation

```python
class GeneformerModel(nn.Module):
    def __init__(self, input_dim=50, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        
        # Dual embedding (like real Geneformer's expression + rank)
        self.value_embedding = nn.Linear(1, d_model // 2)   # 256 dim
        self.rank_embedding = nn.Linear(1, d_model // 2)    # 256 dim
        
        # Learned position embeddings (one per PC)
        self.gene_positions = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        # x: [batch, 50]
        
        # Compute ranks (key Geneformer concept!)
        ranks = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
        ranks = ranks / x.size(1)  # Normalize to [0, 1]
        
        # Dual embedding
        value_emb = self.value_embedding(x.unsqueeze(-1))     # [batch, 50, 256]
        rank_emb = self.rank_embedding(ranks.unsqueeze(-1))   # [batch, 50, 256]
        
        x = torch.cat([value_emb, rank_emb], dim=-1)  # [batch, 50, 512]
        x = x + self.gene_positions  # Add learned positions
        
        # Transformer (processes 50 tokens!)
        x = self.transformer(x)  # [batch, 50, 512]
        
        # Mean pooling
        x = x.mean(dim=1)  # [batch, 512]
        
        return self.classifier(x)
```

### Why Geneformer is Different

| Feature | scGPT-like | Geneformer-like |
|---------|------------|-----------------|
| Tokens | 1 (whole cell) | 50 (one per PC) |
| Attention | Trivial | Between PCs |
| Uses ranks | No | Yes |
| Position embeddings | Simple | Learned per PC |

---

## 6.5 Attention Explained

### What is Self-Attention?

For each token, compute:
1. **Query (Q):** What am I looking for?
2. **Key (K):** What do I contain?
3. **Value (V):** What do I contribute?

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

### Multi-Head Attention

```
Instead of one attention:
HEAD 1: Focuses on one pattern
HEAD 2: Focuses on another pattern
...
HEAD 8: Focuses on yet another pattern

Then combine all heads.
```

### In Our Geneformer Model

With 50 PC tokens, attention can learn:
- "When PC1 is high, look at PC15"
- "PC3 and PC22 are related"
- Patterns across PCs

---

## 6.6 Training Configuration

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Batch size | 128 | Samples per update |
| Learning rate | 2e-4 | Step size |
| Optimizer | AdamW | Weight-decayed Adam |
| Weight decay | 0.01 | L2 regularization |
| Epochs | 30 | Training iterations |
| Early stopping | 5 | Stop if no improvement |
| Dropout | 0.1-0.4 | Prevent overfit |
| Gradient clip | 1.0 | Prevent exploding gradients |

### Learning Rate Schedule

```
Learning Rate over Training:

LR     |\ 
2e-4 --|--\
       |   \
       |    \____
       |         \____
2e-5 --|              \____
       └────────────────────→ Epochs
       0    10   20   30
       
Cosine annealing: starts high, gradually decreases
```

### Mixed Precision Training

```python
# Use 16-bit floats for speed
with torch.cuda.amp.autocast():
    logits = model(batch_X)
    loss = criterion(logits, batch_y)
```

---

## 6.7 Key Takeaways

### What Our Transformers Provide

Even without full foundation model capabilities:

1. **Deep representations** - 4-6 layers of transformation
2. **Non-linearity** - GELU activations
3. **Regularization** - Dropout, LayerNorm, weight decay
4. **Gradient flow** - Residual connections
5. **Architectural diversity** - Different from tree-based methods

### What They Don't Provide

1. Pre-trained knowledge from millions of cells
2. Gene-level tokenization
3. True cross-gene attention (except Geneformer variant)

### Why Use Them Anyway?

**For consensus methodology:**
- They provide a different "lens" on the data
- Agreement across diverse architectures increases confidence
- Even if not optimal, they add to model diversity

---

## Navigation

← [Previous: ML Models](./05_MACHINE_LEARNING_MODELS.md) | [Return to Index](./00_README.md) | [Next: Gene Importance →](./07_GENE_IMPORTANCE_EXTRACTION.md)
