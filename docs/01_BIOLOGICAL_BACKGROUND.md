# Chapter 1: Biological Background

## Understanding Ischemic Stroke

---

## 1.1 What is Ischemic Stroke?

### Simple Explanation
Imagine your brain is a city, and blood vessels are the roads that deliver oxygen (like electricity) to every building. An **ischemic stroke** happens when one of these roads gets blocked—usually by a blood clot. The buildings (brain cells) that don't get oxygen start to die.

### Technical Definition
Ischemic stroke is the sudden loss of blood circulation to a brain region due to arterial occlusion, resulting in corresponding loss of neurologic function. It accounts for ~87% of all strokes.

### Key Statistics
- 15 million people worldwide suffer strokes annually
- 5 million die, 5 million are permanently disabled
- Stroke is the #2 cause of death globally
- Every 40 seconds, someone in the US has a stroke

---

## 1.2 What Happens Inside the Brain During Stroke?

### The Ischemic Cascade (Step by Step)

```
NORMAL BRAIN                    DURING STROKE
============                    =============
                                
Blood flows freely              Blood flow blocked
     ↓                               ↓
Oxygen delivered                No oxygen (hypoxia)
     ↓                               ↓
ATP produced                    ATP depleted
     ↓                               ↓
Cells function normally         Ion pumps fail
                                     ↓
                                Calcium floods in
                                     ↓
                                Glutamate released
                                     ↓
                                Excitotoxicity
                                     ↓
                                Inflammation begins
                                     ↓
                                Cell death (necrosis/apoptosis)
```

### The Two Zones

When a stroke happens, there are two important areas:

1. **Ischemic Core**
   - Center of the blocked area
   - Cells die within minutes
   - Usually cannot be saved

2. **Penumbra** (The "Twilight Zone")
   - Surrounding area with reduced blood flow
   - Cells are stressed but not dead yet
   - THIS IS WHERE WE CAN INTERVENE
   - Can be saved if treated quickly (hours)

---

## 1.3 The Immune Response to Stroke

### Why This Matters for Our Study
After stroke, the immune system activates. This is a double-edged sword:
- **Good:** Clears dead cells, initiates repair
- **Bad:** Can cause additional damage (inflammation)

### The Players (Cell Types)

| Cell Type | Role in Stroke | Our Data |
|-----------|----------------|----------|
| **Microglia** | Brain's resident immune cells; first responders | ✅ Present |
| **Macrophages** | Infiltrate from blood; eat dead cells | ✅ Present |
| **Neutrophils** | First blood cells to arrive; can cause damage | ✅ Present |
| **Astrocytes** | Support cells; form scar tissue | ✅ Present |
| **Neurons** | The dying cells we're trying to save | ✅ Present |

### Key Pathways in Stroke

1. **IL-1 Pathway (Interleukin-1)**
   - Pro-inflammatory
   - IL-1β drives inflammation
   - IL-1rn (receptor antagonist) blocks it
   - **Drug target:** Anakinra (FDA-approved for other conditions)

2. **Interferon Response**
   - Type I IFN (IFN-α, IFN-β): Usually antiviral
   - Type II IFN (IFN-γ): Immune activation
   - **Our finding:** Both types activated in stroke

3. **NF-κB Pathway**
   - Master regulator of inflammation
   - Activated by IL-1 and other signals
   - Downstream of many stroke-related genes

---

## 1.4 The MCAO Mouse Model

### What is MCAO?

**MCAO = Middle Cerebral Artery Occlusion**

This is the "gold standard" animal model for studying ischemic stroke.

### How It Works

```
NORMAL MOUSE BRAIN              MCAO SURGERY
==================              ============

Middle Cerebral                 A thread is inserted
Artery (MCA) →                 through the carotid
supplies most of     →         artery and pushed
the cortex                     up to block the MCA
                               
                                     ↓
                               
                               Blood flow stops
                               to MCA territory
                               
                                     ↓
                               
                               STROKE!
                               (Similar to human stroke)
```

### Why We Use This Model

| Advantage | Explanation |
|-----------|-------------|
| Reproducible | Same lesion location every time |
| Clinically relevant | Mimics human stroke pathophysiology |
| Well-characterized | Decades of literature |
| Genetic tools | Can use transgenic mice |

### Our Specific Setup

| Parameter | Value |
|-----------|-------|
| Mouse strain | C57BL/6J |
| Model | Transient MCAO (60-90 min occlusion) |
| Timepoint | 24 hours post-stroke |
| Samples | 3 MCAO (stroke), 3 Sham (control surgery) |
| Analysis | Single-cell RNA sequencing |

### Sham Surgery (Control)
- Same surgical procedure
- Thread inserted but NOT blocking the artery
- No stroke occurs
- Controls for surgical stress

---

## 1.5 Why Study Gene Expression?

### The Central Dogma

```
DNA  →  RNA  →  Protein  →  Function
        ↑
    We measure
    this (mRNA)
```

### What Changes During Stroke?

When cells experience stroke:
1. They activate **survival programs** (try to survive)
2. They activate **inflammation programs** (call for help)
3. They activate **death programs** (if damage too severe)

Each of these involves turning genes ON or OFF.

### RNA as a Biomarker

By measuring which genes are turned ON in stroke cells vs. control cells, we can:
1. **Understand** what biological processes are happening
2. **Identify** potential drug targets
3. **Develop** diagnostic tests

---

## 1.6 Key Terms Glossary

| Term | Simple Definition |
|------|-------------------|
| **Ischemia** | Lack of blood flow |
| **Hypoxia** | Lack of oxygen |
| **Penumbra** | "Almost dead" zone around stroke core |
| **Excitotoxicity** | Cell death from too much glutamate |
| **Microglia** | Brain's immune cells |
| **Cytokine** | Signaling molecule (like text messages between cells) |
| **IL-1** | Inflammatory cytokine family |
| **Interferon** | Antiviral/immune cytokine family |
| **MCAO** | Middle Cerebral Artery Occlusion (stroke model) |
| **Sham** | Control surgery (no stroke) |

---

## 1.7 What Our Study Adds

### The Gap We Fill

**Previous work:** Many studies identified stroke-related genes using:
- Bulk RNA-seq (averages all cells together)
- Single ML model (findings might be model-specific)

**Our innovation:**
- Single-cell resolution (see individual cell types)
- 8 different ML models (findings are robust if all agree)
- Consensus approach (only trust genes multiple methods find)

### Our Hypothesis

> If a gene is identified as important by multiple independent machine learning algorithms with different mathematical foundations, it is more likely to be a genuine stroke biomarker than a statistical artifact.

---

## Navigation

← [Return to Index](./00_README.md) | [Next: Single-Cell RNA-Seq →](./02_SINGLE_CELL_RNA_SEQ.md)
