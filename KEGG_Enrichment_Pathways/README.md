# KEGG pathway analysis — negative result

**No KEGG pathway is significantly enriched in the 63-gene consensus signature
after multiple-testing correction.** This folder documents that negative
result rather than hiding it.

Regenerate with:

```bash
Rscript keggPathways.R
```

## Result

| | |
|---|---|
| Input | 63 consensus genes (39 mapped to KEGG) |
| Background | 3,000 highly variable genes (1,431 mapped to KEGG) |
| Pathways tested | 122 |
| Nominal *p* < 0.05 | 21 |
| **FDR < 0.05** | **0** |
| Smallest adjusted *p* | 0.162 |

## Files

| File | Contents |
|---|---|
| `KEGG_All_Tested_Pathways.csv` | All 122 pathways with raw and adjusted p-values |
| `KEGG_Nominal_P05_NotFDRSignificant.csv` | The 21 reaching nominal *p* < 0.05 — **none FDR-significant** |
| `KEGG_Session_Info.txt` | R and package versions for the run |

Both CSVs carry an `MHCII_only` column, explained below.

There is deliberately **no** `KEGG_Enrichment_Results.csv`. That filename is
reserved for FDR-significant pathways, and the script writes it only if any
exist. Its absence is the result.

## Why nothing is significant

**1. The background is correct.** Consensus genes were selected by
back-projection through PCA loadings, which are defined only over the 3,000
highly variable genes. A gene outside that set had zero probability of being
selected, so the HVG set is the correct universe. That pool is already
dominated by immune transcripts, because immune activation is what varies most
after stroke. Against an honest background, the consensus is no longer
unusually immune *at the pathway level*.

Using the whole genome (~11,000 KEGG-annotated genes) instead would return
roughly 23 pathways — but it measures the HVG selection step, not the models.
See Huang et al. 2009 (*NAR*) and Timmons et al. 2015 (*Genome Biology*).

**2. MHC class II saturation.** Ten of the 21 nominally enriched pathways are
driven *entirely* by the same three genes — `H2-Aa`, `H2-Ab1`, `H2-Eb1`:

```
Asthma                            H2-Eb1/H2-Aa/H2-Ab1
Autoimmune thyroid disease        H2-Eb1/H2-Aa/H2-Ab1
Allograft rejection               H2-Eb1/H2-Aa/H2-Ab1
Graft-versus-host disease         H2-Eb1/H2-Aa/H2-Ab1
Type I diabetes mellitus          H2-Eb1/H2-Aa/H2-Ab1
...
```

These genes are annotated to dozens of unrelated KEGG disease maps, so any
list containing them produces a long tail of diseases that has nothing to do
with stroke. This is a property of KEGG's annotation structure, not a finding.

Gene Ontology does not have this problem here: its leading terms are supported
by 10, 16 and 25 genes rather than the same 3 repeated. GO is therefore
reported as the primary enrichment analysis, and the MHC class II signal is
captured there through `GO:0019886` and `GO:0042611`.

## Note on earlier versions

Prior to this correction, this folder held a `KEGG_Enrichment_Results.csv`
listing 28 pathways, plus dot/bar/heat plots and five `pathview` diagrams
(`mmu04612`, `mmu04640`, `mmu04672`, `mmu05310`, `mmu05323`). Those outputs
came from a run that used a 39-gene list and the whole-genome background, and
have been removed because neither matches the analysis reported in the
manuscript. The GO analysis was always run on the correct 63 genes.
