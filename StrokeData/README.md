# StrokeData

The raw 10x Genomics matrices go here. They are **not** committed — roughly
251 MB, and freely available from NCBI GEO.

Each of the six sample folders must end up containing exactly three files, with
these exact names (`Read10X()` requires them literally):

```
barcodes.tsv.gz
features.tsv.gz
matrix.mtx.gz
```

To populate this directory:

```bash
bash data/download_GSE174574.sh
```

then follow the mapping instructions the script prints.

Full guidance, including the per-sample cell counts to verify against, is in
[`../data/README.md`](../data/README.md).

**Source:** GEO accession
[GSE174574](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174574) —
mouse tMCAO model, 6 animals.

> `sham3` and `mcao3` are the held-out test set. Assigning GEO samples to the
> wrong folders will change your results relative to the paper.
