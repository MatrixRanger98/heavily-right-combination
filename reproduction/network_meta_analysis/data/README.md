# Senn2013 comparison data

`senn2013.csv` is a plain-text snapshot of the comparison-based `Senn2013`
data distributed with R package `netmeta` 3.6-1. The package documentation
describes the data as preprocessed for the example in:

> Senn S, Gavini F, Magrez D, Scheen A (2013). Issues in performing a network
> meta-analysis. *Statistical Methods in Medical Research*, 22, 169–189.

The snapshot retains all 28 rows and all seven package fields. It lets the
paper experiments run without R or an installed copy of `netmeta`.

The three `Willms1999` contrasts form a multi-arm trial. The Python loader in
`../senn2013.py` applies the corresponding variance adjustment before fitting
the common-effect weighted least-squares model. Tests compare its resulting
interval-width matrix with the portable netmeta reference in
`tests/fixtures/nma_wls_widths.json` at the repository root. Neither R nor
PyArrow is required for Python computation or this parity test.

CSV SHA-256: `91105017c5c04deba854b7362886ca87d9e91796ba64d0c96191f3c81517bf57`.
`TE` is treat1 minus treat2; `seTE` is its standard error. HCCT's comparison-level
reproduction explicitly uses unadjusted reported standard errors and the
documented study-set policy; this is not a joint multi-arm covariance model.
See [THIRD_PARTY.md](../../../THIRD_PARTY.md) for source attribution and terms.
