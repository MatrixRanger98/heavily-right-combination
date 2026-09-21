# Additional comparison curves

`exploratory/bayes-factor-comparisons` generates four Bayes-factor comparison
panels. They are included as additional results, marked **Unused/NA** in the
[results index](../../results/README.md).

From the repository root:

```bash
python -m reproduction.run exploratory/bayes-factor-comparisons \
  --profile paper --run-id my-bayes-comparison --results-root results/runs
```

The output PDFs are
`bayes-factor-comparisons-{beta-one,goodman,edwards,edwards-two}.pdf`.
Both profiles evaluate the same deterministic curves.
