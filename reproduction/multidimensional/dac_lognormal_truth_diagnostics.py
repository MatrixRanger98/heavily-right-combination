"""Lognormal DAC truth-centered numerical diagnostics."""

from reproduction.multidimensional.dac import DacConfig, run_diagnostics

CONFIG = DacConfig(
  experiment_id="multidimensional/dac-lognormal-truth-diagnostics",
  slug="dac-lognormal-truth",
  distribution="lognormal",
  center="truth",
  seed=2024,
  coverage_levels=(0.05, 0.01),
)


def main() -> None:
  run_diagnostics(CONFIG)


if __name__ == "__main__":
  main()
