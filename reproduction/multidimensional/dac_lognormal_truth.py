"""Lognormal DAC slices centered at the true mean."""

from reproduction.multidimensional.dac import DacConfig, run_slices

CONFIG = DacConfig(
  experiment_id="multidimensional/dac-lognormal-truth",
  slug="dac-lognormal-truth",
  distribution="lognormal",
  center="truth",
  seed=2024,
  coverage_levels=(0.05, 0.01),
)


def main() -> None:
  run_slices(CONFIG)


if __name__ == "__main__":
  main()
