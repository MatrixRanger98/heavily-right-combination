"""Lognormal DAC slices centered at the projected estimate."""

from reproduction.multidimensional.dac import DacConfig, run_slices

CONFIG = DacConfig(
  experiment_id="multidimensional/dac-lognormal-estimate",
  slug="dac-lognormal-estimate",
  distribution="lognormal",
  center="estimate",
  seed=2024,
)


def main() -> None:
  run_slices(CONFIG)


if __name__ == "__main__":
  main()
