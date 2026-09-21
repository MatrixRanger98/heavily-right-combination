"""Lognormal DAC estimate-centered numerical diagnostics."""

from reproduction.multidimensional.dac import DacConfig, run_diagnostics

CONFIG = DacConfig(
  experiment_id="multidimensional/dac-lognormal-estimate-diagnostics",
  slug="dac-lognormal-estimate",
  distribution="lognormal",
  center="estimate",
  seed=2024,
)


def main() -> None:
  run_diagnostics(CONFIG)


if __name__ == "__main__":
  main()
