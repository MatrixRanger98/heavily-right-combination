"""Normal divide-and-combine figures and numerical diagnostics."""

from reproduction.multidimensional.dac import DacConfig, run_slices

CONFIG = DacConfig(
  experiment_id="multidimensional/dac-normal",
  slug="dac-normal",
  distribution="normal",
  center="estimate",
  seed=2025,
)


def main() -> None:
  run_slices(CONFIG)


if __name__ == "__main__":
  main()
