"""Normal DAC direction-width and coverage diagnostics."""

from reproduction.multidimensional.dac import DacConfig, run_diagnostics

CONFIG = DacConfig(
  experiment_id="multidimensional/dac-normal-diagnostics",
  slug="dac-normal",
  distribution="normal",
  center="estimate",
  seed=2025,
)


def main() -> None:
  run_diagnostics(CONFIG)


if __name__ == "__main__":
  main()
