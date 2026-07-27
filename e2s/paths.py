"""Project paths configuration.

All paths are resolved relative to the project root, making scripts runnable
from any working directory. Add a @property for each new data file introduced
in the pipeline.
"""

from pathlib import Path


class ProjPaths:
    """Centralized project paths.

    The root is inferred from the location of this file (e2s/), so scripts
    run correctly regardless of the working directory they are invoked from.
    """

    def __init__(self):
        self._pkg_path = Path(__file__).resolve().parent  # e2s/
        self._project_path = self._pkg_path.parent  # project root

    # ------------------------------------------------------------------ #
    # Top-level directories                                              #
    # ------------------------------------------------------------------ #

    @property
    def project_path(self) -> Path:
        """Root project directory."""
        return self._project_path

    @property
    def pkg_path(self) -> Path:
        """Source package directory (e2s/)."""
        return self._pkg_path

    @property
    def pipeline_path(self) -> Path:
        """Pipeline scripts directory."""
        return self._project_path / "pipeline"

    @property
    def data_path(self) -> Path:
        """Main data directory (git-ignored, DVC-cached)."""
        return self._project_path / "data"

    @property
    def output_path(self) -> Path:
        """Generated outputs root (tracked in git)."""
        return self._project_path / "output"

    # ------------------------------------------------------------------ #
    # Experiment 1: ensemble forecast + validation                      #
    # ------------------------------------------------------------------ #

    @property
    def ensemble_data_path(self) -> Path:
        return self.data_path / "ensemble"

    @property
    def ensemble_zarr_path(self) -> Path:
        """FCN3 ensemble forecast, written by pipeline/ensemble/01_run.py."""
        return self.ensemble_data_path / "fcn3_ensemble.zarr"

    @property
    def ensemble_output_path(self) -> Path:
        return self.output_path / "ensemble"

    @property
    def ensemble_validation_path(self) -> Path:
        return self.ensemble_output_path / "validation"

    @property
    def ensemble_validation_tables_path(self) -> Path:
        return self.ensemble_validation_path / "tables"

    @property
    def ensemble_analysis_path(self) -> Path:
        return self.ensemble_output_path / "analysis"

    @property
    def ensemble_book_path(self) -> Path:
        """Assets rendered specifically for the book chapters (e.g. the
        spaghetti chart, the one "hero" gif) - kept separate from
        ensemble_analysis_path so DVC's book-notebook stage can own this
        directory as its own output without nesting inside
        ensemble_analyse's already-declared output/ensemble/analysis."""
        return self.ensemble_output_path / "book"

    @property
    def ensemble_gifs_path(self) -> Path:
        """Per-member Robinson-projection animations - regenerable and too
        large for git, so these live under data/ (DVC-cached) rather than
        output/ (git-tracked)."""
        return self.ensemble_data_path / "gifs"

    # ------------------------------------------------------------------ #
    # Experiment 2: temporal downscaling (6h -> 1h via InterpModAFNO)    #
    # ------------------------------------------------------------------ #

    @property
    def downscaling_data_path(self) -> Path:
        return self.data_path / "downscaling"

    @property
    def downscaling_zarr_path(self) -> Path:
        return self.downscaling_data_path / "sfno_downscaled.zarr"

    @property
    def downscaling_output_path(self) -> Path:
        return self.output_path / "downscaling"

    @property
    def downscaling_analysis_path(self) -> Path:
        return self.downscaling_output_path / "analysis"

    @property
    def downscaling_book_path(self) -> Path:
        """Assets rendered specifically for the book chapter - kept separate
        from downscaling_analysis_path for the same reason as
        ensemble_book_path; see that property's docstring."""
        return self.downscaling_output_path / "book"

    @property
    def downscaling_gifs_path(self) -> Path:
        """Per-member Robinson-projection animations - regenerable and too
        large for git, so these live under data/ (DVC-cached) rather than
        output/ (git-tracked); mirrors ensemble_gifs_path."""
        return self.downscaling_data_path / "gifs"

    # ------------------------------------------------------------------ #
    # Experiment 3: per-step variable subset write behavior              #
    # ------------------------------------------------------------------ #

    @property
    def variable_subset_data_path(self) -> Path:
        return self.data_path / "variable_subset"

    @property
    def variable_subset_output_path(self) -> Path:
        return self.output_path / "variable_subset"

    # ------------------------------------------------------------------ #
    # Experiment 4: diagnostic model (precipitation / solar radiation)  #
    # ------------------------------------------------------------------ #

    @property
    def diagnostic_data_path(self) -> Path:
        return self.data_path / "diagnostic"

    @property
    def diagnostic_zarr_path(self) -> Path:
        return self.diagnostic_data_path / "fcn3_diagnostic.zarr"

    @property
    def diagnostic_output_path(self) -> Path:
        return self.output_path / "diagnostic"

    # ------------------------------------------------------------------ #
    # Shared: Germany population-weighted regional statistics           #
    # ------------------------------------------------------------------ #

    @property
    def germany_data_path(self) -> Path:
        return self.data_path / "germany"

    @property
    def germany_population_mask_raw_path(self) -> Path:
        """Raw 'weights and masks' download from CDS (dataset
        sis-energy-pecd), or the fake stand-in - see
        pipeline/germany/01_run.py. Kept separate from the processed mask
        so re-running pipeline/germany/02_build_germany_mask.py after a
        crop/regrid bugfix doesn't require re-fetching this."""
        return self.germany_data_path / "pecd_population_mask_raw.nc"

    @property
    def germany_population_mask_path(self) -> Path:
        """Germany-cropped population weights, regridded onto the
        FCN3/SFNO native 0.25-deg grid - built by
        pipeline/germany/02_build_germany_mask.py. Used by both
        pipeline/ensemble/04_analyse.py and pipeline/downscaling/02_analyse.py
        to compare a plain area-weighted vs. population-weighted Germany
        mean temperature."""
        return self.germany_data_path / "germany_population_mask.nc"

    @property
    def germany_output_path(self) -> Path:
        return self.output_path / "germany"

    @property
    def germany_book_path(self) -> Path:
        """Assets rendered specifically for the population-weighted-mean
        book chapter (pipeline/germany/03_population_weighted_overview.py)
        - mirrors ensemble_book_path/downscaling_book_path so the book
        stage owns this directory as its own DVC output."""
        return self.germany_output_path / "book"

    # ------------------------------------------------------------------ #
    # Experiment 5: perturbation strategy comparison                    #
    # ------------------------------------------------------------------ #

    @property
    def perturbation_data_path(self) -> Path:
        return self.data_path / "perturbation"

    def perturbation_zarr_path(self, config_name: str) -> Path:
        """One zarr store per perturbation strategy compared in
        pipeline/perturbation/01_run.py (e.g. "zero", "gaussian",
        "bred_vector", "per_step_brown") - a method rather than a
        @property like everything else here, since the number of configs
        is open-ended rather than one fixed path per experiment."""
        return self.perturbation_data_path / f"{config_name}.zarr"

    @property
    def perturbation_output_path(self) -> Path:
        return self.output_path / "perturbation"

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def ensure_directories(self) -> None:
        """Create all standard directories if they do not yet exist."""
        dirs = [
            self.ensemble_data_path,
            self.ensemble_validation_tables_path,
            self.ensemble_analysis_path,
            self.ensemble_book_path,
            self.ensemble_gifs_path,
            self.downscaling_data_path,
            self.downscaling_output_path,
            self.downscaling_analysis_path,
            self.downscaling_book_path,
            self.downscaling_gifs_path,
            self.variable_subset_data_path,
            self.variable_subset_output_path,
            self.diagnostic_data_path,
            self.diagnostic_output_path,
            self.germany_data_path,
            self.germany_output_path,
            self.germany_book_path,
            self.perturbation_data_path,
            self.perturbation_output_path,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
