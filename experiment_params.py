# MARIO forecasting experiment grid.
#
# Each dict in `datasets_to_run` defines the parameter GRID for one dataset. The
# orchestrator (run_experiment.py) expands a Cartesian product per stage:
#   Stage 1 abstraction  : d_method x b x ig
#   Stage 2 mining        : mvs x mg x rel x sf x e
#   Stage 5 aggregation   : aggregation_method x context_window x warmup
# plus per-dataset scalars: data_path, horizon (t+horizon forecast lead / Stage 0
# embargo) and target_variable (TemporalPropertyID whose future symbol is forecast).
#
# Only these keys are read. FCPM-era keys (event_symbol, split_event_class,
# event_window, num_tirps_for_selection, TTE_W, e_w, Tirp_selection_methods) are
# gone — MARIO has no classes / event of interest.

datasets_to_run = [
    {
        'dataset_name': 'diabetes',
        # Absolute path, or relative to where the orchestrator is launched.
        'data_path': "/mnt/new_groups/robertmo_group/Tali/CPM_Framework_2/diabetes/hemoglobin_data.csv",
        # Forecast lead time (t + horizon); also the Stage 0 embargo size.
        # A LIST runs one independent experiment per horizon, each with its own output
        # tree named '<dataset_name>_h<horizon>' (e.g. diabetes_h1, diabetes_h5,
        # diabetes_h10). A scalar (e.g. 5) runs a single experiment and keeps the plain
        # dataset name. Launch a single horizon on its own with:
        #   sbatch run_experiment.slurm --datasets diabetes_h5
        # or all horizons of the dataset with:  --datasets diabetes
        'horizon': [5, 10],
        'target_variable': 39,  # TemporalPropertyID whose future abstracted state is forecast

        # Quick-run entity subsampling: set to an int N to run on only the first N
        # entities (sorted, deterministic), applied in Stage 0 BEFORE the split so
        # train/test/manifest + all downstream stages stay consistent. None = full
        # dataset. Falls back to config.SUBSAMPLE_ENTITIES when this key is absent.
        'subsample_entities': None,

        # --- Stage 1: abstraction (unsupervised methods only in MARIO) ---
        "d_method": ['equal_frequency'],
        "b": [5],   # number of bins
        "ig": [1],     # interpolation gap (passed as max_gap to Hugobot)

        # --- Stage 2: TIRP mining ---
        "mvs": [0.1],   # minimum vertical support
        "mg": [10, 20],     # max_gap
        "rel": [7],     # number of Allen relations
        "sf": [False],   # skip_followers
        "e": [0],       # epsilon

        # --- Stage 5: cross-TIRP aggregation grid ---
        "aggregation_method": ['average'],  # how active TIRPs' distributions are combined
        "context_window": [0],                  # a TIRP is active at t if its last row is in [t-C, t]
        "warmup": [0],                             # per-entity warm-up before new_entity rows are scored
    },

    # --- Template for an additional dataset (uncomment and fill in) ---
    # {
    #     'dataset_name': 'icu',
    #     'data_path': "/mnt/new_groups/robertmo_group/Eldar/CPM_dataset/icu/icu_with_outcome_paa_10.csv",
    #     'horizon': 5,
    #     'target_variable': 1,
    #     "d_method": ['equal_frequency'],
    #     "b": [3, 5], "ig": [1],
    #     "mvs": [0.3], "mg": [120], "rel": [7], "sf": [True], "e": [0],
    #     "aggregation_method": ['average'], "context_window": [0], "warmup": [0],
    # },
]
