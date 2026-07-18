from config import MAX_TIRPS_FOR_SELECTION


# Define the list of datasets and their specific parameters for the experiment
# Ensure parameter keys match those expected by the main script and wrappers
# (e.g., 'b' for num_bins, 'ig' for interpolation_gap, 'mg' for max_gap etc.)

datasets_to_run = [
    #     {
    #     'dataset_name': 'falls_trigger',
    #     # Use absolute paths or paths relative to where the orchestration script is run
    #     'data_path': "/mnt/new_groups/robertmo_group/Niv/projects/CPM_model_TVS_exp/Preprocessing/PAA7/falls_triger_top15.csv",
    #     'event_symbol': 999, # Example event symbol

    #     # Examples: "tide", "tide_ks", "tide_kl_c1longer", "tide_mw_c0longer"
    #     "d_method": (
    #         ['equal_frequency','equal_width','sax','td4c','tid3','tid3_c0longer','tid3_c1longer','mdlp']
    #     ), # existing TID3 duration-preference variants + 10 seeded random-ordering runs for ablation
    #     "b": [2,3,4,5], # num_of_bins used in submit_stage1_job
    #     "ig": [7], # interpolation_gap used in submit_stage1_job

    #     # --- New parameters for Supervised Abstraction ---
    #     # split_event_class: Determines how to split the data for abstraction.
    #     #   - False: Standard split (classify entity as a whole).
    #     #   - True: Event-based split (data near event = Class 1, rest = Class 0).
    #     "split_event_class": [False],

    #     # event_window: Defines the window size around the event.
    #     #   - Continuous (0-1): Percentage of entity length.
    #     #   - Integer (>1): Number of time units before the event.
    #     #   - Only relevant if split_event_class is True and method is Supervised.
    #     "event_window": [10],

    #     # Stage 2: Mining parameters
    #     "mvs": [0.2],
    #     "mg": [10], # max_gap used in submit_stage2_job
    #     "rel": [3,7], # num_relations used in submit_stage3_job
    #     "sf": [True], # skip_followers used in submit_stage2_job
    #     "e": [0], # epsilon used in submit_stage2_job / submit_stage3_job

    #     # Stage 4: Aggregation/Evaluation parameters
    #     "aggregation_method": ['avg'],
    #     "num_tirps_for_selection": MAX_TIRPS_FOR_SELECTION, # Number of TIRPs to consider in Stage 2 Tirp_selection
    #     "TTE_W": [10000], # TTE_window_size used in submit_stage4_job
    #     "e_w": [0,1,2,3,4,5] # early_warning_value used in submit_stage4_job
    # },
    
    {
        'dataset_name': 'diabetes',
        # Use absolute paths or paths relative to where the orchestration script is run
        'data_path': "/mnt/new_groups/robertmo_group/Eldar/CPM_dataset/diabetes/diabetes_with_outcome.csv",
        'event_symbol': 999, # Example event symbol
        'horizon': 5, # MARIO forecast horizon (t + horizon); also the Stage 0 embargo size.
        'target_variable': 1, # MARIO: TemporalPropertyID whose future abstracted state is forecast.

        "d_method": (
            ['equal_width','equal_frequency']
        ), # existing TID3 duration-preference variants + 10 seeded random-ordering runs for ablation

        "b": [2,5], # num_of_bins used in submit_stage1_job
        "ig": [1], # interpolation_gap used in submit_stage1_job

        # Stage 2: Mining parameters
        "mvs": [0.1],
        "mg": [12], # max_gap used in submit_stage2_job
        "rel": [7], # num_relations used in submit_stage3_job
        "sf": [True], # skip_followers used in submit_stage2_job
        "e": [0], # epsilon used in submit_stage2_job / submit_stage3_job

        # Stage 5: MARIO cross-TIRP aggregation hyperparameter grid (Cartesian product).
        "aggregation_method": ['average','max'],  # how active TIRPs' distributions are combined
        "context_window": [0, 5],                 # a TIRP is active at t if its last row is in [t-C, t]
        "warmup": [0],                            # per-entity warm-up before new_entity rows are scored
    },
    # {
    #     'dataset_name': 'icu',
    #     # Use absolute paths or paths relative to where the orchestration script is run
    #     'data_path': "/mnt/new_groups/robertmo_group/Eldar/CPM_dataset/icu/icu_with_outcome_paa_10.csv",
    #     'event_symbol': 999, # Example event symbol
    
    #     # Examples: "tide", "tide_ks", "tide_kl_c1longer", "tide_mw_c0longer"
    #     "d_method": (
    #         ['tid3_20','tid3_c1longer_20','tid3_c0longer_20','tid3_150','tid3_c1longer_150','tid3_c0longer_150']
    #     ), # existing TID3 duration-preference variants + 10 seeded random-ordering runs for ablation
    
    #     "b": [2,3,4,5], # num_of_bins used in submit_stage1_job
    #     "ig": [1], # interpolation_gap used in submit_stage1_job
    
    #     # --- New parameters for Supervised Abstraction ---
    #     # split_event_class: Determines how to split the data for abstraction.
    #     #   - False: Standard split (classify entity as a whole).
    #     #   - True: Event-based split (data near event = Class 1, rest = Class 0).
    #     "split_event_class": [False],
    
    #     # event_window: Defines the window size around the event.
    #     #   - Continuous (0-1): Percentage of entity length.
    #     #   - Integer (>1): Number of time units before the event.
    #     #   - Only relevant if split_event_class is True and method is Supervised.
    #     "event_window": [120],
    
    #     # Stage 2: Mining parameters
    #     "mvs": [0.3],
    #     "mg": [120], # max_gap used in submit_stage2_job
    #     "rel": [7], # num_relations used in submit_stage3_job
    #     "sf": [True], # skip_followers used in submit_stage2_job
    #     "e": [0], # epsilon used in submit_stage2_job / submit_stage3_job
    
    #     # Stage 4: Aggregation/Evaluation parameters
    #     "aggregation_method": ['avg'],
    #     "num_tirps_for_selection": MAX_TIRPS_FOR_SELECTION, # Number of TIRPs to consider in Stage 2 Tirp_selection
    #     "TTE_W": [10000], # TTE_window_size used in submit_stage4_job
    #     "e_w": [0,1,2,3,4,5,10,15] # early_warning_value used in submit_stage4_job
    # },
    # {
    #     'dataset_name': 'ahe_small',
    #     # Use absolute paths or paths relative to where the orchestration script is run
    #     'data_path': "/mnt/new_groups/robertmo_group/Eldar/CPM_dataset/ahe_small/small_ahe_paa_15.csv",
    #     'event_symbol': 999, # Example event symbol
    
    #     # Examples: "tide", "tide_ks", "tide_kl_c1longer", "tide_mw_c0longer"
    #     "d_method": (
    #         ['tid3_20','tid3_c1longer_20','tid3_c0longer_20','tid3_150','tid3_c1longer_150','tid3_c0longer_150']
    #     ), # existing TID3 duration-preference variants + 10 seeded random-ordering runs for ablation
    #     "b": [2,3,4,5], # num_of_bins used in submit_stage1_job
    #     "ig": [1], # interpolation_gap used in submit_stage1_job
    
    #     # --- New parameters for Supervised Abstraction ---
    #     # split_event_class: Determines how to split the data for abstraction.
    #     #   - False: Standard split (classify entity as a whole).
    #     #   - True: Event-based split (data near event = Class 1, rest = Class 0).
    #     "split_event_class": [False],
    
    #     # event_window: Defines the window size around the event.
    #     #   - Continuous (0-1): Percentage of entity length.
    #     #   - Integer (>1): Number of time units before the event.
    #     #   - Only relevant if split_event_class is True and method is Supervised.
    #     "event_window": [100],
    
    #     # Stage 2: Mining parameters
    #     "mvs": [0.3],
    #     "mg": [100], # max_gap used in submit_stage2_job
    #     "rel": [7], # num_relations used in submit_stage3_job
    #     "sf": [True], # skip_followers used in submit_stage2_job
    #     "e": [0], # epsilon used in submit_stage2_job / submit_stage3_job
    
    #     # Stage 4: Aggregation/Evaluation parameters
    #     "aggregation_method": ['avg'],
    #     "num_tirps_for_selection": MAX_TIRPS_FOR_SELECTION, # Number of TIRPs to consider in Stage 2 Tirp_selection
    #     "TTE_W": [10000], # TTE_window_size used in submit_stage4_job
    #     "e_w": [0,1,2,3,4,5] # early_warning_value used in submit_stage4_job
    # },

]

# Original line kept commented for reference
Tirp_selection_methods = ['diff_horizontal_support','diff_mean_duration', 'diff_vertical_support','all','random']


