import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit # Added GroupShuffleSplit
import os
import argparse
import sys
import traceback # Keep for detailed error printing
from sklearn.utils import shuffle
from config import SEED

def split_single_dataset_folds(dataset_name, data_path, matching_path, n_folds, dataset_base_dir):
    """
    Splits a single dataset. If n_folds > 1, uses GroupKFold.
    If n_folds == 1, uses GroupShuffleSplit for a single train/test split
    (approximating 80/20 ratio, respecting groups).
    Saves train/test splits into fold-specific subdirectories.
    Keeps original validation and prints.
    """
    # --- Keep original prints ---
    print(f"--- Starting Stage 0: Fold Splitting for Dataset: {dataset_name} ---")
    print(f"Data Path: {data_path}")
    print(f"Matching Path: {matching_path}")
    print(f"Number of Folds: {n_folds}")
    print(f"Output Base Directory for Folds: {dataset_base_dir}")
    # --- Keep original try/except structure ---
    try:
        # --- Keep original Input Validation ---
        if not os.path.exists(data_path):
            print(f"ERROR: Data file not found at {data_path}")
            sys.exit(1)
        if not os.path.exists(matching_path):
            print(f"ERROR: Matching file not found at {matching_path}")
            sys.exit(1)

        # --- Keep original Load Data ---
        print("Loading data...")
        data = pd.read_csv(data_path)
        matching = pd.read_csv(matching_path)
        print(f"Data loaded: {data.shape[0]} rows")
        print(f"Matching loaded: {matching.shape[0]} rows")

        # Keep original Check required columns
        if 'EntityID' not in data.columns or 'EntityID' not in matching.columns or 'GroupID' not in matching.columns:
             print("ERROR: Required columns ('EntityID' in data/matching, 'GroupID' in matching) not found.")
             sys.exit(1)

        # --- Prepare for Splitting (common part) ---
        X_ids = matching['EntityID'] # Use a distinct name
        groups = matching['GroupID']
        X_ids, groups = shuffle(X_ids, groups, random_state=SEED)

        # --- Conditional Splitting Logic ---
        if n_folds == 1:
            # --- Logic for Single Split ---
            print("Performing single train/test split (n_folds=1) using GroupShuffleSplit (approx 80/20 ratio)...")
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED) # Use fixed random state
            # Get the single pair of indices
            try:
                 # Using list comprehension to get the first (and only) split
                 train_index, test_index = next(iter(gss.split(X_ids, y=None, groups=groups)))
            except StopIteration:
                 # This might happen if data is too small or groups don't allow splitting at the desired ratio
                 print("ERROR: GroupShuffleSplit did not yield any splits. Check group structure, data size, and test_size.")
                 sys.exit(1)

            train_entities = X_ids.iloc[train_index]
            test_entities = X_ids.iloc[test_index]

            # Keep prints similar to the loop in original code
            print(f"  Processing Single Split (Fold 1)...")
            print(f"    Train entities: {len(train_entities)}, Test entities: {len(test_entities)}")

            # Filter data
            train_data = data[data['EntityID'].isin(train_entities)]
            test_data = data[data['EntityID'].isin(test_entities)]

            print(f"    Train data shape: {train_data.shape}")
            print(f"    Test data shape: {test_data.shape}")

            # Save to fold_1 directory
            fold_num = 1
            fold_dir = os.path.join(dataset_base_dir, f'fold_{fold_num}')
            print(f"    Creating output directory: {fold_dir}")
            os.makedirs(fold_dir, exist_ok=True)

            train_data_file = os.path.join(fold_dir, 'train.csv')
            test_data_file = os.path.join(fold_dir, 'test.csv')

            print(f"    Saving {os.path.basename(train_data_file)}...")
            train_data.to_csv(train_data_file, index=False)
            print(f"    Saving {os.path.basename(test_data_file)}...")
            test_data.to_csv(test_data_file, index=False)

        elif n_folds > 1:
            # --- Keep original GroupKFold Logic ---
            print(f"Performing {n_folds}-fold split using GroupKFold...")
            gkf = GroupKFold(n_splits=n_folds)
            fold_num = 1
            # Check if generator yields anything to avoid issues with too few groups
            split_generator = gkf.split(X_ids, y=None, groups=groups)
            try:
                # Peek at the first split without consuming it from the main generator if possible
                # A simpler way is just to check if the generator is empty after trying to iterate once
                # Let's keep it simple and assume it works or fails in the loop
                 pass
            except ValueError as ve:
                 # GroupKFold raises ValueError if n_splits > number of groups
                 print(f"ERROR during GroupKFold split generation: {ve}")
                 print("Ensure n_splits is not greater than the number of unique groups.")
                 sys.exit(1)


            for train_index, test_index in split_generator: # Use the original generator
                train_entities = X_ids.iloc[train_index]
                test_entities = X_ids.iloc[test_index]

                # Keep original prints inside the loop
                print(f"  Processing Fold {fold_num}...")
                print(f"    Train entities: {len(train_entities)}, Test entities: {len(test_entities)}")

                train_data = data[data['EntityID'].isin(train_entities)]
                test_data = data[data['EntityID'].isin(test_entities)]

                print(f"    Train data shape: {train_data.shape}")
                print(f"    Test data shape: {test_data.shape}")

                fold_dir = os.path.join(dataset_base_dir, f'fold_{fold_num}')
                print(f"    Creating output directory: {fold_dir}")
                os.makedirs(fold_dir, exist_ok=True)

                train_data_file = os.path.join(fold_dir, 'train.csv')
                test_data_file = os.path.join(fold_dir, 'test.csv')

                print(f"    Saving {os.path.basename(train_data_file)}...")
                train_data.to_csv(train_data_file, index=False)
                print(f"    Saving {os.path.basename(test_data_file)}...")
                test_data.to_csv(test_data_file, index=False)

                fold_num += 1
        else:
             # Handle n_folds < 1 case
             print(f"ERROR: n_folds must be >= 1, but received {n_folds}")
             sys.exit(1)

        # --- Keep original finished print ---
        print(f"--- Finished Stage 0: Fold Splitting for Dataset: {dataset_name} ---")

    # --- Keep original except blocks ---
    except FileNotFoundError as e:
        print(f"ERROR: File not found during Stage 0 execution: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Missing expected column in input file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during Stage 0 execution: {e}")
        traceback.print_exc() # Keep detailed traceback
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: Split dataset into K folds (GroupKFold) or single split (GroupShuffleSplit if K=1).")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--matching_path", required=True)
    parser.add_argument("--n_folds", required=True, type=int, help="Number of folds (>=1). Use 1 for single 80/20 split.")
    parser.add_argument("--dataset_base_dir", required=True)
    args = parser.parse_args()

    # Add explicit check for n_folds >= 1 as argparse type=int doesn't prevent negative values alone
    if args.n_folds < 1:
        print(f"ERROR: --n_folds must be 1 or greater, received {args.n_folds}")
        sys.exit(1)

    # Call the main processing function
    split_single_dataset_folds(
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        matching_path=args.matching_path,
        n_folds=args.n_folds,
        dataset_base_dir=args.dataset_base_dir
    )

    sys.exit(0)