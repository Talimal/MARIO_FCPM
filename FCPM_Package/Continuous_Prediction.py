# import sys
# sys.path.append('/sise/robertmo-group/Eldar/projects/CPM_Framework')

import os
from collections import defaultdict
import pandas as pd
import math
import pickle
import gc
from CPM_Feature_Matrix.Create_feature_matrix import create_test_feature_matrix
from FCPM_Package.build_and_evaluate import *
from FCPM_Package.FCPM import *
from FCPM_Package.TTE import *


def get_tirps_models_path(base_dir_path):
    """
    Retrieves a dictionary mapping TIRP names to their corresponding model file paths and test files.
    The method scans the base directory for subdirectories starting with 'tirp_'. 
    It extracts the suffix after 'tirp_' as the TIRP name. Then it enters each directory, 
    collects the models ('FCPM.pkl' and 'TTE.pkl') and the test file 'final_test_feature_matrix.csv'.

    Args:
        base_dir_path (str): Path to the base directory containing multiple 'tirp_' directories.

    Returns:
        dict: A nested dictionary with keys being TIRP names and values being dictionaries
              containing paths to 'FCPM', 'TTE', and 'Test_File'.
        Example:
        {
            "3-2_1_999_0_0_0": {
                "FCPM": "/path/to/tirp_3-2_1_999_0_0_0/models/FCPM.pkl",
                "TTE": "/path/to/tirp_3-2_1_999_0_0_0/models/TTE.pkl",
                "Test_File": "/path/to/tirp_3-2_1_999_0_0_0/test/final_test_feature_matrix.csv"
            }
        }
    """
    tirp_models_dict = defaultdict(dict)

    # Iterate over all directories in the base directory
    for dir_name in os.listdir(base_dir_path):
        dir_path = os.path.join(base_dir_path, dir_name)

        # Check if the directory name starts with 'tirp_' and is indeed a directory
        if os.path.isdir(dir_path) and dir_name.startswith('tirp_'):
            # Extract the TIRP name by removing the 'tirp_' prefix
            tirp_name = dir_name.replace('tirp_', '')

            # Look for models and test files within the current 'tirp_' directory
            models_dir = os.path.join(dir_path, 'models')
            test_dir = os.path.join(dir_path, 'test')

            event_time_dict = os.path.join(test_dir, 'event_time_dict.pkl')
            duration_test_file_path = os.path.join(test_dir, 'durations_merged_df.csv')

            # Collect model paths
            if os.path.isdir(models_dir):
                for file_name in os.listdir(models_dir):
                    if file_name.endswith('.pkl'):
                        if 'FCPM' in file_name:
                            tirp_models_dict[tirp_name]['FCPM'] = os.path.join(models_dir, file_name)
                        elif 'CPML' in file_name:
                            tirp_models_dict[tirp_name]['CPML'] = os.path.join(models_dir, file_name)
                        elif 'TTE' in file_name:
                            tirp_models_dict[tirp_name]['TTE'] = os.path.join(models_dir, file_name)

            # Collect the test file path
            if os.path.isfile(duration_test_file_path):
                tirp_models_dict[tirp_name]['duration_test_file'] = duration_test_file_path
            if os.path.isfile(event_time_dict):
                tirp_models_dict[tirp_name]['event_time_dict'] = event_time_dict

    return tirp_models_dict


def predict_TTE(data, tte_model):
    # Filter out invalid TTEs
    data["TTE"] = data["TTE"] + 1
    TFS = data["current_time"]

    # Determine outcome_class: 0 if event_time is null, else 1
    outcome_class = data["event_time"].notna().astype(int)

    # Extract true labels and features
    y_test = data["TTE"]
    X_test = data.drop(
        columns=["EntityID", "instance_ID", "instance_start_time", "current_time", "TTE", "event_time"])

    # Make predictions
    y_pred = tte_model.predict(X_test)

    # Save predictions alongside true values
    results_df = pd.DataFrame({
        "EntityID": data["EntityID"],
        "instance_ID": data["instance_ID"],
        "TFS": TFS,
        "TTE_prediction": y_pred,
        "TTE_true": y_test,
        "outcome_class": outcome_class
    })

    return results_df




def predict_continuous(base_dir, output_dir, class_file, entities_list=[], epsilon=1,models_path=None):
    """
    Performs continuous prediction for FCPM and TTE models for specified entities and TIRPs.

    Args:
        base_dir (str): Base directory where TIRP-specific subdirectories are located.
        output_dir (str): Directory to save the prediction results.
        class_file (str): Path to the CSV file containing entity class information (EntityID, ClassID).
        entities_list (list, optional): A list of EntityIDs to process. If empty, processes all. Defaults to [].
        epsilon (int, optional): Epsilon parameter for FCPM prediction. Defaults to 1.

    Returns:
        str: The output directory path where predictions are saved.
    """
    if models_path is None:
        models_path = get_tirps_models_path(base_dir)

    class_df = pd.read_csv(class_file, dtype={"EntityID": int, "ClassID": int})
    class_map = dict(zip(class_df["EntityID"], class_df["ClassID"]))
    del class_df
    gc.collect()

    # Set up our separate output directories
    output_dir_fcpm = str(output_dir) + "_FCPM"
    
    # Check if CPML exists in ANY TIRP model mapping
    has_cpml = any(p.get("CPML") is not None for p in models_path.values())
    active_output_dirs = [output_dir_fcpm]
    if has_cpml:
        output_dir_cpml = str(output_dir) + "_CPML"
        active_output_dirs.append(output_dir_cpml)

    for tirp_name, paths in models_path.items():
        # skip if the tirp name doesnt start with tirp_3
      
        print(f"Processing TIRP: {tirp_name}")
        TTE_model_path = paths.get("TTE")
        FCPM_model_path = paths.get("FCPM")
        CPML_model_path = paths.get("CPML")
        duration_test_file_path = paths.get("duration_test_file")
        event_time_dict_path = paths.get("event_time_dict")

        # Check if all necessary paths are found for the current TIRP
        if not all([TTE_model_path, FCPM_model_path, duration_test_file_path, event_time_dict_path]):
            print(f"Skipping TIRP {tirp_name} due to missing model or data files.")
            # Create dummy files for this TIRP if critical files are missing
            for entity_id in entities_list:
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id], "instance_ID": [-1], "TFS": [0],
                        "FCPM_Prediction": [0.0], "TTE_prediction": [0.0],
                        "TTE_true": [0], "outcome_class": [class_map.get(entity_id, 0)],
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"), index=False, compression='gzip')
            continue

        # Load the specific test file for this TIRP
        try:
            durations_test_df = pd.read_csv(duration_test_file_path, low_memory=False)
        except FileNotFoundError:
            print(f"Duration test file not found for TIRP {tirp_name}: {duration_test_file_path}")
            # Create dummy files and skip
            for entity_id in entities_list:
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id], "instance_ID": [-1], "TFS": [0],
                        "FCPM_Prediction": [0.0], "TTE_prediction": [0.0],
                        "TTE_true": [0], "outcome_class": [class_map.get(entity_id, 0)],
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"), index=False, compression='gzip')
            continue
        except pd.errors.EmptyDataError:  # Catch error for empty or unparsable CSV
            print(
                f"Duration test file is empty or has no columns to parse for TIRP {tirp_name}: {duration_test_file_path}")
            # Create dummy files and skip
            for entity_id in entities_list:
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id], "instance_ID": [-1], "TFS": [0],
                        "FCPM_Prediction": [0.0], "TTE_prediction": [0.0],
                        "TTE_true": [0], "outcome_class": [class_map.get(entity_id, 0)],
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"), index=False, compression='gzip')
            continue

        duration_filtered_df = durations_test_df[durations_test_df["EntityID"].isin(entities_list)]

        if duration_filtered_df.empty and entities_list:  # Ensure entities_list is not empty
            print(f"No duration data for specified entities in TIRP: {tirp_name}. Creating dummy files.")
            for entity_id in entities_list:
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id],
                        "instance_ID": [-1],
                        "TFS": [0],
                        "FCPM_Prediction": [0.0],
                        "TTE_prediction": [0.0],
                        "TTE_true": [0],
                        "outcome_class": [class_map.get(entity_id, 0)],  # Default to 0 if not in class_map
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(
                        os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"),
                        index=False, compression='gzip'
                    )
            del durations_test_df  # Free memory
            gc.collect()
            continue  # Skip to next TIRP

        del durations_test_df  # Free memory
        gc.collect()

        try:
            with open(event_time_dict_path, "rb") as f:
                event_time_dict = pickle.load(f)
        except FileNotFoundError:
            print(f"Event time dictionary not found for TIRP {tirp_name}: {event_time_dict_path}")
            # Create dummy files and skip
            for entity_id in entities_list:
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id], "instance_ID": [-1], "TFS": [0],
                        "FCPM_Prediction": [0.0], "TTE_prediction": [0.0],
                        "TTE_true": [0], "outcome_class": [class_map.get(entity_id, 0)],
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"), index=False, compression='gzip')
            continue

        event_time_dict = {float(k): v for k, v in event_time_dict.items()}

        print(f"Creating test feature matrix for TIRP: {tirp_name}")
        # This is where the create_test_feature_matrix function is called.
        # Ensure this function can handle empty duration_filtered_df gracefully or returns an empty df.
        if duration_filtered_df.empty:  # If, after filtering, it became empty (e.g. entities_list was empty initially)
            print(
                f"Duration data became empty after filtering for TIRP: {tirp_name} (possibly empty entities_list). Creating dummy files.")
            # This case might be redundant if the previous check for duration_filtered_df.empty covers it
            # However, keeping it for safety if entities_list was initially empty.
            # If entities_list is truly empty, this loop won't run.
            for entity_id in entities_list:  # This loop will only run if entities_list is not empty
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id], "instance_ID": [-1], "TFS": [0],
                        "FCPM_Prediction": [0.0], "TTE_prediction": [0.0],
                        "TTE_true": [0], "outcome_class": [class_map.get(entity_id, 0)],
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"), index=False, compression='gzip')
            # Clean up, ensure variables exist before deleting
            if 'duration_filtered_df' in locals(): del duration_filtered_df
            if 'event_time_dict' in locals(): del event_time_dict
            gc.collect()
            continue

        filtered_df = create_test_feature_matrix(duration_filtered_df, event_time_dict)

        # --- NEWLY ADDED LOGIC in previous turn ---
        if filtered_df.empty and entities_list:  # Check if filtered_df is empty and entities_list is not
            print(
                f"Feature matrix (filtered_df) is empty for TIRP: {tirp_name} after creation. Creating dummy files for entities.")
            for entity_id in entities_list:
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id],
                        "instance_ID": [-1],
                        "TFS": [0],
                        "FCPM_Prediction": [0.0],
                        "TTE_prediction": [0.0],
                        "TTE_true": [0],
                        "outcome_class": [class_map.get(entity_id, 0)],  # Default to 0 if not in class_map
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(
                        os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"),
                        index=False, compression='gzip'
                    )
            # Clean up, ensure variables exist before deleting
            if 'duration_filtered_df' in locals(): del duration_filtered_df
            if 'event_time_dict' in locals(): del event_time_dict
            gc.collect()
            continue  # Continue to the next TIRP
        # --- END OF NEWLY ADDED LOGIC ---

        # Clean up, ensure variables exist before deleting
        if 'duration_filtered_df' in locals(): del duration_filtered_df
        if 'event_time_dict' in locals(): del event_time_dict
        gc.collect()

        try:
            with open(TTE_model_path, 'rb') as file:
                tte_model = pickle.load(file)
            with open(FCPM_model_path, 'rb') as file:
                fcpm_model = pickle.load(file)
            cpml_model = None
            if CPML_model_path:
                with open(CPML_model_path, 'rb') as file:
                    cpml_model = pickle.load(file)
        except FileNotFoundError as e:
            print(f"Error loading model for TIRP {tirp_name}: {e}. Creating dummy files.")
            for entity_id in entities_list:
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id], "instance_ID": [-1], "TFS": [0],
                        "FCPM_Prediction": [0.0], "TTE_prediction": [0.0],
                        "TTE_true": [0], "outcome_class": [class_map.get(entity_id, 0)],
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"), index=False, compression='gzip')
            if 'filtered_df' in locals(): del filtered_df  # Ensure filtered_df is deleted if it exists
            gc.collect()
            continue

        for entity_id in entities_list:
            # --- Check Continuous Prediction Resumption ---
            fcpm_done_path = os.path.join(output_dir_fcpm, str(entity_id), f"{tirp_name}.csv.gz")
            if has_cpml:
                cpml_done_path = os.path.join(output_dir_cpml, str(entity_id), f"{tirp_name}.csv.gz")
                if os.path.exists(fcpm_done_path) and os.path.exists(cpml_done_path):
                    continue
            else:
                if os.path.exists(fcpm_done_path):
                    continue

            print(f'Start prediction for entity_id: {entity_id} in TIRP: {tirp_name}')
            # Check if filtered_df is empty before trying to access it
            # This check is crucial if entities_list was initially empty,
            # as previous dummy file creation loops might not have run.
            if filtered_df.empty:
                print(
                    f"Feature matrix (filtered_df) is empty for TIRP {tirp_name} before processing entity {entity_id}. Creating dummy file.")
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id], "instance_ID": [-1], "TFS": [0],
                        "FCPM_Prediction": [0.0], "TTE_prediction": [0.0],
                        "TTE_true": [0], "outcome_class": [class_map.get(entity_id, 0)],
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"), index=False, compression='gzip')
                continue

            entity_df = filtered_df[filtered_df["EntityID"] == entity_id].copy()

            if entity_df.empty:
                # print(f"No data for entity_id: {entity_id} in TIRP: {tirp_name} after filtering. Creating dummy file.")
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id],
                        "instance_ID": [-1],
                        "TFS": [0],
                        "FCPM_Prediction": [0.0],
                        "TTE_prediction": [0.0],
                        "TTE_true": [0],
                        "outcome_class": [class_map.get(entity_id, 0)],
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(
                        os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"),
                        index=False, compression='gzip'
                    )
                continue

            # Ensure 'current_time' column exists before trying to process it
            if "current_time" not in entity_df.columns:
                print(
                    f"'current_time' column missing for entity_id {entity_id}, TIRP {tirp_name}. Creating dummy file.")
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id], "instance_ID": [-1], "TFS": [0],
                        "FCPM_Prediction": [0.0], "TTE_prediction": [0.0],
                        "TTE_true": [0], "outcome_class": [class_map.get(entity_id, 0)],
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"), index=False, compression='gzip')
                if 'entity_df' in locals(): del entity_df
                gc.collect()
                continue

            entity_df.loc[:, "current_time"] = pd.to_numeric(entity_df["current_time"], errors='coerce')
            entity_df.dropna(subset=["current_time"], inplace=True)  # Use inplace=True

            if entity_df.empty:  # Check if empty after dropping NaNs
                print(
                    f"Entity data became empty after coercing/dropping NaNs in 'current_time' for entity {entity_id}, TIRP {tirp_name}. Creating dummy file.")
                dummy = pd.DataFrame(
                    {
                        "EntityID": [entity_id], "instance_ID": [-1], "TFS": [0],
                        "FCPM_Prediction": [0.0], "TTE_prediction": [0.0],
                        "TTE_true": [0], "outcome_class": [class_map.get(entity_id, 0)],
                    }
                )
                for out_d in active_output_dirs:
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    dummy.to_csv(os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"), index=False, compression='gzip')
                if 'entity_df' in locals(): del entity_df
                gc.collect()
                continue

            entity_df.loc[:, "current_time"] = entity_df["current_time"].astype(int)

            min_time = int(entity_df["current_time"].min())
            max_time = int(entity_df["current_time"].max())

            tirp_entity_predictions_fcpm = []
            tirp_entity_predictions_cpml = []

            for timestamp in range(min_time, max_time + 1):
                df_per_timestamp = entity_df[entity_df["current_time"] == timestamp].copy()
                if df_per_timestamp.empty:
                    continue

                # TTE prediction
                tte_df = pd.DataFrame()
                try:
                    tte_df = predict_TTE(df_per_timestamp, tte_model)
                except Exception as e:
                    print(f"Error during TTE prediction for entity {entity_id}, timestamp {timestamp}: {e}")

                def _predict_and_merge(model, model_name):
                    res_df = pd.DataFrame()
                    try:
                        preds = model.predict(timestamp, entity_id, df_per_timestamp, epsilon=epsilon)
                        if isinstance(preds, list) and all(isinstance(item, dict) for item in preds):
                            if preds: res_df = pd.DataFrame(preds)
                        else:
                            print(f"Warning: {model_name} prediction for entity {entity_id}, timestamp {timestamp} did not return a list of dicts.")
                    except Exception as e:
                        print(f"Error during {model_name} prediction for entity {entity_id}, timestamp {timestamp}: {e}")

                    if not res_df.empty:
                        if "EntityID" not in res_df.columns: res_df["EntityID"] = entity_id
                        if "instance_ID" not in res_df.columns and "instance_ID" in df_per_timestamp.columns:
                            res_df["instance_ID"] = df_per_timestamp["instance_ID"].iloc[0] if not df_per_timestamp.empty else -1
                        if "TFS" not in res_df.columns: res_df["TFS"] = timestamp

                    if not res_df.empty and not tte_df.empty:
                        if 'instance_ID' in res_df.columns and 'instance_ID' in tte_df.columns:
                            try:
                                res_df['instance_ID'] = res_df['instance_ID'].astype(tte_df['instance_ID'].dtype)
                            except Exception as e:
                                print(f"Warning: Could not cast instance_ID types for merge. {model_name}: {res_df['instance_ID'].dtype}, tte: {tte_df['instance_ID'].dtype}. Error: {e}")
                        return pd.merge(res_df, tte_df, on=["EntityID", "instance_ID", "TFS"], how="left")
                    elif not res_df.empty:
                        print(f"Note: Only {model_name} prediction available for entity {entity_id}, timestamp {timestamp}. TTE part will be NaNs.")
                        res_df["TTE_prediction"] = float('nan')
                        res_df["TTE_true"] = float('nan')
                        if "event_time" in df_per_timestamp.columns:
                            res_df["outcome_class"] = df_per_timestamp["event_time"].notna().astype(int).iloc[0] if not df_per_timestamp.empty else class_map.get(entity_id, 0)
                        else:
                            res_df["outcome_class"] = class_map.get(entity_id, 0)
                        return res_df
                    elif not tte_df.empty:
                        print(f"Note: Only TTE prediction available for entity {entity_id}, timestamp {timestamp}. {model_name} part will be NaNs.")
                        temp_tte = tte_df.copy()
                        temp_tte["FCPM_Prediction"] = float('nan')
                        return temp_tte
                    return pd.DataFrame()

                merged_fcpm = _predict_and_merge(fcpm_model, "FCPM")
                if not merged_fcpm.empty:
                    tirp_entity_predictions_fcpm.append(merged_fcpm)

                if cpml_model is not None:
                    merged_cpml = _predict_and_merge(cpml_model, "CPML")
                    if not merged_cpml.empty:
                        tirp_entity_predictions_cpml.append(merged_cpml)

            if 'entity_df' in locals(): del entity_df
            gc.collect()

            # Concatenate predictions for this entity
            preds_configs = [(tirp_entity_predictions_fcpm, output_dir_fcpm)]
            if has_cpml:
                preds_configs.append((tirp_entity_predictions_cpml, output_dir_cpml))
                
            for preds_list, out_d in preds_configs:
                if preds_list:
                    preds_df = pd.concat(preds_list, ignore_index=True)
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    sort_columns = ["EntityID", "instance_ID", "TFS"]
                    existing_sort_columns = [col for col in sort_columns if col in preds_df.columns]
                    if existing_sort_columns:
                        preds_df = preds_df.sort_values(by=existing_sort_columns)
                    preds_df.to_csv(os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"), index=False, compression='gzip')
                    del preds_df
                else:
                    print(f"No predictions generated for entity_id: {entity_id} in TIRP: {tirp_name} for {out_d}. Creating dummy file.")
                    os.makedirs(f'{out_d}/{entity_id}', exist_ok=True)
                    pd.DataFrame({
                        "EntityID": [entity_id],
                        "instance_ID": [-1],
                        "TFS": [0],
                        "FCPM_Prediction": [0.0],
                        "TTE_prediction": [0.0],
                        "TTE_true": [0],
                        "outcome_class": [class_map.get(entity_id, 0)],
                    }).to_csv(os.path.join(f'{out_d}/{entity_id}', f"{tirp_name}.csv.gz"), index=False, compression='gzip')

        # Clean up at the end of TIRP processing
        if 'filtered_df' in locals(): del filtered_df
        if 'fcpm_model' in locals(): del fcpm_model
        if 'tte_model' in locals(): del tte_model
        gc.collect()

    return output_dir
