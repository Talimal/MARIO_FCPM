from CPM_Feature_Matrix.Create_feature_matrix import *
# from Params import *


def create_feature_matrix_for_CPM(class0_file_path, class1_file_path, test_file_path,
                                  max_gap, num_relations, epsilon, tirp_obj, event_symbol,
                                  output_folder_base):
    """
    Run the complete pipeline:
      - For class0 data (where the event does not occur): run_build_tables with TTE=False.
      - For class1 data (where the event occurs): run_build_tables with TTE=True.
      - For test data: run_test_table is used.

    All outputs (prefix tables, merged tables, trimmed tables, and feature matrices) are saved
    under subfolders of output_folder_base.

    Parameters:
      class0_file_path (str): File path for class0 data.
      class1_file_path (str): File path for class1 data.
      test_file_path (str): File path for test data.
      max_gap (int): Maximum gap allowed between intervals.
      num_relations (int): Number of temporal relation modes (e.g. 3 or 7).
      epsilon (int or float): Epsilon value for temporary relation determination.
      tirp_obj: TIRP object.
      event_symbol (str): Symbol representing the event.
      output_folder_base (str): Base folder where all output files will be saved.

    Returns:
      dict: Dictionary with keys "class0", "class1", and "test" containing the respective outputs.
    """
    os.makedirs(output_folder_base, exist_ok=True)

    # Process class0 data (no event)
    class0_folder = os.path.join(output_folder_base, "class0")
    os.makedirs(class0_folder, exist_ok=True)
    print("Processing class0 data (TTE=False)...")
    class0_result = run_build_tables(class0_file_path, max_gap, num_relations, epsilon,
                                     tirp_obj.copy_tirp(),event_symbol=event_symbol,class1=False, TTE=False, output_folder=class0_folder)

    # Process class1 data (event present)
    class1_folder = os.path.join(output_folder_base, "class1")
    os.makedirs(class1_folder, exist_ok=True)
    print("Processing class1 data (TTE=True)...")
    class1_result = run_build_tables(class1_file_path, max_gap, num_relations, epsilon,
                                     tirp_obj.copy_tirp(), TTE=True,event_symbol=event_symbol,class1=True, output_folder=class1_folder)

    # Process test data using run_test_table
    test_folder = os.path.join(output_folder_base, "test")
    os.makedirs(test_folder, exist_ok=True)
    print("Processing test data using run_test_table...")
    test_result = run_test_table(test_file_path, max_gap, num_relations, epsilon,
                                 tirp_obj.copy_tirp(), event_symbol=event_symbol, output_folder=test_folder)

    # Collect and return the results in a dictionary.
    results = {
        "class0": class0_result,
        "class1": class1_result,
        "test": test_result
    }

    return results

