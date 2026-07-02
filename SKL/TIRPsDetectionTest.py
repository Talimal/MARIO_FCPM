from unittest import TestCase

import numpy as np

# from bgukarmalego.TIRPsDetection_Framework.TIRPsDetection import TIRPsDetection
# from bgukarmalego.KarmaLego_Framework import RunKarmaLego
from Tirp_detection import *
from unittest import TestCase
import numpy as np
import time
import pickle
import os
import random



class TestTIRPsDetection(TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up KarmaLego and TIRP detection parameters based on environment variables.
        """
        cls.file_path = os.environ.get("FILE_PATH", r"C:\Users\nivsh\Downloads\KL-class-1.0.txt")
        cls.min_ver_support = float(os.environ.get("MIN_VER_SUPPORT", 0.7))
        cls.max_gap = int(os.environ.get("MAX_GAP", 120))
        cls.num_relations = int(os.environ.get("NUM_RELATIONS", 7))
        cls.epsilon = float(os.environ.get("EPSILON", 0.0))

        print("Running KarmaLego to discover frequent TIRPs...")
        lego, karma = RunKarmaLego.runKarmaLego(
            time_intervals_path=cls.file_path,
            min_ver_support=cls.min_ver_support,
            num_relations=cls.num_relations,
            max_gap=cls.max_gap,
            label='999',
            output_path='123.txt',
            incremental_output=True,
            max_tirp_length=7,
            num_comma=2,
            symbol_type='int',
            skip_followers=False,
            entity_ids_num=2,
            index_same=False,
            semicolon_end=True,
            need_one_sized=True,
            selected_variables=[],
            calc_offsets=False,
            print_instances=True,
            print_params=False
        )
        cls.lego_frequent_tirps = lego.frequent_tirps

        detector = TIRPDetector(
            time_intervals_path=cls.file_path,
            num_relations=cls.num_relations,
            max_gap=cls.max_gap,
            epsilon=cls.epsilon,
            output_path="../KarmaLego_TestsData/detected_tirps_output.txt",
            print_instances=False,
            one_size_tirp=True
        )
        cls.detected_tirps = detector.run_detection(cls.lego_frequent_tirps, use_parallel=False)

    def _compare_detected_and_expected_patterns(self, detected_patterns, expected_patterns):
        """
        Compare detected TIRP patterns with expected TIRP patterns to verify correctness.
        This version accounts for different TIRP object formats while ensuring they contain the same information.

        :param detected_patterns: List of detected TIRP patterns (in one format).
        :param expected_patterns: List of expected TIRP patterns (in another format).
        """
        # Create dictionaries of detected and expected patterns with their string representations as keys
        detected_pat_dict = {pat.to_string(): pat for pat in detected_patterns if pat != None}
        expected_pat_dict = {pat.to_string(): pat for pat in expected_patterns}

        # Assert that both detected and expected patterns have the same keys
        self.assertEqual(detected_pat_dict.keys(), expected_pat_dict.keys())

        for k_det, v_det in detected_pat_dict.items():
            # Retrieve the corresponding expected pattern
            v_exp = expected_pat_dict[k_det]

            # Extract supporting sequences by entity for expected patterns
            pat_detected_entities = {}
            for entity_id, instances in v_det.instances.groupby("EntityID"):
                entity_instances = []
                for _, instance in instances.iterrows():
                    sti = []
                    for symbol, value in instance.items():
                        if symbol != "EntityID":
                            if isinstance(value, tuple) and len(value) == 2:
                                start, end = value
                                sti.append((symbol, (start, end)))
                            else:
                                raise ValueError(f"Expected a tuple (start, end) for symbol {symbol}, but got {value}")
                    entity_instances.append({"sti": sti, "EntityID": instance["EntityID"]})
                pat_detected_entities[entity_id] = entity_instances

            # Extract supporting sequences by entity for detected patterns (original format)
            pat_expected_entities = v_exp._supporting_sequences_by_entity

            # Compare the sets of entity IDs between detected and expected patterns
            detected_entity_ids = set(pat_detected_entities.keys())
            expected_entity_ids = set(pat_expected_entities.keys())
            # self.assertEqual(detected_entity_ids, expected_entity_ids)

            # For each entity, compare the supporting sequences
            for entity_id in detected_entity_ids:
                det_instances = pat_detected_entities[entity_id]
                exp_instances = pat_expected_entities[entity_id]

                # Convert detected and expected instances into comparable dictionaries
                det_dict = {}
                for instance in det_instances:
                    key = ''.join([f"{symbol}[{start}-{end}]" for symbol, (start, end) in instance["sti"]])
                    det_dict[key] = instance

                exp_dict = {}
                for instance in exp_instances:
                    # print(instance)
                    key = ''.join([f"{sti._symbol}[{sti._start_time}-{sti._end_time}]" for sti in instance.sti])
                    exp_dict[key] = instance

                # Assert that both dictionaries have the same keys (instance representation matches despite format differences)
                self.assertEqual(det_dict.keys(), exp_dict.keys())

    def test_sequential_tirps_detection_multiply_entities(self):
        """
        This function runs KarmaLego to discover frequent TIRPs and then tests the detection process on those TIRPs.
        """

        detected_tirps = self.detected_tirps
        lego_frequent_tirps = self.lego_frequent_tirps


        # Compare detected TIRPs with originally discovered frequent TIRPs
        start_comparison = time.time()
        self._compare_detected_and_expected_patterns(detected_tirps, lego_frequent_tirps)
        end_comparison = time.time()
        print(f'Comparison execution time: {end_comparison - start_comparison:.2f} seconds')


    def test_save_and_load_detected_tirps(self):
        """
        Test saving and loading detected TIRPs, ensuring that the loaded TIRPs match the originals.
        """
        # Paths
        tirp_save_folder = r"../TIRPs_Test_Saves"  # Folder to save TIRPs


        detected_tirps = self.detected_tirps

        # Save each detected TIRP to a CSV
        os.makedirs(tirp_save_folder, exist_ok=True)
        for tirp in detected_tirps:
            tirp.save_tirp_to_csv(tirp_save_folder)

        # Load the TIRPs back and compare
        for tirp in detected_tirps:
            file_name = f"{tirp.to_string()}.csv".replace(" ", "_").replace(":", "-")
            file_path = f"{tirp_save_folder}/{file_name}"
            loaded_tirp = TIRP.load_tirp_from_csv(file_path, epsilon=self.epsilon, max_gap=self.max_gap,
                                                  num_relations=self.num_relations)

            # Assert equality of symbols
            self.assertEqual(tirp._symbols, loaded_tirp._symbols)

            # Assert equality of relations
            self.assertEqual(tirp._tirp_matrix._relations, loaded_tirp._tirp_matrix._relations)

            # Assert equality of instances
            pd.testing.assert_frame_equal(tirp.instances, loaded_tirp.instances)

        print("Save and load detected TIRPs test passed!")

    def test_comparison_functions(self):
        """
        Compare get_vertical_support, get_mean_mean_duration, and calculate_mean_horizontal_support
        for TIRPs from RunKarmaLego and the custom TIRP class.
        """


        detected_tirps = self.detected_tirps


        # Create dictionaries of detected and expected TIRPs
        detected_pat_dict = {tirp.to_string(): tirp for tirp in detected_tirps}
        expected_pat_dict = {tirp.to_string(): tirp for tirp in self.lego_frequent_tirps}

        # Assert that both dictionaries have the same keys
        self.assertEqual(detected_pat_dict.keys(), expected_pat_dict.keys())

        # Compare TIRPs using their keys
        for key in detected_pat_dict.keys():
            detected_tirp = detected_pat_dict[key]
            expected_tirp = expected_pat_dict[key]

            # Compare vertical support
            self.assertEqual(
                detected_tirp.get_vertical_support(),
                expected_tirp.get_vertical_support(),
                f"Vertical support mismatch for TIRP: {key}"
            )

            # Compare mean mean duration
            self.assertAlmostEqual(
                detected_tirp.calculate_mean_mean_duration(),
                expected_tirp.get_mean_mean_duration(),
                places=5,
                msg=f"Mean mean duration mismatch for TIRP: {key}"
            )

            # Compare mean horizontal support
            self.assertAlmostEqual(
                detected_tirp.calculate_mean_horizontal_support(),
                expected_tirp.calculate_mean_horizontal_support(),
                places=5,
                msg=f"Horizontal support mismatch for TIRP: {key}"
            )

        # """
        # Test calculate_feature_matrices_score_by_entity for functionality and errors.
        # Randomly select 15 entities for testing.
        # """
        # for detected_tirp in detected_tirps:
        #     # Get unique EntityIDs and randomly select 15 (or fewer if less than 15 exist)
        #     unique_entities = detected_tirp.instances["EntityID"].unique()
        #     selected_entities = random.sample(list(unique_entities), min(3, len(unique_entities)))
        #     selected_entities += [999,777,888]
        #
        #     for entity_id in selected_entities:
        #         try:
        #             bin_score, hs_score, md_score = detected_tirp.calculate_feature_matrices_score_by_entity(entity_id)
        #             print(f"EntityID: {entity_id}, BIN: {bin_score}, HS: {hs_score}, MD: {md_score}")
        #         except Exception as e:
        #             self.fail(
        #                 f"calculate_feature_matrices_score_by_entity raised an exception for EntityID {entity_id}: {e}")
