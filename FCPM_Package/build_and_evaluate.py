# import sys
# sys.path.append('/sise/robertmo-group/Eldar/projects/CPM_Framework')


import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor
import traceback
import os
from collections import defaultdict
from FCPM_Package.FCPM import FCPM
from FCPM_Package.TTE import TTE
from FCPM_Package.aggregation_class import Aggregation


class build_and_evaluate:
    def __init__(self, tirp_config, event_symbol, base_dir, cases_data_path, controls_data_path):
        """
        Initializes and sets up the required configurations and directories
        for the models and prediction outputs.

        :param tirp_config: Configuration settings for the TIRP model.
        :type tirp_config: Any
        :param event_symbol: The event symbol associated with the model.
        :type event_symbol: str
        :param total_entities: The total number of entities to be used in the model.
        :type total_entities: int
        :param base_dir: The base directory path where the model and prediction
            output directories are to be created.
        :type base_dir: str
        """
        self.tirp_config = tirp_config
        self.fcpm_model = None
        self.tte_model = None
        self.event_symbol = event_symbol
        self.cases_data_path = cases_data_path
        self.controls_data_path = controls_data_path



        self.base_dir = base_dir
        self.model_output_dir = os.path.join(base_dir, "models")
        os.makedirs(self.model_output_dir, exist_ok=True)
        prediction_dir = os.path.dirname(base_dir)
        self.prediction_output_dir = os.path.join(prediction_dir, "prediction_output")
        # os.makedirs(self.prediction_output_dir, exist_ok=True)


    def save_model(self, tirp_name, model, model_type, output_dir):
        model_name = f'{tirp_name}-{model_type}.pkl'
        model_path = os.path.join(output_dir, model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    def train_FCPM(self, fcpm_train_cls0, fcpm_train_cls1, cases_data_path, controls_data_path):
        fcpm_object = FCPM()
        fcpm_object.fit(fm_dir_cls1=fcpm_train_cls1,fm_dir_cls0=fcpm_train_cls0,cases_data_path =cases_data_path,controls_data_path =controls_data_path)
  
        self.fcpm_model = fcpm_object

    def train_TTE(self, tte_train):
        #using default alpha of 1.0, max_iter of 1000, and tol of 1e-6
        tte_object = TTE()
        data = pd.read_csv(tte_train)

        data["TTE"] = data["TTE"] + 1

        y_train = data["TTE"]
        # Drop specified columns first, then remove the last two columns (about the event)
        X_train = data.drop(columns=["EntityID", "instance_ID", "TFS", "TTE"])
        self.tte_model = tte_object.fit(X_train, y_train)

    def build_model_and_predict(self):
        self.build_model(tirp_name = self.tirp_config["tirp_name"],fcpm_train_cls0=self.tirp_config["fcpm_cls0_path"],
                         fcpm_train_cls1=self.tirp_config["fcpm_cls1_path"],tte_train = self.tirp_config["tte_train_path"],
                         cases_data_path = self.cases_data_path, controls_data_path = self.controls_data_path
                         )
        # self.predict_continuous(tirp_name = self.tirp_config["tirp_name"],test_path = self.tirp_config["test_path"],
        #                         output_dir = self.prediction_output_dir)



    def build_model(self, tirp_name, fcpm_train_cls0, fcpm_train_cls1, tte_train, cases_data_path, controls_data_path):
        print("Training FCPM...")
        self.train_FCPM(fcpm_train_cls0, fcpm_train_cls1, cases_data_path, controls_data_path)
        print("FCPM trained.")
        self.save_model(tirp_name, self.fcpm_model, 'FCPM', self.model_output_dir)

        print("Training TTE...")
        self.train_TTE(tte_train)
        print("TTE trained.")
        self.save_model(tirp_name, self.tte_model, 'TTE', self.model_output_dir)




    def predict_TTE(self, data):

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
        y_pred = self.tte_model.predict(X_test)

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

    def predict_continuous(self,tirp_name, test_path, output_dir, epsilon=1):

        test_df = pd.read_csv(test_path, low_memory=False)
        test_df["current_time"] = test_df["current_time"].astype(int)

        unique_entities = test_df["EntityID"].unique()
        all_predictions = []

        for entity_id in unique_entities:
            print(f'start entity_id: {entity_id}')
            entity_df = test_df[test_df["EntityID"] == entity_id].copy()

            min_time = entity_df["current_time"].min()
            max_time = entity_df["current_time"].max()

            entity_predictions = []

            for timestamp in range(min_time, max_time + 1):
                df_per_timestamp = entity_df[entity_df["current_time"] == timestamp].copy()
                if df_per_timestamp.empty:
                    continue

                # FCPM prediction (your model's method)
                fcpm_preds = self.fcpm_model.predict(timestamp, entity_id, df_per_timestamp, epsilon)
                fcpm_df = pd.DataFrame(fcpm_preds)

                # TTE prediction
                tte_df = self.predict_TTE(df_per_timestamp)

                # Merge FCPM and TTE results
                merged = pd.merge(fcpm_df, tte_df, on=["EntityID", "instance_ID", "TFS"], how="left")
                entity_predictions.append(merged)

            # Concatenate predictions for this entity
            entity_predictions_df = pd.concat(entity_predictions, ignore_index=True)
            all_predictions.append(entity_predictions_df)

            # Save per entity
            if output_dir:
                os.makedirs(f'{output_dir}/{entity_id}', exist_ok=True)
                entity_predictions_df.to_csv(os.path.join(f'{output_dir}/{entity_id}', f"{tirp_name}.csv"), index=False)

        return output_dir
    
    @staticmethod
    def predict_for_tirp(config_dict, model_output_dir, prediction_output_dir, base_dir):
        tirp_name = config_dict["tirp_name"]
        test_path = config_dict["test_path"]
        print(f"\n===== [PID {os.getpid()}] Predicting for TIRP: {tirp_name} =====")

        try:
            fcpm_model = pickle.load(open(os.path.join(model_output_dir, f"{tirp_name}-FCPM.pkl"), "rb"))
            tte_model = pickle.load(open(os.path.join(model_output_dir, f"{tirp_name}-TTE.pkl"), "rb"))

            # Use correct base_dir
            model = build_and_evaluate([config_dict], event_symbol=None, total_entities=0, base_dir=base_dir)
            model.fcpm_model = fcpm_model
            model.tte_model = tte_model

            model.predict_continuous(tirp_name, test_path, output_dir=prediction_output_dir)
            return f"Done predicting for {tirp_name}"

        except Exception as e:
            print(f"[ERROR] TIRP {tirp_name} failed: {e}")
            traceback.print_exc()
            return f"Failed predicting for {tirp_name}"



    def predict_multiple_tirps(self, max_workers=4):
        print(f"\n===== Starting parallel TIRP prediction with {max_workers} workers =====")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    build_and_evaluate.predict_for_tirp,
                    config,
                    self.model_output_dir,
                    self.prediction_output_dir,
                    self.base_dir  # <-- pass it here!
                )
                for config in self.tirp_config
            ]
            for future in futures:
                print(future.result())

        print(f"\nAll TIRP predictions saved under: {self.prediction_output_dir}")
