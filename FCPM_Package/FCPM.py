from os.path import basename

import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import warnings
import concurrent.futures
from tqdm import tqdm
import os
import re
from glob import glob
from pandas.errors import EmptyDataError
import pickle
import logging
from scipy.integrate import IntegrationWarning
import concurrent.futures
from config import USE_NORMALIZATION_RATIO


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _prefix_sort_index(path):
    """Sort key for prefix feature-matrix files: returns the leading integer of the
    filename (e.g. "5_3-13_1_999~_0_0_0.csv" -> 5). Used so the longest/full prefix is
    merged LAST in merge_prefix_tables (which keeps the last table's columns)."""
    head = basename(path).split("_")[0]
    return int(head) if head.isdigit() else 10 ** 9


class FCPM:
    def __init__(self, fit_method="sse", smoothing_method="regular", student_df=3):
        """
        Initialize the Fully Continuous Prediction Model (FCPM).

        Args:
            feature_matrix_class_0_path (str): Path to feature matrix for class 0.
            feature_matrix_class_1_path (str): Path to feature matrix for class 1.
            fit_method (str): "sse" or "mle". Determines how we choose the best distribution.
              - "sse": Compare PDF vs. histogram to find the smallest SSE.
              - "mle": Use negative log-likelihood (or AIC) on the raw data.
            smoothing_method (str): Chosen smoothing method. Must be one of:
                - "regular" → No smoothing
                - "half_cauchy" → Half-Cauchy smoothing
                - "half_cauchy_student" → Half-Cauchy + Student-t smoothing
            student_df (int): Degrees of freedom for Student-t smoothing (if applicable).
        """

        self.prefix_total_instances_controls = {}
        self.prefix_total_instances_cases = {}

        self.tirp_name = None

        if fit_method not in ["sse", "mle"]:
            raise ValueError("fit_method must be 'sse', 'mle'")
        self.fit_method = fit_method
        self.smoothing_method = smoothing_method
        self.student_df = student_df  # Degrees of freedom for Student-t smoothing

        # Storage for fitted distributions
        # self._distributions = [ss.expon, ss.weibull_min, ss.lognorm, ss.halfnorm, ss.gamma, ss.fisk, ss.gengamma]
        self._distributions = [ss.expon, ss.weibull_min, ss.lognorm, ss.pareto, ss.halfnorm, ss.exponweib]
        self.controls_dist = {}
        self.cases_dist = {}

        # Storage for loaded feature matrices
        self.fm_class_0 = None
        self.fm_class_1 = None

        self.prefixes_feature_map = {}
        self.features_prefixes_map = {}

        self.durations_controls, self.durations_cases = {}, {}

        self.prefixes_support_cases = {}
        self.prefixes_support_controls = {}
        
        # Track duration elements that are always 0 (e.g., equal temporal relations)
        self.always_zero_duration_elements_cases = set()
        self.always_zero_duration_elements_controls = set()
        # self.prefixesVS_class_1, self.prefixesVS_class_0 = {}, {}

    def merge_prefix_tables(self, dataframes_list):
        """
        Merge a list of TIRP-prefix DataFrames iteratively.
        The function starts with the first DataFrame and then merges each subsequent DataFrame.
        For each merge, it finds common columns between the current merged DataFrame and the next DataFrame,
        then retains rows in the merged DataFrame that are not present in the next DataFrame (i.e., non-evolved rows),
        and concatenates these with the next DataFrame.

        Args:
            dataframes (list): List of Pandas DataFrames to merge.

        Returns:
            pd.DataFrame: Final merged DataFrame.
        """
        if not dataframes_list:
            return pd.DataFrame()

        merged_df = dataframes_list[1]
        for next_df in dataframes_list[2:]:
            common_columns = list(set(merged_df.columns) & set(next_df.columns))
            if common_columns:
                # Identify rows in merged_df that do not appear in next_df based on common columns.
                non_evolved_rows = merged_df.merge(next_df, on=common_columns, how='left', indicator=True)
                non_evolved_rows = non_evolved_rows[non_evolved_rows['_merge'] == 'left_only']
                non_evolved_rows = non_evolved_rows.drop(columns=['_merge'])
                non_evolved_rows = non_evolved_rows[common_columns]
            else:
                non_evolved_rows = merged_df
            merged_df = pd.concat([next_df, non_evolved_rows], ignore_index=True)

        # # Remove the last column ---
        # if not merged_df.empty and merged_df.shape[1] >= 2:
        #     # Identify the name of the second-to-last column
        #     column_to_drop = merged_df.columns[-2]
        #     # Drop that column by name
        #     merged_df = merged_df.drop(columns=[column_to_drop])
        return merged_df


    def _load_csv(self, file_path):
        """
        Loads a CSV file into a Pandas DataFrame and updates the prefix-to-feature mapping.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            df = pd.read_csv(file_path)

        except EmptyDataError:
            print(f"Skipping empty file: {file_path}")
            return None, None

        number_of_instances = len(df)
        # Extract prefix name from the filename (assuming filename is structured as "<prefix>.csv")
        prefix_name = os.path.basename(file_path).replace(".csv", "")

        prefix_size = int(prefix_name.split("_")[0]) - 1

        # Identify numerical columns for feature extraction (duration feature columns)
        # duration_features_cols = [col for col in df.columns if col.startswith("(")][:prefix_size]  # TIRP feature columns
        duration_features_cols = [
            col for col in df.columns
            if col.startswith("(") and col != "(999+, 999-)"
        ]


        # Update the prefix-feature mapping with the correct structure
        self.prefixes_feature_map[prefix_name] = duration_features_cols
        self.features_prefixes_map[frozenset(duration_features_cols)] = prefix_name

        return number_of_instances, df, prefix_name

    def extract_durations(self, merged_df, completnece_class):
        """
        Extracts duration values for the given duration_feature in a class.
        
        Args:
            prefix_name (str): The name of the prefix being processed.
            merged_df (pd.DataFrame): The full DataFrame containing all entities and time-indexed rows.
            completnece_class (int): The class to extract durations for.

        Returns:
            dict: A dictionary where the key is the prefix name, and the value is another dictionary
        """
        duration_features_columns = [col for col in merged_df.columns if col.startswith("(")]
        #get duration list for each feature, filterour duration of 1
        durations = {feature: merged_df[feature].dropna().tolist() for feature in duration_features_columns}
        # durations = {feature: [duration for duration in durations[feature] if duration > 1] for feature in duration_features_columns}

        # Detect duration elements that are always 0 (equal temporal relations)
        for feature, duration_list in durations.items():
            if duration_list and all(d == 0 for d in duration_list):
                if completnece_class == 0:
                    self.always_zero_duration_elements_controls.add(feature)
                    print(f"Detected always-zero duration element in controls: {feature}")
                else:
                    self.always_zero_duration_elements_cases.add(feature)
                    print(f"Detected always-zero duration element in cases: {feature}")

        if completnece_class == 0:
            self.durations_controls = durations
        else:
            self.durations_cases = durations

        return durations    

    
        

    def extract_durations_old(self, prefix_name, df):
        """
        Extracts duration values for the given prefix where its binary column is 1
        in the entity's last row (i.e., the row with the maximum TFS).

        Args:
            prefix_name (str): The name of the prefix being processed.
            df (pd.DataFrame): The full DataFrame containing all entities and time-indexed rows.

        Returns:
            dict: A dictionary where the key is the prefix name, and the value is another dictionary
                  mapping each duration feature to a list of extracted duration values.

            Example Output:
            {
                'prefix_1': {
                    'feature_1': [3, 4, 2, ...],
                    'feature_2': [5, 1, 8, ...]
                },
                'prefix_2': {
                    'feature_1': [...],
                    ...
                }
            }
        """
        # Initialize dictionary for the given prefix
        durations = {prefix_name: {}}

        # Ensure prefix exists in feature mapping
        if prefix_name not in self.prefixes_feature_map:
            raise KeyError(f"Prefix '{prefix_name}' not found in self.prefixes_feature_map")

        # Get duration features for this prefix
        duration_features = self.prefixes_feature_map[prefix_name]

        # Extract values from the DataFrame for each duration feature
        for duration_feature in duration_features:
            if duration_feature in df.columns:
                durations[prefix_name][duration_feature] = df[duration_feature].dropna().tolist()
            else:
                durations[prefix_name][duration_feature] = []  # If feature column is missing, return empty list

        return durations

       


    def fit_with_sse(self, dist, data, number_of_bins):
        """Fits distribution using SSE minimization."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try: # Add a try-except block for robustness during fitting itself
                params = dist.fit(data)
                # Handle distributions returning different numbers of parameters
                if len(params) >= 2:
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                elif len(params) == 1: # E.g., expon needs only scale (loc=0 implied often)
                    arg = ()
                    loc = 0 # Assume location is 0 if not returned
                    scale = params[0]
                else: # Should not happen with standard scipy distributions
                     logging.warning(f"    - SSE: Unexpected number of parameters {len(params)} from {dist.name}.fit(). Skipping.")
                     return np.inf, None, None

                # Check scale > 0, important for most distributions
                # Add a small epsilon check for floating point issues near zero
                # if scale <= 1e-9: # Check if scale is non-positive or extremely small
                #     logging.debug(f"    - SSE: Invalid or near-zero scale ({scale:.4f}) for {dist.name}. Returning Inf score.")
                #     return np.inf, None, None # Return Inf score if scale is invalid

                # Calculate histogram
                y_hist, x_edges = np.histogram(data, bins=number_of_bins, density=True)

                # Ensure histogram calculation was meaningful
                if len(x_edges) < 2 or np.all(y_hist == 0):
                     logging.warning(f" - SSE: Histogram for {dist.name} resulted in < 2 edges or all zero counts with {number_of_bins} bins. Cannot calculate SSE.")
                     return np.inf, None, None

                x_mid = 0.5 * (x_edges + np.roll(x_edges, -1))[:-1] / 2.0
                pdf_est = dist.pdf(x_mid, loc=loc, scale=scale, *arg)
                # Handle potential NaN in pdf estimation
                pdf_est = np.nan_to_num(pdf_est, nan=0.0)

                sse = np.sum((y_hist - pdf_est) ** 2)

                # Check if SSE is valid (finite and non-negative)
                if np.isfinite(sse) and sse >= 0:
                    # logging.debug(f"  - SSE for {dist.name}: {sse}") # Uncomment if needed
                    # Return the valid SSE score, distribution object, and parameters
                    return sse, dist, params
                else:
                    # Handles cases like NaN SSE or negative SSE (which shouldn't happen)
                    logging.warning(f"    - SSE: Invalid SSE value ({sse}) calculated for {dist.name}. Returning Inf score.")
                    return np.inf, None, None # Return Inf score if SSE is invalid

            except (ValueError, RuntimeError, FloatingPointError, Exception) as e:
                # Catch errors during the fitting process (e.g., bad data for a distribution)
                logging.warning(f"    - SSE: Error during fitting or SSE calculation for {dist.name}: {e}")
                return np.inf, None, None # Return Inf score on any error during the process
            
    def fit_with_mle(self, dist, data):
        """Fits distribution using MLE, calculates AIC."""
        n_params = 0
        n_data = len(data)
        params = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # Fit using MLE (default for many scipy dists, force location=0 for positive data)
                params = dist.fit(data, floc=0)
                n_params = len(params)

                # Check if scale is valid
                if n_params > 1 and params[-1] <= 0:
                    logging.debug(f"    - MLE: Invalid scale ({params[-1]}) for {dist.name}. Skipping.")
                    return np.inf, None, None # Return infinite AIC if params invalid

                # Calculate Negative Log-Likelihood (NLL)
                loglikelihood = dist.logpdf(data, *params)
                # Handle -inf from logpdf (e.g., data point exactly at boundary)
                loglikelihood[loglikelihood == -np.inf] = -1e10 # Replace with very small number
                if np.isnan(loglikelihood).any():
                        logging.warning(f"    - MLE: NaN in loglikelihood for {dist.name}. Skipping.")
                        return np.inf, None, None
                nll = -np.sum(loglikelihood)

                # Calculate AIC
                aic = 2 * n_params + 2 * nll
                # Add penalty if fit is poor (e.g., nll is huge or inf) - AIC should be finite
                if not np.isfinite(aic):
                    logging.debug(f"    - MLE: Non-finite AIC ({aic}) for {dist.name}. Assigning Inf.")
                    aic = np.inf


            logging.debug(f"    - AIC for {dist.name}: {aic:.4f} (NLL: {nll:.4f}, Params: {n_params})")
            return aic, dist, params

        except (ValueError, RuntimeError, FloatingPointError, Exception) as e:
            logging.warning(f"    - MLE: Error fitting or calculating AIC for {dist.name}: {e}")
            return np.inf, None, None # Return infinite AIC on error
          
    def _fit_feature(self, durations, feature_name, outcome_class, plot_dir=None, create_plots=False):
        """
        Fit distribution (parametric) to a single feature, apply smoothing (param only), and optionally plot.

        Args:
            durations (list): List of durations for the feature.
            prefix_name (str): Name of the parent prefix.
            feature_name (str): Name of the feature being fitted.
            outcome_class (int): Class label (0 or 1).
            plot_dir (str, optional): Directory to save the plot. Defaults to None.
            create_plots (bool): Whether to generate and save plots.
            num_of_bins (int): Default number of histogram bins (overridden by FD rule for parametric).

        Returns:
            dict: Fitting details (distribution type, parameters/object, CDF), or None if fitting failed.
        """
        if not durations:
            logging.warning("  - Empty durations list provided. Cannot fit.")
            return None

        # Check if this is an always-zero duration element (equal temporal relation)
        always_zero_set = self.always_zero_duration_elements_cases if outcome_class == 1 else self.always_zero_duration_elements_controls
        if feature_name in always_zero_set:
            # print(f"  - Feature '{feature_name}' is always-zero duration element. Creating special distribution.")
            return {
                "distribution": "always_zero",
                "parameters": None,
                "cdf": None,
                "is_always_zero": True
            }

        data = pd.Series(durations)
                # Freedman-Diaconis
        if data.max() - data.min() != 0:
        # if data.sparse.to_dense().max() - data.sparse.to_dense().min() != 0:
            iqr_value = ss.iqr(data, rng=(25, 75), interpolation='midpoint')
            up = data.max() - data.min()
            down = 2 * (iqr_value / (len(durations) ** (1 / 3)))
            if down != 0:
                sse_bins = int(up / down)
            else:
                sse_bins = 3
        else:
            sse_bins = int(len(durations) ** (1 / 2))

        if sse_bins <= 2:
            sse_bins = 3

        print(f"  - Using {sse_bins} bins based on FD rule.")
        # Filter out non-positive values, especially important for KDE on durations
        if data.empty or len(data) < 2:
            logging.warning(f"  - Feature '{feature_name}': No positive duration data or < 2 points ({len(data)}) after filtering. Cannot fit.")
            return None
        
        n_data_points = len(data)
        num_unique_durations = len(data.unique())
        # logging.info(f"  - Fitting feature '{feature_name}' ({n_data_points} data points) using '{self.fit_method}'")

       
        y, x = np.histogram(data, bins=sse_bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        best_distribution = ss.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf
        fit_result_dict = None

        for distribution in self._distributions:
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse

            except Exception:
                pass
        



        # --- Post-Fitting ---
        if best_distribution is None or best_params is None:
            logging.warning(f"  - Feature '{feature_name}': No suitable distribution found using method '{self.fit_method}'.")
            return None
        # logging.info(f"  - Best fit for '{feature_name}': {best_distribution.name} (Method: {self.fit_method.upper()}, Score: {best_sse})")

        # Create the final CDF using the best fit
        # Smoothing is applied here based on self.smoothing_method
        pdf_series, cdf_series = self._make_pdf_cdf(best_distribution, best_params)

        #save cdf_series in txt file
        # cdf_series_name = f"Class{outcome_class}__{prefix_name}__{feature_name}__{self.fit_method}_cdf.txt"
        # cdf_series_path = os.path.join(plot_dir, cdf_series_name)
        # cdf_series.to_csv(cdf_series_path, header=True, index=True, sep="\t")

        fit_result_dict = {
            "distribution": best_distribution.name,
            "parameters": best_params,
            "cdf": cdf_series # Store the generated CDF Series
        }


                # --- Plotting ---
        if create_plots and plot_dir:
            # Determine histogram bins for plotting *visually* (independent of fitting method now)

            # if self.fit_method == "sse":
            #     plot_bins= sse_bins # Use the same bins as for SSE fitting
            # Use 'auto' bins for the plot background histogram
            # plot_bins = len(np.histogram_bin_edges(a=data, bins='auto')) - 1
            plot_bins = sse_bins
            if plot_bins < 1 : plot_bins = 10 # Ensure at least some bins


            fig, ax = plt.subplots(figsize=(12, 7))
            try:
                # 1. Plot Histogram of original data (visual aid)
                ax.hist(data, bins=plot_bins, density=True, alpha=0.6, color='skyblue', label=f'Data Histogram (Bins={plot_bins})')

                # 2. Plot the PDF of the *selected* best distribution
                x_min_plot = 0
                # Adjust max plot range if needed, avoid excessively large ranges
                x_max_plot = max(data.quantile(0.995), data.mean() + 3 * data.std()) * 1.1
                if x_max_plot <= 0: x_max_plot = 1.0 # Handle edge case of zero max

                x_pdf = np.linspace(x_min_plot, x_max_plot, 500)

                dist_plot = best_distribution # The selected distribution object
                params_plot = fit_result_dict["parameters"]
                with warnings.catch_warnings(): # Suppress warnings during PDF calculation for plot
                     warnings.simplefilter("ignore")
                     y_pdf = dist_plot.pdf(x_pdf, *params_plot)
                     y_pdf = np.nan_to_num(y_pdf, nan=0.0, posinf=0.0, neginf=0.0) # Clean PDF

                # Format parameters for display
                param_names = ['loc', 'scale']
                if len(params_plot) > 2:
                    param_names = ['shape'] + param_names
                param_str = ', '.join([f'{name}={val:.3f}' for name, val in zip(param_names, params_plot)])

                plot_label_fit = f'Best Fit PDF: {dist_plot.name}\nParams: {param_str}\nMethod: {self.fit_method.upper()}, Score: {best_sse:.4f}'
                ax.plot(x_pdf, y_pdf, 'r-', lw=2.5, label=plot_label_fit)

                # 3. Add Titles and Labels
                plot_title = (f"Fit: Feature '{feature_name}'\n"
                              f"Class: {outcome_class} | N={n_data_points}")
                ax.set_title(plot_title, fontsize=14)
                ax.set_xlabel("Duration", fontsize=12)
                ax.set_ylabel("Density", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
                ax.set_xlim(left=0) # Ensure x-axis starts at 0

                # 4. Save the plot
                filename = f"Class{outcome_class}__{feature_name}__{self.fit_method}_fit.png"
                filepath = os.path.join(plot_dir, filename)

                plt.savefig(filepath, bbox_inches='tight')
                # logging.info(f"  - Plot saved to: {filepath}")

            except Exception as e:
                logging.error(f"  - Warning: Could not create/save plot for feature '{feature_name}'. Error: {e}")
            finally:
                plt.close(fig) # IMPORTANT: Close the figure to free memory
        # --- End Plotting ---

        return fit_result_dict
    
   

    def _make_pdf_cdf(self, dist, params, size=10000):
        """
        Generate distribution's Probability Distribution Function and CDF.
        
        Args:
            dist: The distribution object
            params: Distribution parameters
            size: Number of points to generate
        
        Returns:
            tuple: (pdf_series, cdf_series)
        """
        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(0, end, size)  # Start from 0 since durations are non-negative

        # Apply the selected smoothing method
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        if self.smoothing_method == "regular":
            y = dist.pdf(x, loc=loc, scale=scale, *arg)
        # elif self.smoothing_method == "half_cauchy":
        #     y = dist.pdf(x, loc=loc, scale=scale, *arg) + ss.halfcauchy().pdf(x)
        # else:  # 'half_cauchy_student'
        #     y = dist.pdf(x, loc=loc, scale=scale, *arg) + ss.t(df=self.student_df).pdf(x)

        pdf = pd.Series(data=y, index=x)  # a matrix of x and y, where x is the indices and y are the values
        if np.isinf(pdf).any():
            pdf[np.isinf(pdf)] = 0
        cdf = pdf.cumsum() / pdf.sum()
        return pdf, cdf



    def calculate_prefixes_support(self, merged_df, completnece_class, normalization_ratio):
        '''
        for each prefix, the function filters the merged_df for each prefix in self.prefixes_feature_map
        it filter out rows in which the prefix features has at least one null value
        '''
        for prefix, features in self.prefixes_feature_map.items():
            # Filter merged_df for the current prefix
            prefix_df = merged_df[merged_df[list(features)].notna().all(axis=1)]

            if completnece_class == 1:
                if normalization_ratio > 1:
                    self.prefixes_support_cases[prefix] = len(prefix_df) * normalization_ratio
                else:
                    self.prefixes_support_cases[prefix] = len(prefix_df)

            else:
                if normalization_ratio <= 1:
                    self.prefixes_support_controls[prefix] = len(prefix_df) * normalization_ratio   
                else:
                    self.prefixes_support_controls[prefix] = len(prefix_df)


    def calculate_avg_LOS(self, cases_data_path, controls_data_path):
        # Load cases and controls full TIRP interval datasets to compute LOS
        cases_df = pd.read_csv(cases_data_path)
        controls_df = pd.read_csv(controls_data_path)

        # Compute average LOS per entity in each class
        los_cases = cases_df.groupby("EntityID").agg(
            LOS=("EndTime", "max"), Start=("StartTime", "min"))

   
        los_cases["LOS"] = los_cases["LOS"] - los_cases["Start"]
        avg_los_cases = los_cases["LOS"].mean()
    

        los_controls = controls_df.groupby("EntityID").agg(
            LOS=("EndTime", "max"), Start=("StartTime", "min"))
        los_controls["LOS"] = los_controls["LOS"] - los_controls["Start"]
        avg_los_controls = los_controls["LOS"].mean()
        normalization_ratio = avg_los_cases / avg_los_controls if avg_los_controls != 0 else 1.0

        return normalization_ratio
    
    def fit(self, fm_dir_cls1, fm_dir_cls0, cases_data_path, controls_data_path, plots_dir = None):
        """
        Train the FCPM model by fitting distributions for both classes.
        """
        create_plots = plots_dir is not None
        if create_plots:
            os.makedirs(plots_dir, exist_ok=True) # Create dir if it doesn't exist
            
        if USE_NORMALIZATION_RATIO: 
            normalization_ratio = self.calculate_avg_LOS(cases_data_path, controls_data_path)
            
        else:
            normalization_ratio = 1.0

 

        # read the prefixe instances files for dir of class 1 and merge them into one dataframe
        # NOTE: sort by the leading prefix index (e.g. "5_..." -> 5) so the longest/full prefix
        # is merged LAST. merge_prefix_tables keeps the last table's columns, so without this the
        # full prefix's columns can be dropped and calculate_prefixes_support raises KeyError.
        prefixe_dataframes_class_1 = []
        for prefix_fm_file_path in sorted(glob(os.path.join(fm_dir_cls1, "*.csv")), key=_prefix_sort_index):
            if basename(prefix_fm_file_path).startswith("merged_df") or basename(prefix_fm_file_path).startswith("TTE_") or basename(prefix_fm_file_path).startswith("1_"):
                continue
            number_of_instances, prefix_df, prefix_name = self._load_csv(prefix_fm_file_path)
            prefixe_dataframes_class_1.append(prefix_df)
            self.prefix_total_instances_cases[prefix_name] = number_of_instances
        merged_df_class_1 = self.merge_prefix_tables(prefixe_dataframes_class_1)
        #save the merged_df as csv file
        merged_df_class_1.to_csv(os.path.join(fm_dir_cls1, "merged_df.csv"), index=False)
        

        #read the prefixe instances files for dir of class 0 and merge them into one dataframe
        # sorted by leading prefix index so the full prefix is merged last (see class-1 note above)
        prefixe_dataframes_class_0 = []
        for prefix_fm_file_path in sorted(glob(os.path.join(fm_dir_cls0, "*.csv")), key=_prefix_sort_index):
            if basename(prefix_fm_file_path).startswith("merged_df") or basename(prefix_fm_file_path).startswith("TTE_") or basename(prefix_fm_file_path).startswith("1_"):
                continue
            number_of_instances, prefix_df, prefix_name = self._load_csv(prefix_fm_file_path)
            prefixe_dataframes_class_0.append(prefix_df)
            self.prefix_total_instances_controls[prefix_name] = number_of_instances
        merged_df_class_0 = self.merge_prefix_tables(prefixe_dataframes_class_0)
        #save the merged_df as csv file
        merged_df_class_0.to_csv(os.path.join(fm_dir_cls0, "merged_df.csv"), index=False)


        #calculate the prefixes support for both classes
        self.calculate_prefixes_support(merged_df_class_1, 1, normalization_ratio)
        self.calculate_prefixes_support(merged_df_class_0, 0, normalization_ratio)

        #now calculate the total_instances based on the sum of the prefixes support
        total_instances = sum(self.prefixes_support_cases.values()) + sum(self.prefixes_support_controls.values())

        self._write_support_file(
            out_path=os.path.join(fm_dir_cls1, "prefix_support_summary.txt"),
            class_name="CASES (1)",
            norm_ratio=normalization_ratio,
            raw_support_dict=self.prefixes_support_cases,          # still raw here
            total_inst=total_instances,
            norm_support_dict={p: s/total_instances for p, s in self.prefixes_support_cases.items()})
        self._write_support_file(
            out_path=os.path.join(fm_dir_cls0, "prefix_support_summary.txt"),
            class_name="CONTROLS (0)",
            norm_ratio=normalization_ratio,
            raw_support_dict=self.prefixes_support_controls,
            total_inst=total_instances,
            norm_support_dict={p: s/total_instances for p, s in self.prefixes_support_controls.items()})


        #normlize the supports with total_entities
        self.prefixes_support_cases = {prefix: support / total_instances for prefix, support in self.prefixes_support_cases.items()}
        self.prefixes_support_controls = {prefix: support / total_instances for prefix, support in self.prefixes_support_controls.items()}

        print(f"Prefixes support for CASES: {self.prefixes_support_cases}")
        print(f"Prefixes support for CONTROLS: {self.prefixes_support_controls}")       
        #extract the durations for both classes
        self.extract_durations(merged_df_class_1, 1)
        self.extract_durations(merged_df_class_0, 0)

        #fit the distributions for both classes
        for feature_name in self.durations_cases.keys():
            fitted_dist_info = self._fit_feature(self.durations_cases[feature_name], feature_name, 1, plots_dir, create_plots)
            self.cases_dist[feature_name] = fitted_dist_info
        for feature_name in self.durations_controls.keys():
            fitted_dist_info = self._fit_feature(self.durations_controls[feature_name], feature_name, 0, plots_dir, create_plots)
            self.controls_dist[feature_name] = fitted_dist_info



    def predict(self, timestamp, entity_id, entitiy_df_per_timestamp, epsilon=3):
        binary_cols = [col for col in entitiy_df_per_timestamp.columns if col.endswith("_Binary")]

        def predict_row(row):
            instance_id = row["instance_ID"]

            # Active binary features
            active_features = [feature for feature in binary_cols if row[feature] == 1]
            clean_active_features = [f.replace("_Binary", "") for f in active_features]

            # prefix = self.features_prefixes_map.get(frozenset(clean_active_features), None)

            # Match prefix
            prefix = None
            for candidate_prefix, feature_list in self.prefixes_feature_map.items():
                if set(feature_list) == set(clean_active_features):
                    prefix = candidate_prefix
                    break

            # Extract durations
            features_values_dict = {
                feature: row.get(f"{feature}_Duration", 0.0)
                for feature in clean_active_features
            }

            prob = self.compute_probability(prefix, features_values_dict, epsilon=epsilon)

            return {
                "EntityID": entity_id,
                "instance_ID": instance_id,
                "TFS": timestamp,
                "FCPM_Prediction": prob
            }

        predictions = entitiy_df_per_timestamp.apply(predict_row, axis=1).tolist()
        return predictions


    def _get_probability_from_cdf(self, dist_info, value, epsilon, with_csf_series = False):
        """
        Calculate probability P(value - epsilon < X <= value + epsilon)
        using the THEORETICAL CDF of the fitted distribution.

        Args:
            dist_info (dict): Dictionary containing 'distribution' (name)
                            and 'parameters' for the fitted distribution.
            value (float): Observed duration value.
            epsilon (float): Half-width for the probability interval.

        Returns:
            float: Probability P(value - epsilon < X <= value + epsilon), clamped to [0, 1].
        """
        # Handle always-zero duration elements (equal temporal relations)
        if dist_info and dist_info.get("is_always_zero", False):
            return 1.0 if value == 0 else 0.0
        
        # Handle small values more appropriately - don't automatically return 1
        if value < 0:
            value = 0  # Clamp negative values to 0
        if with_csf_series:
            cdf = dist_info["cdf"]
            cdf_max = max(cdf.index.values)
            cdf_min = min(cdf.index.values)
            top = min(cdf.index.values, key=lambda x: abs(x - (value + epsilon)))
            closest_num_max = min(top, cdf_max)
            bottom = min(cdf.index.values, key=lambda x: abs(x - (value - epsilon)))
            closest_num_min = max(bottom, cdf_min)
            if value < cdf_min:  # in case the duration was not seen in the training set
                return 0
            elif value > cdf_max:
                return 0
            prob = cdf[closest_num_max] - cdf[closest_num_min]
            if not isinstance(prob, (np.floating, float)):  # in case of weird scenarios such as: only durations of 3
                return 1
            return prob

        else:
            dist   = getattr(ss, dist_info["distribution"])
            params = dist_info["parameters"]
            hi = dist.cdf(value + epsilon, *params)
            lo = dist.cdf(max(value - epsilon, 0), *params)
            prob = hi - lo
            return max(0.0, min(1.0, prob))


    def _get_censoring_probability(self, dist_info, value, with_csf_series = False):
        """
        Compute the survival probability P(X > value) = 1 - CDF(value), using the
        THEORETICAL distribution's CDF or Survival Function (SF).

        Args:
            dist_info (dict): Dictionary containing 'distribution' (name)
                            and 'parameters' for the fitted distribution.
            value (float): Observed duration value (censoring time).

        Returns:
            float: Survival probability P(X > value), clamped to [0, 1].
        """
        # Handle always-zero duration elements (equal temporal relations)
        if dist_info and dist_info.get("is_always_zero", False):
            return 1.0 if value == 0 else 0.0
        
        # Handle small values more appropriately - don't automatically return 1
        if value < 0:
            value = 0  # Clamp negative values to 0
        if with_csf_series:
            cdf = dist_info["cdf"]
            opp_cdf = 1 - cdf
            bottom = min(opp_cdf.index.values, key=lambda x: abs(x - value))
            prob = opp_cdf[bottom]
            return prob
        else:
            dist  = getattr(ss, dist_info["distribution"])
            params = dist_info["parameters"]
            # sf is numerically stabler if it exists
            prob = dist.sf(value, *params) if hasattr(dist, "sf") else 1.0 - dist.cdf(value, *params)

            return max(0.0, min(1.0, prob))



    def _compute_psi(self, prefix_prob, features_values_dict, class_dist, epsilon, censor_last_duration=True):
        """
        Compute:
           psi(t_c) = prefix_prob * Π[ Probability(duration_i) ]
        but if censor_last=True, the final duration feature uses survival (1 - CDF)
        for partial/incomplete data.

        Args:
            prefix_prob (float): Pr(prefix in this class).
            prefix (str): The prefix key. For example: 'completed_17+_17-_26+'.
            features_values_dict (dict): e.g. {"(17+)": 5.2, "(17-)": 3.1, ...}
            class_dist (dict): e.g. self.class_1_dist or self.class_0_dist
            epsilon (float): Half-width for numerical approximation used in _get_probability_from_cdf.
            censor_last_duration (bool): If True, treat the last item in features_values_dict as censored.

        Returns:
            float: The computed ψ(t_c).
        """
        psi_val = prefix_prob

        # Convert the features_values_dict items into a list so we know which is "last"
        items = list(features_values_dict.items())

        for i, (feature_name, duration_value) in enumerate(items):
            dist_info = class_dist[feature_name]
            if not dist_info:
                continue
            # Always-zero distributions don't have CDF
            if dist_info.get("cdf") is None and not dist_info.get("is_always_zero", False):
                continue

            x = float(duration_value)

            # If this is the last duration AND we want to treat it as incomplete
            is_last_duration = (i == len(items) - 1)
            if censor_last_duration and is_last_duration:
                # Use survival = 1 - CDF(x)
                prob_val = self._get_censoring_probability(dist_info, x)
                # print(f"prob_val_censoring: {prob_val} for: {feature_name} and duration:{x}")
            else:
                # Otherwise, use the normal approximate PDF = cdf(x+ε) - cdf(x−ε)
                prob_val = self._get_probability_from_cdf(dist_info, x, epsilon)
                # print(f"prob_val_pdf: {prob_val} for: {feature_name} and duration:{x}")
            psi_val *= prob_val

        return psi_val

    def compute_probability(self, prefix, features_values_dict, epsilon=1.0, censor_last=True):
        """
        Compute P(Q | t_c) = psi(t_c) / [psi(t_c) + psi_bar(t_c)]
        ...
        :param censor_last: Whether the last duration in features_values_dict is incomplete/censored.
        """
        # -----------------------------------------------------------------------
        # 1) Prefix Probability in class 1 and class 0
        #    => prefix vertical support / total_entities
        # -----------------------------------------------------------------------
        prefix_prob_class1 = self.prefixes_support_cases.get(prefix, 0.0)
        prefix_prob_class0 = self.prefixes_support_controls.get(prefix, 0.0)



        # --- 2) psi(t_c) for class1 ---
        psi = self._compute_psi(
            prefix_prob=prefix_prob_class1,
            features_values_dict=features_values_dict,
            class_dist=self.cases_dist,
            epsilon=epsilon,
            censor_last_duration=censor_last
        )
        # --- 3) psi_bar(t_c) for class0 ---
        psi_bar = self._compute_psi(
            prefix_prob=prefix_prob_class0,
            features_values_dict=features_values_dict,
            class_dist=self.controls_dist,
            epsilon=epsilon,
            censor_last_duration=censor_last
        )

        # --- 4) Combine ---
        denom = psi + psi_bar
        if denom == 0:
            return 0.0
        return psi / denom



    def _lookup_cdf(self, cdf_series, query_x, epsilon):
        """
        Return cdf value at the x in cdf_series that is closest to query_x.
        You can also do more sophisticated interpolation if you like.
        """
        cdf_max = max(cdf_series.index.values)
        cdf_min = min(cdf_series.index.values)

        top = min(cdf_series.index.values, key=lambda x: abs(x - (query_x + epsilon)))
        closest_num_max = min(top, cdf_max)  # in case its after the distribution
        bottom = min(cdf_series.index.values, key=lambda x: abs(x - (query_x - epsilon)))
        closest_num_min = max(bottom, cdf_min)  # in case its before the distribution

        if query_x < cdf_min:  # in case the duration was not seen in the training set
            return 0
        elif query_x > cdf_max:
            return 0

        prob = cdf_series[closest_num_max] - cdf_series[closest_num_min]

        if not isinstance(prob, (np.floating, float)):  # in case of weird scenarios such as: only durations of 3
            return 1

        return prob



    def _write_support_file(self, out_path, class_name, norm_ratio,
                        raw_support_dict, total_inst, norm_support_dict):
        with open(out_path, "w") as f:
            f.write(f"Class: {class_name}\n")
            f.write(f"Normalization ratio (cases/controls): {norm_ratio:.6f}\n\n")

            # Raw
            f.write("Raw prefix supports (before total-instances division, but after normalization)\n")
            f.write("#\tPrefix\t\tRawSupport\n")
            for i, (p, val) in enumerate(raw_support_dict.items()):
                f.write(f"{i}\t{p}\t{val}\n")

            f.write(f"\nTotal pseudo-instances (Σ raw):\t{total_inst}\n\n")

            # Normalized
            f.write("Normalized prefix supports (after division)\n")
            f.write("#\tPrefix\t\tNormalizedSupport\n")
            for i, (p, val) in enumerate(norm_support_dict.items()):
                f.write(f"{i}\t{p}\t{val:.6f}\n")
