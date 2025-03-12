import decimal
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.elasticity_generator.postprocessing.postprocessing import (
    add_columns,
    encode_categorical_as_idx,
    get_elasticity_fit_pivot_table,
    tag_metadata_details,
)
from src.elasticity_generator.postprocessing.shap_classes import SHAPUtils
from src.elasticity_generator.utils.utils_experiment_tracking import (
    ExperimentUtils,
    ExperimentUtilsDecorator,
)
from src.utils.io import move_files_to_new_folder

# Global config for postprocessing
postprocessing_config = yaml.safe_load(
    open("conf/default/parameters_postprocess.yml", "r")
)


def interpret_model(
    df_train: pd.DataFrame,
    categorical_columns: List[str],
    model_dict: Dict,
    model_output_shap_folder_path: str,
    target_column: str,
    pred_columns: List[str],
    experiment_utils: ExperimentUtils,
) -> None:
    """
    Interpret the model using SHAP values.

    Args:
        df_train: Training DataFrame.
        categorical_columns: List of categorical columns.
        model_dict: Dictionary containing the trained model.
        model_output_shap_folder_path: Path to the folder where
        SHAP figures will be saved.
        target_column: The target column for the model.
        pred_columns: List of predictor columns.

    """
    df_shap = encode_categorical_as_idx(df_train, categorical_columns)
    shaputils = SHAPUtils(
        model_dict["model_fitted"]["model"],
        df_shap,
        df_shap[target_column],
        pred_columns,
        experiment_utils,
    )
    shaputils.plot_bar(save_folder=model_output_shap_folder_path)
    shaputils.plot_beeswarm(save_folder=model_output_shap_folder_path)


def pivot_to_parquet(df_pivot: pd.DataFrame, path: str) -> None:
    """
    Cast pivot table column types to str before saving as parquet

    Args:
        df_pivot: Dataframe pivot table
        path: Path to save location
    """
    df_pivot.columns = df_pivot.columns.astype(str)
    df_pivot.to_parquet(path)


@ExperimentUtilsDecorator.log_elasticity_decorator
def postprocess_data(
    df_elasticity: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    filters: Dict,
    columns_to_add: List[str],
    renaming_dict: Dict,
    elasticity_metadata_dict: Dict,
    experiment_utils: ExperimentUtils,
) -> None:
    """
    Postprocess the data after model training and evaluation.

    Args:
        df_elasticity: DataFrame containing elasticity information.
        df_test_raw: Raw testing DataFrame.
        filters: Dictionary containing filters to be applied on the data.
        columns_to_add: List of columns to be added to the DataFrame.
        renaming_dict: Dictionary containing column renaming information.
        elasticity_metadata_dict: Dictionary containing PIN dates
                                  metadata and experiment metadata
    """
    df_elasticity = add_columns(
        df_output=df_elasticity,
        df_source=df_test_raw,
        columns_to_add=columns_to_add,
        filters=filters,
    )

    # Generate pivot tables
    pivot_table = df_elasticity.groupby("key_cutoff_date").apply(
        lambda x: get_elasticity_fit_pivot_table(x)
    )

    pivot_table_series = df_elasticity.groupby("fea_model").apply(
        lambda x: get_elasticity_fit_pivot_table(x)
    )

    pivot_table_channel = df_elasticity.groupby("fea_type_of_sale").apply(
        lambda x: get_elasticity_fit_pivot_table(x)
    )

    total_elasticity = (
        df_elasticity.groupby(["key_cutoff_date", "fea_model"])
        .agg(
            {
                "cftp_for_incentive=0": "mean",
                "cftp_for_incentive=500": "mean",
                "predicted_volume_for_incentive=0": "sum",
                "predicted_volume_for_incentive=500": "sum",
            }
        )
        .reset_index()
    )
    total_elasticity["elasticity"] = np.abs(
        (
            total_elasticity["predicted_volume_for_incentive=500"]
            / total_elasticity["predicted_volume_for_incentive=0"]
            - 1
        )
        / (
            (
                total_elasticity["cftp_for_incentive=500"]
                / total_elasticity["cftp_for_incentive=0"]
                - 1
            )
        )
    )
    total_elasticity["fea_type_of_sale"] = "all_type_of_sale"
    df_elasticity_all = pd.concat([df_elasticity, total_elasticity])
    renaming_dict.update(
        {
            "cftp_for_incentive=0": "cftp",
            "predicted_volume_for_incentive=0": "predicted_volume_cftp",
        }
    )
    df_elasticity_all = df_elasticity_all.rename(columns=renaming_dict)

    columns_to_output = [
        "fea_type_of_sale",
        "fea_model",
        "key_cutoff_date",
        "elasticity",
        "elasticity_for_incentive=500",
        "elasticity_for_incentive=1000",
        "elasticity_for_incentive=1500",
        "cftp",
        "predicted_volume_cftp",
        "actual_total_volume",
        "actual_channel_volume",
        "pre_incentive_price",
        "incentive",
        "model_volume_prediction",
        "base_volume_prediction",
        "base_volume_curve",
        "elasticity_err",
        "fit_r2",
        "fea_comp_top_3_avg_comp_nameplate_channel_specific_incentive_per_unit",
        "fea_all_channel_ground_stock_count",
        "fea_prev_month_toyota_market_share",
        "fea_all_channel_dealer_stock_count",
    ]

    # Select the columns and create an output
    df_elasticity_all = df_elasticity_all[columns_to_output]
    df_elasticity_all = df_elasticity_all.applymap(
        lambda x: round(x, 2) if type(x) in [float, decimal.Decimal] else x
    )
    df_elasticity_all_dup = df_elasticity_all.copy()
    df_elasticity_all["helper_deduplication"] = "keep"
    df_elasticity_all["incentive_bound"] = "low"
    df_elasticity_all["incentive_range"] = 0
    df_elasticity_all_dup["helper_deduplication"] = "don't keep"
    df_elasticity_all_dup["incentive_bound"] = "high"
    df_elasticity_all_dup["incentive_range"] = 6000
    df_elasticity_all = pd.concat([df_elasticity_all, df_elasticity_all_dup])

    df_elasticity_all = tag_metadata_details(
        df_elasticity_all, elasticity_metadata_dict
    )

    return df_elasticity_all, pivot_table, pivot_table_series, pivot_table_channel


def manage_output_artifacts(base_folder: str, files_to_merge_conf: Dict):
    """Organizes output artifacts by moving files into folders based on a configuration.

    Args:
    base_folder: The base directory where the files are located.
    files_to_merge_conf: A dictionary where keys are folder names and
    values are lists of files to move to those folders.
    """
    for folder_name, files in files_to_merge_conf.items():
        move_files_to_new_folder(base_folder, files, folder_name)


def combine_eval_metrics(
    bias: Dict, error_std: Dict, sigma: Dict, train_set: bool = True
):
    """Combines evaluation metrics (bias, error standard deviation, sigma)
    into a Pandas DataFrame.

    Args:
    bias: A dictionary containing bias values.
    Keys are tuples (fea_type_of_sale, fea_model).
    error_std: A dictionary containing error standard deviation values.
    Keys are tuples (fea_type_of_sale, fea_model).
    sigma: A dictionary containing sigma values.
    Keys are tuples (fea_type_of_sale, fea_model).
    train_set: A boolean indicating whether the data is from the training set.

    Returns:
    A Pandas DataFrame containing the combined evaluation metrics.
    """
    try:
        bias_df = pd.DataFrame(bias).reset_index()
        # print(bias_df)

        error_std_df = pd.DataFrame()
        for col, value in error_std.items():
            error_std_df = pd.concat(
                [
                    error_std_df,
                    pd.DataFrame(
                        {
                            "fea_type_of_sale": col[0],
                            "fea_model": col[1],
                            "error_std": value,
                        },
                        index=[0],
                    ),
                ]
            )
        error_std_df.reset_index(drop=True, inplace=True)
        # print(error_std_df)

        sigma_df = pd.DataFrame()
        for col, value in sigma.items():
            sigma_df = pd.concat(
                [
                    sigma_df,
                    pd.DataFrame(
                        {
                            "fea_type_of_sale": col[0],
                            "fea_model": col[1],
                            "sigma": value,
                        },
                        index=[0],
                    ),
                ]
            )
        sigma_df.reset_index(drop=True, inplace=True)
        # print(sigma_df)

        eval_metrics_df = (
            bias_df.set_index(["fea_type_of_sale", "fea_model"])
            .join(
                error_std_df.set_index(["fea_type_of_sale", "fea_model"]), how="inner"
            )
            .join(sigma_df.set_index(["fea_type_of_sale", "fea_model"]), how="inner")
            .reset_index()
        )

        eval_metrics_df["train_set"] = train_set
        return eval_metrics_df
    except Exception as e:
        print(str(e))
        return pd.DataFrame(
            {
                "fea_type_of_sale": None,
                "fea_model": None,
                "bias": None,
                "error_std": None,
                "sigma": None,
                "train_set": train_set,
            },
            index=[0],
        ).reset_index(drop=True)


def consolidate_deliveries(delivery_df, data_catalog, delivered_output_folder):
    """
    Remake elasticities_consolidated.csv file

    Args:
        delivery_df: Dataframe with 1 row for each previous month
        data_catalog: Dict of data paths
        delivered_output_folder: Path of delivery folder
    """
    elasticities_delivered_path = (
        f"{delivered_output_folder}" "elasticities_consolidated.csv"
    )
    column_names = [
        "fea_type_of_sale",
        "fea_model",
        "key_cutoff_date",
        "elasticity",
        "elasticity_for_incentive=500",
        "elasticity_for_incentive=1000",
        "elasticity_for_incentive=1500",
        "cftp",
        "predicted_volume_cftp",
        "actual_total_volume",
        "actual_channel_volume",
        "pre_incentive_price",
        "incentive",
        "model_volume_prediction",
        "base_volume_prediction",
        "base_volume_curve",
        "elasticity_err",
        "fit_r2",
        "fea_comp_top_3_avg_comp_nameplate_channel_specific_incentive_per_unit",
        "fea_all_channel_ground_stock_count",
        "fea_prev_month_toyota_market_share",
        "fea_all_channel_dealer_stock_count",
        "helper_deduplication",
        "incentive_bound",
        "incentive_range",
        "sagemaker_experiment_name",
        "sagemaker_run_name",
        "current_branch_name",
        "sagemaker_experiment_folder",
        "data_until",
        "elasticities_as_of",
    ]
    elasticities_delivered = pd.DataFrame(columns=column_names)

    for index, row in delivery_df.iterrows():
        experiment_name = row["Experiment name"]
        run_name = row["Run name"]
        this_elasticities_full = pd.read_csv(
            f"{data_catalog['experiments_folder']['file_path']}/"
            f"{experiment_name}-"
            f"{run_name}/elasticities_full.csv"
        )

        elasticities_delivered = pd.concat(
            [elasticities_delivered, this_elasticities_full]
        )
    elasticities_delivered.to_csv(elasticities_delivered_path, index=False)
    return elasticities_delivered


def smooth_elasticities(
    elasticities_delivered: pd.DataFrame,
) -> None:
    """
    Smooth elasticities considering delivery elasticities

    Args:
        elasticities_delivered: Dataframe with all delivered elasticities
                                plus this month's elasticities
    """
    elasticities_delivered["elasticity"] = elasticities_delivered["elasticity"].astype(
        float
    )
    elasticities_delivered["elasticities_as_of"] = pd.to_datetime(
        elasticities_delivered["elasticities_as_of"]
    )
    elasticities_delivered["key_cutoff_date"] = pd.to_datetime(
        elasticities_delivered["key_cutoff_date"]
    )
    series_list = elasticities_delivered["fea_model"].drop_duplicates()
    type_of_sale_list = elasticities_delivered["fea_type_of_sale"].drop_duplicates()
    delivery_list = elasticities_delivered["elasticities_as_of"].drop_duplicates()
    # Initialize smoothed elasticities
    elasticities_delivered["elasticity_smooth"] = np.nan
    # Initialize flag for which elasticities were delivered
    elasticities_delivered["carry_forward"] = False
    for series in series_list:
        for type_of_sale in type_of_sale_list:
            for i_delivery in range(len(delivery_list)):
                delivery = delivery_list.values[i_delivery]
                # Get slice of elasticities by series/type/delivery date
                smooth_slice = (
                    (elasticities_delivered["helper_deduplication"] == "keep")
                    & (elasticities_delivered["fea_model"] == series)
                    & (elasticities_delivered["fea_type_of_sale"] == type_of_sale)
                    & (elasticities_delivered["elasticities_as_of"] == delivery)
                )
                elasticities_delivered_this = elasticities_delivered[smooth_slice]
                if len(elasticities_delivered_this) > 0:
                    if i_delivery > 0:
                        # Get delivered elasticities from previous month
                        delivery_m1 = delivery_list.values[i_delivery - 1]
                        carryforwardslice1 = (
                            (elasticities_delivered["helper_deduplication"] == "keep")
                            & (elasticities_delivered["fea_model"] == series)
                            & (
                                elasticities_delivered["fea_type_of_sale"]
                                == type_of_sale
                            )
                            & (
                                elasticities_delivered["elasticities_as_of"]
                                == delivery_m1
                            )
                            & (elasticities_delivered["carry_forward"])
                        )
                        if (
                            len(
                                elasticities_delivered[carryforwardslice1][
                                    "elasticity_smooth"
                                ]
                            )
                            > 0
                        ):
                            carried_forward_elasticity1 = elasticities_delivered[
                                carryforwardslice1
                            ]["elasticity_smooth"].values[0]
                        else:
                            carried_forward_elasticity1 = np.nan
                    else:
                        carried_forward_elasticity1 = np.nan
                    if i_delivery > 1:
                        # Get delivered elasticities from 2 month's ago
                        delivery_m2 = delivery_list.values[i_delivery - 2]
                        carryforwardslice2 = (
                            (elasticities_delivered["helper_deduplication"] == "keep")
                            & (elasticities_delivered["fea_model"] == series)
                            & (
                                elasticities_delivered["fea_type_of_sale"]
                                == type_of_sale
                            )
                            & (
                                elasticities_delivered["elasticities_as_of"]
                                == delivery_m2
                            )
                            & (elasticities_delivered["carry_forward"])
                        )
                        if (
                            len(
                                elasticities_delivered[carryforwardslice2][
                                    "elasticity_smooth"
                                ]
                            )
                            > 0
                        ):
                            carried_forward_elasticity2 = elasticities_delivered[
                                carryforwardslice2
                            ]["elasticity_smooth"].values[0]
                        else:
                            carried_forward_elasticity2 = np.nan
                    else:
                        carried_forward_elasticity2 = np.nan
                    elasticities_delivered_this.iloc[
                        0,
                        elasticities_delivered_this.columns.get_loc(
                            "elasticity_smooth"
                        ),
                    ] = carried_forward_elasticity2
                    elasticities_delivered_this.iloc[
                        1,
                        elasticities_delivered_this.columns.get_loc(
                            "elasticity_smooth"
                        ),
                    ] = carried_forward_elasticity1
                    for i in range(2, len(elasticities_delivered_this)):
                        elasticities_delivered_this.iloc[
                            i,
                            elasticities_delivered_this.columns.get_loc(
                                "elasticity_smooth"
                            ),
                        ] = np.nanmean(
                            np.append(
                                elasticities_delivered_this.iloc[i - 2 : i][
                                    "elasticity_smooth"
                                ].values,
                                elasticities_delivered_this.iloc[
                                    i,
                                    elasticities_delivered_this.columns.get_loc(
                                        "elasticity"
                                    ),
                                ],
                            )
                        )
                    cols = [
                        "fea_type_of_sale",
                        "fea_model",
                        "key_cutoff_date",
                        "elasticity",
                        "elasticity_smooth",
                        "elasticities_as_of",
                    ]
                    keys = [
                        "fea_type_of_sale",
                        "fea_model",
                        "key_cutoff_date",
                        "elasticity",
                        "elasticities_as_of",
                    ]
                    # display(elasticities_delivered_this[cols])
                    elasticities_delivered = elasticities_delivered.merge(
                        elasticities_delivered_this[cols], on=keys, how="left"
                    )
                    elasticities_delivered[
                        "elasticity_smooth_x"
                    ] = elasticities_delivered["elasticity_smooth_x"].combine_first(
                        elasticities_delivered["elasticity_smooth_y"]
                    )
                    elasticities_delivered = elasticities_delivered.drop(
                        "elasticity_smooth_y", axis=1
                    )
                    elasticities_delivered = elasticities_delivered.rename(
                        columns={"elasticity_smooth_x": "elasticity_smooth"}
                    )
                    # Persist current month's smoothed elasticities
                    #check array length to handle index error
                    if elasticities_delivered_this.shape[0] > 2:
                        carry_forward_slice = (
                        (elasticities_delivered["helper_deduplication"] == "keep")
                        & (elasticities_delivered["fea_model"] == series)
                        & (elasticities_delivered["fea_type_of_sale"] == type_of_sale)
                        & (elasticities_delivered["elasticities_as_of"] == delivery)
                        & (
                            elasticities_delivered["key_cutoff_date"]
                            == elasticities_delivered_this.iloc[2]["key_cutoff_date"]
                        )
                    )
                    else:
                        carry_forward_slice = (
                        (elasticities_delivered["helper_deduplication"] == "keep")
                        & (elasticities_delivered["fea_model"] == series)
                        & (elasticities_delivered["fea_type_of_sale"] == type_of_sale)
                        & (elasticities_delivered["elasticities_as_of"] == delivery)
                    )    
                    
                    elasticities_delivered.loc[
                        carry_forward_slice, "carry_forward"
                    ] = True

    elasticities_delivered["smoothed"] = "No"
    elasticities_delivered_smoothonly = elasticities_delivered.copy()
    elasticities_delivered_smoothonly["elasticity"] = elasticities_delivered_smoothonly[
        "elasticity_smooth"
    ]
    elasticities_delivered_smoothonly["smoothed"] = "Yes"
    elasticities_delivered_all = pd.concat(
        [elasticities_delivered, elasticities_delivered_smoothonly]
    ).reset_index(drop=True)
    elasticities_delivered_all = elasticities_delivered_all.drop(
        ["elasticity_smooth", "carry_forward"], axis=1
    )
    latest_delivery = elasticities_delivered_all["elasticities_as_of"].max()
    return elasticities_delivered_all


def map_series_rundown(df: pd.DataFrame, series_col="fea_model"):
    """
    Map series to be consistent with series names in rundown

    Args:
        df: Dataframe with series to be mapped
        series_col: Column to be mapped
    """
    series_map = postprocessing_config["rundown_series_mapping"]
    df[series_col] = df[series_col].map(series_map).fillna(df[series_col])
    return df
