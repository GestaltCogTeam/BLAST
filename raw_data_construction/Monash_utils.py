import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from distutils.util import strtobool

# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    if "web_traffic_extended_dataset_without_missing_values" in full_file_path_and_name:
        encoding = "utf-8"
    else:
        encoding = "cp1252"
    with open(full_file_path_and_name, "r", encoding=encoding) as file:
    # with open(full_file_path_and_name, "r") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )

def get_2d_values_unequal_length(nested_list):
    max_len = max(len(sublist) for sublist in nested_list)
    matrix = np.full((len(nested_list), max_len), np.nan)
    for i, sublist in enumerate(nested_list):
        matrix[i, :len(sublist)] = sublist
    return matrix

def get_2d_time_stamps_unequal_length(nested_list):
    max_len = max(len(sublist) for sublist in nested_list)
    matrix = np.empty((len(nested_list), max_len), dtype='<M8[ns]')
    for i, sublist in enumerate(nested_list):
        matrix[i, :len(sublist)] = sublist
    return matrix

def get_frequency(frequency):
    if frequency is None: return None
    if frequency == "half_hourly": return "30min"
    freq_map = {"hourly": "h", "monthly": "M", "seconds": "s", "minutes": "min", "minutely": "min", "daily": "D", "weekly": "W", "yearly": "Y", "quarterly": "Q"}

    # preprocess frequency
    if "_" in frequency:
        count = frequency.split("_")[0]
        frequency = frequency.split("_")[1]
    else:
        count = "1"

    freq = count + freq_map[frequency]

    return freq

def get_monash_time_stamsp_and_values(data_path):

    loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(data_path)
    contain_date = "start_timestamp" in loaded_data.columns
    freq = get_frequency(frequency)
    meta = [frequency, forecast_horizon, contain_missing_values,contain_equal_length]
    # get all series names
    series_names = loaded_data['series_name'].unique()
    # get values and time stamps
    values = []
    if contain_date: time_stamps = []
    for name in tqdm(series_names):
        value = loaded_data.loc[loaded_data['series_name'] == name]["series_value"].values[0].to_numpy()
        values.append(value)
        if contain_date:
            start_time = loaded_data.loc[loaded_data['series_name'] == name]["start_timestamp"].values[0]
            #print(start_time)

            if pd.Timestamp(start_time).year >= 1677:
                if freq != '1Y':
                    time_stamp = pd.date_range(start=start_time, periods=len(value), freq=freq).to_numpy()
                    time_stamps.append(time_stamp)
                elif pd.Timestamp(start_time).year + len(value) <= 2262:
                    time_stamp = pd.date_range(start=start_time, periods=len(value), freq=freq).to_numpy()
                    time_stamps.append(time_stamp)


    # contain_equal_length
    if contain_equal_length:
        values = np.array(values).T # [N, L] -> [L, N]
        if contain_date: time_stamps = np.array(time_stamps).T # [N, L] -> [L, N]
    else:
        values = get_2d_values_unequal_length(values).T # [L, N]
        if contain_date: time_stamps = get_2d_time_stamps_unequal_length(time_stamps).T
    if contain_date: return time_stamps, values, meta
    else: return None, values, meta
