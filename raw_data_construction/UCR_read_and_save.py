import os
import pyarrow as pa
import numpy as np
import json
from datetime import datetime
import re

def load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_meta_data=False,
    return_type="auto",
):
    """Load time series .ts file into X and (optionally) y.

    Parameters
    ----------
    full_file_path_and_name : string
        full path of the file to load, .ts extension is assumed.
    replace_missing_vals_with : string, default="NaN"
        issing values in the file are replaces with this value
    return_meta_data : boolean, default=False
        return a dictionary with the meta data loaded from the file
    return_type : string, default = "auto"
        data type to convert to.
        If "auto", returns numpy3D for equal length and list of numpy2D for unequal.
        If "numpy2D", will squash a univariate equal length into a numpy2D (n_cases,
        n_timepoints). Other options are available but not supported medium term.

    Returns
    -------
    data: Union[np.ndarray,list]
        time series data, np.ndarray (n_cases, n_channels, series_length) if equal
        length time series, list of [n_cases] np.ndarray (n_channels, n_timepoints)
        if unequal length series.
    y : target variable, np.ndarray of string or int
    meta_data : dict (optional).
        dictionary of characteristics, with keys
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    Raises
    ------
    IOError if the load fails.
    """
    # Check file ends in .ts, if not, insert
    if not full_file_path_and_name.endswith(".ts"):
        full_file_path_and_name = full_file_path_and_name + ".ts"
    # Open file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        # Read in headers
        meta_data = _load_header_info(file)
        # load into list of numpy
        data, y, meta_data = _load_data(file, meta_data)

    # if equal load to 3D numpy
    if meta_data["equallength"]:
        data = np.array(data)
        if return_type == "numpy2D" and meta_data["univariate"]:
            data = data.squeeze()
    # If regression problem, convert y to float
    if meta_data["targetlabel"]:
        y = y.astype(float)

    return (data, y, meta_data) if return_meta_data else (data, y)


def _load_header_info(file):
    """Load the meta data from a .ts file and advance file to the data.

    Parameters
    ----------
    file : stream.
        input file to read header from, assumed to be just opened

    Returns
    -------
    meta_data : dict.
        dictionary with the data characteristics stored in the header.
    """
    meta_data = {
        "problemname": "none",
        "timestamps": False,
        "missing": False,
        "univariate": True,
        "equallength": True,
        "classlabel": True,
        "targetlabel": False,
        "class_values": [],
    }
    boolean_keys = ["timestamps", "missing", "univariate", "equallength", "targetlabel"]
    for line in file:
        line = line.strip().lower()
        if line and not line.startswith("#"):
            tokens = line.split(" ")
            token_len = len(tokens)
            key = tokens[0][1:]
            if key == "data":
                if line != "@data":
                    raise IOError("data tag should not have an associated value")
                return meta_data
            if key in meta_data.keys():
                if key in boolean_keys:
                    if token_len != 2:
                        raise IOError(f"{tokens[0]} tag requires a boolean value")
                    if tokens[1] == "true":
                        meta_data[key] = True
                    elif tokens[1] == "false":
                        meta_data[key] = False
                elif key == "problemname":
                    meta_data[key] = tokens[1]
                elif key == "classlabel":
                    if tokens[1] == "true":
                        meta_data["classlabel"] = True
                        if token_len == 2:
                            raise IOError(
                                "if the classlabel tag is true then class values "
                                "must be supplied"
                            )
                    elif tokens[1] == "false":
                        meta_data["classlabel"] = False
                    else:
                        raise IOError("invalid class label value")
                    meta_data["class_values"] = [token.strip() for token in tokens[2:]]
        if meta_data["targetlabel"]:
            meta_data["classlabel"] = False
    return meta_data


def _load_data(file, meta_data, replace_missing_vals_with="NaN"):
    """Load data from a file with no header.

    this assumes each time series has the same number of channels, but allows unequal
    length series between cases.

    Parameters
    ----------
    file : stream, input file to read data from, assume no comments or header info
    meta_data : dict.
        with meta data in the file header loaded with _load_header_info

    Returns
    -------
    data: list[np.ndarray].
        list of numpy arrays of floats: the time series
    y_values : np.ndarray.
        numpy array of strings: the class/target variable values
    meta_data :  dict.
        dictionary of characteristics enhanced with number of channels and series length
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    """
    data = []
    n_cases = 0
    n_channels = 0  # Assumed the same for all
    current_channels = 0
    series_length = 0
    y_values = []
    for line in file:
        line = line.strip().lower()
        line = line.replace("?", replace_missing_vals_with)
        channels = line.split(":")
        n_cases += 1
        current_channels = len(channels)
        if meta_data["classlabel"] or meta_data["targetlabel"]:
            current_channels -= 1
        if n_cases == 1:  # Find n_channels and length  from first if not unequal
            n_channels = current_channels
            if meta_data["equallength"]:
                series_length = len(channels[0].split(","))
        else:
            if current_channels != n_channels:
                raise IOError(
                    f"Inconsistent number of dimensions in case {n_cases}. "
                    f"Expecting {n_channels} but have read {current_channels}"
                )
            if meta_data["univariate"]:
                if current_channels > 1:
                    raise IOError(
                        f"Seen {current_channels} in case {n_cases}."
                        f"Expecting univariate from meta data"
                    )
        if meta_data["equallength"]:
            current_length = series_length
        else:
            current_length = len(channels[0].split(","))
        np_case = np.zeros(shape=(n_channels, current_length))
        for i in range(0, n_channels):
            single_channel = channels[i].strip()
            data_series = single_channel.split(",")
            data_series = [float(x) for x in data_series]

            if len(data_series) != current_length:
                #print(len(data_series), current_length)
                if len(data_series) >  current_length:
                    data_series = data_series[: current_length]
                else:
                    data_series = data_series + [np.nan] * (current_length - len(data_series))
                '''raise IOError(
                    f"Unequal length series, in case {n_cases} meta "
                    f"data specifies all equal {series_length} but saw "
                    f"{len(single_channel)}"
                )'''

            np_case[i] = np.array(data_series)
        data.append(np_case)
        if meta_data["classlabel"] or meta_data["targetlabel"]:
            y_values.append(channels[n_channels])
    if meta_data["equallength"]:
        data = np.array(data)
    return data, np.asarray(y_values), meta_data

def format_dataset_name(name):
    # 移除包含"monash"或"Monash"的字段
    name = re.sub(r'monash', '', name, flags=re.IGNORECASE)

    # 去除开头的下划线
    name = re.sub(r'^_+', '', name)
    # 处理连续大写的情况，将其转换为首字母大写，其余小写
    name = re.sub(r'([A-Z]+)', lambda m: m.group(0).capitalize(), name)

    # 使用正则表达式在每个大写字母前添加下划线，但不在数字之间添加下划线
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name)

    # 将所有字母转换为小写
    name = name.lower()

    # 将首字母和每个下划线后的第一个字母转换为大写
    formatted_name = '_'.join([word.capitalize() if not word.isdigit() else word for word in name.split('_')])

    # 处理数字之间不添加下划线的情况
    formatted_name = re.sub(r'(\d)_+(\d)', r'\1\2', formatted_name)
    formatted_name = re.sub(r'_{2,}', '_', formatted_name)

    return formatted_name

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义数据集文件夹路径
datasets_folder = os.path.join(current_dir, '..', 'datasets/raw_datasets/Timeseries-PILE/classification/UCR')
output_folder = os.path.join(current_dir, '..', 'datasets/processed_datasets/UCR')
count = 0
total = len(os.listdir(datasets_folder))
# 遍历数据集文件夹
for dataset_name in os.listdir(datasets_folder):
    count += 1
    dataset_path = os.path.join(datasets_folder, dataset_name)
    if os.path.isdir(dataset_path):
        # 创建输出文件夹
        dataset_name = format_dataset_name(dataset_name)
        print(f"Processing dataset {count}/{total}: {dataset_name}")
        output_dataset_folder = os.path.join(output_folder, dataset_name)
        os.makedirs(output_dataset_folder, exist_ok=True)

        # 初始化shape信息
        shapes_info = {}

        # 特殊的字符串labels处理
        dic = {}
        label_num = 0
        str_label = False
        # 遍历Arrow文件
        for file_name in os.listdir(dataset_path):

            if file_name.endswith('.ts'):
                file_path = os.path.join(dataset_path, file_name)
                #print(file_name)
                #print(file_path)
                data, labels, meta = load_from_tsfile(
                    file_path, return_meta_data=True
                )

                #print(data.shape, labels.shape)
                #print(meta)
                #print(labels)
                # 判断是训练集还是测试集
                if 'TRAIN' in file_name:
                    split = 'TRAIN'
                elif 'TEST' in file_name:
                    split = 'TEST'
                else:
                    continue
                meta['phase'] = split
                # print(labels.shape)
                # print(labels)
                if isinstance(labels[0], str) and not labels[0].isdigit():
                    for i in set(labels):
                        if i not in dic.keys():
                            dic[i] = label_num
                            label_num += 1
                    # 使用字典将字符串映射到数字
                    labels = np.array([dic[item] for item in labels])
                    str_label = True
                else:
                    labels = labels.astype(int)

                if labels.ndim == 1:
                    labels = np.expand_dims(labels, axis=1)
                '''if True in np.isnan(data):
                    pass
                else:
                    data[data == 0] = np.nan'''

                if not meta['equallength']:
                    L = max([seq.shape[1] for seq in data])
                    for i, seq in enumerate(data):
                        data[i] = np.pad(seq, ((0, 0), (0, L - seq.shape[1])), 'constant',
                                           constant_values=np.nan)  # 行维度不padding，列维度左侧不padding，右侧padding(L - seq.shape[1])列
                    data = np.array(data)

                data_count_non_nan = np.sum(~np.isnan(data))

                N, C, L = data.shape
                
                # 初始化结果数组
                valid_lengths = np.zeros((N, C, 1), dtype=int)
                valid_indices = np.zeros((N, C, 2), dtype=int)

                # 计算有效长度和索引
                for n in range(N):
                    for c in range(C):
                        non_nan_indices = np.where(~np.isnan(data[n, c]))[0]
                        if non_nan_indices.size > 0:
                            first_non_nan = non_nan_indices[0]
                            last_non_nan = non_nan_indices[-1]
                            valid_lengths[n, c, 0] = last_non_nan - first_non_nan + 1
                            valid_indices[n, c, 0] = first_non_nan
                            valid_indices[n, c, 1] = last_non_nan


                # 添加 data 和 label 的形状信息到 meta 字典
                #meta['str_label'] = str_label
                if str_label:
                    meta['labels_str'] = dic
                meta['data_shape'] = data.shape
                meta['label_shape'] = labels.shape
                meta['data_valid_value'] = int(data_count_non_nan)
                meta["min_length"] = int(valid_lengths.min())
                meta['max_length'] = int(valid_lengths.max())
                meta["mean_length"] = float(valid_lengths.mean())
                if split == 'TRAIN':
                    meta['label_min'] = int(labels.min())
                    meta['label_max'] = int(labels.max())
                    meta['label_number'] = int(len(np.unique(labels)))
                meta["valid_lengths"] = valid_lengths.tolist()
                meta["valid_indices"] = valid_indices.tolist()
                #print(labels.shape)
                #print(data.shape, labels.shape)
                # 保存数据和标签为 .npy 文件
                np.save(os.path.join(output_dataset_folder, f'{dataset_name}_{split}_data.npy'), data)
                np.save(os.path.join(output_dataset_folder, f'{dataset_name}_{split}_label.npy'), labels)
                with open(os.path.join(output_dataset_folder, f'{dataset_name}_{split}.json'), 'w') as json_file:
                    json.dump(meta, json_file, indent=4)
                #print(f'{dataset_name} {split} 数据已处理并保存。')
