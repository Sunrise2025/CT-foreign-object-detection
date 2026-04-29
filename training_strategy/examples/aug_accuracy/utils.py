from configparser import ConfigParser
import numpy as np
from sklearn.metrics import accuracy_score


def split_prediction(y_pred, y_true, obj_proj):
    """Compare class predictions for images corresponding to different objects
    """
    total_num = y_pred.shape[0]
    num_obj = total_num // obj_proj
    assert total_num % obj_proj == 0

    class_seq = []
    for i in range(num_obj):
        counter = 0
        for j in range(obj_proj):
            if y_pred[i * obj_proj + j] == y_true[i * obj_proj + j]:
                counter += 1
        class_seq.append(counter)

    return "[" + ",".join(str(el) for el in class_seq) + "]"


def prediction_per_object(y_pred, y_true, num_classes, obj_proj):
    total_num = y_pred.shape[0]
    assert total_num % obj_proj == 0

    for i in range(total_num // obj_proj):
        counter = np.zeros((num_classes))
        for j in range(obj_proj):
            counter[y_pred[i * obj_proj + j]] += 1
        print(i, y_true[i * obj_proj], counter)

def compute_accuracy(y_pred, y_true):
    return accuracy_score(y_true, y_pred)

def read_config(fname, encoding='utf-8'):
    parser = ConfigParser()
    try:
        with open(fname, 'r', encoding=encoding) as f:
            parser.read_file(f)
        print(f"成功读取配置文件 {fname} 使用编码 {encoding}")
    except UnicodeDecodeError as e:
        print(f"尝试使用 {encoding} 编码读取 {fname} 失败: {e}")
        if encoding == 'utf-8':
            print("尝试 gbk 编码...")
            return read_config(fname, encoding='gbk')
        else:
            print("尝试其他编码失败，请检查文件编码。")
            raise
    # 继续处理配置文件...
    if not parser.has_section('General'):
        raise ValueError("Configuration file must have a 'General' section.")

    config = {s: dict(parser.items(s)) for s in parser.sections()}

    data_dict_keys = list(config.keys())
    if 'General' in data_dict_keys:
        data_dict_keys.remove('General')

    def get_int_value(section, option, default):
        try:
            return int(parser.get(section, option))
        except (ValueError, TypeError):
            return default

    def get_boolean_value(section, option, default):
        try:
            return parser.getboolean(section, option)
        except (ValueError, TypeError):
            return default

    config['General']['batch_size'] = get_int_value('General', 'batch_size', -1)
    config['General']['max_epochs'] = get_int_value('General', 'max_epochs', -1)
    config['General']['use_deterministic'] = get_boolean_value('General', 'use_deterministic', False)
    for key in data_dict_keys:
        config[key]['c_in'] = get_int_value(key, 'c_in', -1)
        config[key]['c_out'] = get_int_value(key, 'c_out', -1)
        config[key]['img_per_obj'] = get_int_value(key, 'img_per_obj', -1)

    return config

def get_available_data_types(config):
    data_dict_keys = list(config.keys())
    data_dict_keys.remove('General')
    return data_dict_keys
