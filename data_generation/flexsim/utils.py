from typing import Dict

import numpy as np
from pathlib import Path
from configparser import ConfigParser
import cupy
import cupyx.scipy.ndimage
import imageio.v2 as imageio
  # 确保你已经导入了imageio

def get_volume_properties(obj_folder):
    '''Gets volume properties.
    :param obj_folder: Path to the folder containing slices of the segmentation.
    :type obj_folder: :class:`pathlib.Path`
    :return: Object shape (height, width, width)
    :rtype: :class:`np.ndarray`
    '''
    height = len(list(obj_folder.glob("*.tiff")))    # 获取文件夹中所有 TIFF 格式切片文件的数量，表示对象体积的高度
    sl = imageio.imread(obj_folder / "slice_{:06d}.tiff".format(0))    # 读取第一个切片图像以获取对象体积的宽度和长度（假设每个切片图像的宽度和深度相同）
    obj_shape = (height, *sl.shape)    # 创建一个包含对象体积形状的元组 (height, width, width)
    return obj_shape    # 返回对象体积的形状，类型为 (height, width, width)



def read_volume(obj_folder):
    '''Reads slices from the folder and creates np.ndarray containing the object volume

    :param obj_folder: Path to the folder containing slices of the segmentation.
    :type obj_folder: :class:`pathlib.Path`
    :return: Object volume
    :rtype: :class:`np.ndarray`

    '''
    obj_shape = get_volume_properties(obj_folder) # 获取对象体积的属性，例如形状等
    vol = np.zeros(obj_shape)    # 根据对象体积的形状初始化一个全零的 numpy 数组
    for i in range(obj_shape[0]):    # 遍历文件夹中的每个切片文件
        vol[i, :] = imageio.imread(obj_folder / "slice_{:06d}.tiff".format(i))
        # 使用 imageio 读取当前切片的图像，并存储在对应的卷积矩阵位置
    return vol    # 返回对象体积的 numpy 数组



def read_config(fname):
    """Reads the config and converts strings with numbers into number types
    :param fname: Path to the .ini config flie
    :type fname: :class:`string`
    :return: Dictionary of configuration parameters
    :rtype: class:`dict`
    """
    # parser = ConfigParser()
    # parser.read(fname)
    # config = {s: dict(parser.items(s)) for s in parser.sections()}
    # sim_config = config['Simulation']

    parser = ConfigParser()

    # 使用 open() 以 UTF-8 编码方式打开文件
    with open(fname, encoding='utf-8') as f:
        parser.read_file(f)  # 使用 read_file() 读取内容

    config = {s: dict(parser.items(s)) for s in parser.sections()}
    sim_config = config['Simulation']

    # 安全转换为整数
    def safe_int(value, default):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    # 安全转换为浮点数
    def safe_float(value, default):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # Simulation配置
    sim_config['num_angles'] = safe_int(parser['Simulation'].get('num_angles', '1'), 1)
    sim_config['augmentation_samples'] = safe_int(parser['Simulation'].get('augmentation_samples', '1'), 1)
    sim_config['energy_bins'] = safe_int(parser['Simulation'].get('energy_bins', '1'), 1)
    # 显式获取布尔值并检查
    noise_value = parser['Simulation'].get('noise', 'False').strip().lower()
    save_noiseless_value = parser['Simulation'].get('save_noiseless', 'False').strip().lower()
    # 手动转换为布尔值
    # 显式将 'true' 和 'false' 转换为布尔类型
    sim_config['noise'] = noise_value in ['true', '1', 'yes']  # 允许多种真值形式
    sim_config['save_noiseless'] = save_noiseless_value in ['true', '1', 'yes']

    # Materials配置
    mat_config = config['Materials']
    mat_config['material_count'] = safe_int(parser['Materials'].get('material_count', '0'), 0)
    # 确保 material_count 为整数
    material_count = int(mat_config['material_count'])
    # 遍历材料数量
    for i in range(material_count):
        par_name = 'lac_{}'.format(i + 1)
        mat_config[par_name] = safe_float(parser['Materials'].get(par_name, '0.0'), 0.0)

    # Noise配置
    noise_config = config['Noise']
    noise_config['Flatfield'] = safe_float(parser['Noise'].get('Flatfield', '1.0'), 1.0)
    noise_config['Poisson_scaling'] = safe_float(parser['Noise'].get('Poisson_scaling', '1.0'), 1.0)
    noise_config['Gaussian_std'] = safe_float(parser['Noise'].get('Gaussian_std', '0.0'), 0.0)
    noise_config['Blur_width'] = safe_float(parser['Noise'].get('Blur_width', '1.0'), 1.0)

    return config


def apply_median_filter(vol):
    ''' Applies median filter to the volume using GPU acceleration (`cupyx.scipy.ndimage.median_filter`).

    :param vol: Array containing the object's model
    :type vol: :class:`np.ndarray`
    :return: Filtered volume
    :rtype: :class:`np.ndarray`

    '''
    vol_gpu = cupy.asarray(vol)
    vol_gpu = cupyx.scipy.ndimage.median_filter(vol_gpu, 3)
    vol_cpu = vol_gpu.get()

    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return vol_cpu
