import re
import subprocess
import os


# 获取下一个运行编号
def get_next_run_number(log_base_path, dataset_name, nn_type):
    """
    获取下一个运行编号，确保每个运行编号唯一。
    """
    run_num = 1
    while True:
        log_path = os.path.join(log_base_path, f"{dataset_name}_{nn_type}_r{run_num}")
        if not os.path.exists(log_path):  # 如果目录不存在，则可以使用该编号
            break
        run_num += 1  # 如果目录已存在，则继续增加编号
    return run_num


# 执行训练命令并返回退出状态
def run_training(nn_type, data, obj, run_id):
    """
    执行训练命令并返回退出状态。
    """
    # command = f"python train原始未早停最优对照.py --nn {nn_type} --data {data} --obj {obj} --run {run_id}"
    command = f"python train.py --nn {nn_type} --data {data} --obj {obj} --run {run_id}"
    print(f"正在执行命令: {command}")

    # 执行命令并
    # 等待其完成
    process = subprocess.Popen(command, shell=True)
    process.wait()  # 等待命令执行完成
    return process.returncode  # 返回进程的退出状态


def main():
    # nn_type = "efficientnetb4"
    nn_type = "resnet50"

    idx = 2
    data_str = f"gen_train{idx:03d}"
    run_ids = list(range(idx * 1000 + 20, idx * 1000 + 30))
    data = re.match(r"(gen_train\d+)", data_str).group(1)
    obj = "playdoh"
    # log_base_path = "../log"  # 基本日志目录路径
    log_base_path = r"M:\pycharm_project\project3_train\training_strategy\log"
    os.makedirs(log_base_path, exist_ok=True)

    for run_id in run_ids:
        # 为每个训练更新日志路径
        log_path = os.path.join(log_base_path, f"{data}_{nn_type}_r{run_id}")


        # 检查日志文件夹是否已存在
        if os.path.exists(log_path):
            print(f"训练编号 {run_id} 的日志文件夹已存在，跳过该训练。")
            continue  # 跳过当前训练，继续执行下一个训练

        # 创建日志目录
        os.makedirs(log_path, exist_ok=True)

        print(f"\n开始执行训练编号: {run_id}")

        # 执行训练命令
        return_code = run_training(nn_type, data, obj, run_id)

        # 检查训练是否成功
        if return_code == 0:
            print(f"训练编号 {run_id} 完成，准备执行下一个训练。\n")
        else:
            print(f"训练编号 {run_id} 失败，停止执行后续训练。")
            break  # 如果失败，停止执行后续训练


if __name__ == "__main__":
    main()
