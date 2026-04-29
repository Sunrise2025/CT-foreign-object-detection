from torch.utils.tensorboard import SummaryWriter
from examples import aug_accuracy

class Logger:
    def __init__(self, log_path, model, model_name):
        self.writer = SummaryWriter(log_path, comment=model_name)  # 为TensorBoard添加模型名注释
        self.model = model
        self.model_name = model_name  # 存储模型名称

    def check_validation(self, val_dl, epoch):
        conf_mat, y_pred, y_true = self.model.test(val_dl)
        accuracy = aug_accuracy.utils.compute_accuracy(y_pred, y_true)
        self.writer.add_scalar(f'{self.model_name}-Accuracy/Validation', accuracy, epoch)

    def log_train(self, loss, epoch):
        print("\n{:05d} {:>15} = {:.6f}".format(epoch, "Training loss", loss))
        self.writer.add_scalar(f'{self.model_name}-Loss/train', loss, epoch)

    def log_validation(self, loss, epoch):
        print("{:05d} {:>15} = {:.6f}".format(epoch, "Validation loss", loss))
        self.writer.add_scalar(f'{self.model_name}-Loss/validation', loss, epoch)

    # def info(self, param):
    #     # 你可以在这里打印或记录模型的其他信息
    #     print(f"Model: {self.model_name}")
    #     print(f"Additional info: {param}")