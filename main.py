# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
from util import Focal_loss
import torch
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    loss_fl = Focal_loss.BCEFocalLoss()
    loss_bce = torch.nn.CrossEntropyLoss(reduction='none')
    pred = [[0.1, 0.9], [0.9, 0.1]]
    y = [1, 0]
    print(loss_fl(torch.FloatTensor(pred), torch.LongTensor(y)))
    print(loss_bce(torch.FloatTensor(pred), torch.LongTensor(y)))
    print(F.one_hot(torch.LongTensor(y)))

    # pred = [0.5, 0.5, 0.9, 0.9, 0.1]
    # y = [0, 1, 0, 1, 0]
    # pred1 = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]
    # y1 = [[1, 0], [0, 1], [0, 1]]
    # print(F.one_hot(torch.LongTensor(y))).float
    # print(sigmoid_focal_loss(torch.tensor(pred), torch.FloatTensor(y), alpha=0.25, gamma=2, reduction='none'))
    # print(loss(torch.FloatTensor(pred), torch.LongTensor(y)))

