import torch
import torch.nn.functional as F

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='sum'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        # 原始的Focal Loss用的是sigmoid，输入是[N]，即分类为1的概率
        # pt = torch.sigmoid(predict)     # sigmoid获取概率

        # 我改了一下，我想用softmax，输入是[N, 2]，把它定义成多分类的形式
        pt = F.softmax(predict, dim=-1)       # softmax获取概率
        pt = pt[:, 1]

        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt + 10 ** (-10)) \
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt + 10 ** (-10))

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
