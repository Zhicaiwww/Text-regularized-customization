import torch
import torch.nn as nn
from accelerate import Accelerator

accelerator = Accelerator()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 3)
        self.layer3 = nn.Linear(3, 1)

        # 将第二层的参数设置为不需要梯度
        for param in self.layer2.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def main():
    # 创建输入数据
    input_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    input_data2 = torch.tensor([[2.0, 2.0], [3.0, 4.0]])

    # 创建模型实例
    model = SimpleModel()
    model, criterion = accelerator.prepare(model, nn.MSELoss())
    # 进行前向传播
    model.zero_grad()
    output2 = model(input_data2.to(accelerator.device))

    # 计算损失并进行反向传播
    loss = criterion(output2, torch.tensor([[0.0], [1.0]]).to(accelerator.device))
    accelerator.backward(loss, retain_graph=True)

if __name__ == '__main__':
    main()