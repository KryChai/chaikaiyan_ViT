from SETR_Model import *
from Dataset import *
# 在模型训练前，建议先加载vit的权重
'''
    num_classes（分割任务中类别的数量）、image_size（图像大小）、patch_size（补丁大小）、
    dim（Transformer 层的维度）、depth（Transformer 层的数量）、heads（注意力头的数量）、
    mlp_dim（前馈网络的隐藏层维度）、channels（图像的通道数，默认为 3）、dim_head（每个注意力头的维度）、
    dropout（嵌入层的 Dropout 比率）、emb_dropout（Transformer 层的 Dropout 比率）和
     out_indices（Transformer 层的输出索引）。
'''
from SETR_Model import *  # 导入SETR模型和相关函数
from Dataset import *  # 导入自定义的数据集

# 定义模型参数
num_classes = 1  # 类别数量，对于冠状血管分割，我们只关注一个类别
image_size = 256  # 图像大小
patch_size = 256 // 16  # 补丁大小，512除以32
dim = 1024  # Transformer层的维度
depth = 24  # Transformer层的数量
channels = 1  # 图像的通道数，灰度图只有一个通道
heads = 16  # 注意力头的数量
mlp_dim = 2048  # 前馈网络的隐藏层维度

# 创建模型
model = SETR(num_classes=num_classes, image_size=image_size, patch_size=patch_size, dim=dim, depth=depth, channels=channels, heads=heads, mlp_dim=mlp_dim).cuda()

import openpyxl
wb = openpyxl.Workbook()
# 导入必要的库
from d2l import torch as d2l
import pandas as pd
import monai
import numpy as np
from torchcontrib.optim import SWA

# 训练参数
epochs_num = 100  # 训练轮数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 使用SGD优化器
schedule = monai.optimizers.LinearLR(optimizer, end_lr=0.05, num_iter=int(epochs_num * 0.75))  # 使用线性学习率衰减

# 使用SWA优化 来提升SGD的效果
steps_per_epoch = int(len(train_loader.dataset) / train_loader.batch_size)
swa_start = int(epochs_num * 0.75)  # SWA优化的开始轮数
optimizer = SWA(optimizer, swa_start=swa_start * steps_per_epoch, swa_freq=steps_per_epoch, swa_lr=0.05)  # 使用SWA优化

# 损失函数
lossf = nn.CrossEntropyLoss(ignore_index=255)  # 多分类交叉熵损失函数，忽略背景类别

# 评估函数
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device

    metric = d2l.Accumulator(2)  # 创建一个累加器
    with torch.no_grad():  # 关闭梯度计算
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            output = net(X)
            pred = output[-1]
            metric.add(d2l.accuracy(pred, y), d2l.size(y))
    return metric[0] / metric[1]  # 返回准确率

# 训练函数
def train_ch13(net, train_iter, test_iter, loss, optimizer, num_epochs, schedule, swa_start=swa_start, devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)  # 创建计时器和批次计数器
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])  # 创建动画器
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])  # 设置数据并行

    # 训练过程中的数据
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_list = []
    time_list = []
    lr_list = []

    # optimizer.swap_swa_sgd()
    for epoch in range(num_epochs):
        # 训练过程中的数据累加器
        metric = d2l.Accumulator(4)

        for i, (X, labels) in enumerate(train_iter):
            timer.start()  # 开始计时

            if isinstance(X, list):
                X = [x.to(devices[0]) for x in X]  # 将输入数据转移到GPU上
            else:
                X = X.to(devices[0])
            gt = labels.long().to(devices[0])  # 将标签转移到GPU上

            net.train()  # 设置模型为训练模式

            # optimizer.zero_grad()  # 清空梯度
            for param in model.parameters():
                param.grad = None  # 或者 param.grad.zero_()

            result = net(X)  # 执行前向传播

            pred = result[-1]  # 获取最后的预测结果
            seg_loss = loss(result[-1], gt)  # 计算分割损失

            aux_loss_1 = loss(result[0], gt)
            aux_loss_2 = loss(result[1], gt)
            aux_loss_3 = loss(result[2], gt)

            loss_sum = seg_loss + 0.2 * aux_loss_1 + 0.3 * aux_loss_2 + 0.4 * aux_loss_3  # 计算总损失

            loss_sum.sum().backward()  # 反向传播
            optimizer.step()  # 更新权重

            acc = d2l.accuracy(pred, gt)  # 计算准确率
            metric.add(loss_sum, acc, labels.shape[0], labels.numel())  # 累加损失和准确率

            timer.stop()  # 停止计时
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))  # 更新动画器

        if optimizer.state_dict()['param_groups'][0]['lr'] > 0.05:
            schedule.step()  # 调整学习率

        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 在测试集上评估准确率

        if (epoch + 1) >= swa_start:
            if epoch == 0 or epoch % 5 == 5 - 1 or epoch == num_epochs - 1:
                optimizer._reset_lr_to_swa()  # 重置学习率到SWA值
                optimizer.swap_swa_sgd()  # 交换SWA和SGD
                optimizer.bn_update(train_iter, net, device='cuda')  # 更新批量归一化层
                test_acc = evaluate_accuracy_gpu(net, test_iter)  # 重新评估准确率
                optimizer.swap_swa_sgd()  # 交换回SWA和SGD

        animator.add(epoch + 1, (None, None, test_acc))  # 更新动画器

        print(f"epoch {epoch + 1}/{epochs_num} --- loss {metric[0] / metric[2]:.3f} --- train acc {metric[1] / metric[3]:.3f} --- test acc {test_acc:.3f} --- lr {optimizer.state_dict()['param_groups'][0]['lr']} --- cost time {timer.sum()}")  # 打印日志

        # 保存训练数据
        df = pd.DataFrame()
        loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)
        epochs_list.append(epoch + 1)
        time_list.append(timer.sum())
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

        df['epoch'] = epochs_list
        df['loss'] = loss_list
        df['train_acc'] = train_acc_list
        df['test_acc'] = test_acc_list
        df["lr"] = lr_list
        df['time'] = time_list

        df.to_excel("/tmp/ViT/Result/savefile/test.xlsx")
        # ----------------保存模型-------------------
        if np.mod(epoch + 1, 5) == 0:
            torch.save(net, f'/tmp/ViT/Result/checkpoints/test{epoch + 1}.pth')

    # 保存下最后的model
    torch.save(net, f'/tmp/ViT/Result/test.pth')


train_ch13(model, train_loader, val_loader, lossf, optimizer, epochs_num, schedule=schedule)