# 这个工程定义了四个类，分别是channel_shuffle、InvertedResidual、DUC和ShuffleNetV2HeatMap。
# channel_shuffle类是一个函数，用于对输入的张量进行通道混洗操作，即将每个分组中的通道交叉排列，以增加通道间的信息流动。
# InvertedResidual类是一个反向残差模块，用于实现ShuffleNetV2的基本单元。它包含两个分支，一个分支不变，另一个分支进行卷积操作，并在最后进行通道混洗操作。它可以根据步长的不同，实现特征图的下采样或者保持尺寸不变。
# DUC类是一个密集上采样卷积模块，用于实现特征图的上采样操作。它包含一个卷积层，一个批量归一化层，一个激活函数层和一个像素重排层。它可以根据上采样因子的不同，实现特征图的放大操作。
# ShuffleNetV2HeatMap类是一个生成热图的网络，用于实现人体姿态估计的主要功能。它包含五个阶段的卷积层，每个阶段由若干个反向残差模块组成，并根据通道比例的不同，设置不同的网络层参数。它还包含两个密集上采样卷积模块，用于将特征图放大到原始图像的尺寸，并输出热图。它还定义了一个焦点损失函数，用于计算预测热图和真实热图之间的损失值，并根据正负样本的比例进行平衡。它还定义了一个训练步骤和一个评估步骤，用于实现网络的训练和测试。

# ShuffleNetV2HeatMap类的forward方法调用了内部的_forward_impl方法，该方法依次调用了conv1、maxpool、stage2、stage3、stage4、conv5、conv_compress、duc1、duc2和conv_result这些网络层的forward方法，得到输出的热图。
# ShuffleNetV2HeatMap类的train_step方法调用了forward方法，得到预测的热图，然后调用了nn.functional.sigmoid函数，将值映射到0-1之间，然后调用了loss_func方法，计算损失值，并返回一个包含不同关键点损失值的字典。
# ShuffleNetV2HeatMap类的eval_step方法调用了forward方法，得到预测的热图，然后调用了nn.functional.sigmoid函数，将值映射到0-1之间，并返回热图。
# InvertedResidual类的forward方法根据步长的不同，将输入张量分为两个分支，一个分支不变，另一个分支依次通过branch2中的网络层的forward方法，并在最后进行通道混洗操作，然后将两个分支在通道维上拼接起来，并返回输出张量。
# DUC类的forward方法依次通过conv、bn、relu和pixel_shuffle这些网络层的forward方法，并返回输出张量。
# FocalLoss类的forward方法调用了ce方法，计算输入和目标之间的对数概率，然后使用指数函数计算概率，并使用焦点损失函数的公式计算损失值，并返回损失值的均值。

# 导入torch库，用于深度学习
import torch
# 导入torch.nn库，用于定义神经网络层
import torch.nn as nn
# 导入DLEngine.modules.visualize.visual_util模块，用于可视化
from DLEngine.modules.visualize.visual_util import *

# 定义一个全局变量，表示该模块中可以被导入的类或函数
__all__ = ['ShuffleNetV2HeatMap']


# 定义一个函数，名为channel_shuffle，用于对输入的张量进行通道混洗操作
def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    # 获取输入张量的维度信息，包括批量大小、通道数、高度和宽度
    batchsize, num_channels, height, width = x.data.size()
    # 计算每个分组中的通道数，等于总通道数除以分组数
    channels_per_group = num_channels // groups

    # 重塑张量，将其变为(batchsize, groups, channels_per_group, height, width)的形状
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # 交换第二维和第三维，使得每个分组中的通道交叉排列
    x = torch.transpose(x, 1, 2).contiguous()

    # 展平张量，将其变为(batchsize, num_channels, height, width)的形状
    x = x.view(batchsize, -1, height, width)

    # 返回混洗后的张量
    return x


# 定义一个类，名为InvertedResidual，继承自nn.Module类，用于实现反向残差模块
class InvertedResidual(nn.Module):
    # 定义初始化方法，接受三个参数：inp（输入通道数），oup（输出通道数），stride（步长）
    def __init__(self, inp, oup, stride):
        # 调用父类的初始化方法
        super(InvertedResidual, self).__init__()

        # 如果步长不在1到3之间，抛出异常
        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        # 将步长赋值给类的属性
        self.stride = stride

        # 计算分支特征的通道数，等于输出通道数除以2
        branch_features = oup // 2
        # 断言步长不等于1或者输入通道数等于分支特征通道数乘以2
        assert (self.stride != 1) or (inp == branch_features << 1)

        # 如果步长大于1，定义第一个分支，包括以下层：
        # 深度卷积层，输入和输出通道数都是inp，卷积核大小为3，步长为self.stride，填充为1
        # 批量归一化层，输入和输出通道数都是inp
        # 卷积层，输入通道数为inp，输出通道数为branch_features，卷积核大小为1，步长为1，填充为0，无偏置
        # 批量归一化层，输入和输出通道数都是branch_features
        # 激活函数层，使用ReLU函数，并在原地操作
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        # 否则，定义第一个分支为空的顺序容器
        else:
            self.branch1 = nn.Sequential()

        # 定义第二个分支，包括以下层：
        # 卷积层，输入通道数为inp（如果步长大于1）或branch_features（如果步长等于1），输出通道数为branch_features，
        # 卷积核大小为1，步长为1，填充为0，无偏置
        # 批量归一化层，输入和输出通道数都是branch_features
        # 激活函数层，使用ReLU函数，并在原地操作
        # 深度卷积层，输入和输出通道数都是branch_features，卷积核大小为3，步长为self.stride，填充为1
        # 批量归一化层，输入和输出通道数都是branch_features
        # 卷积层，输入和输出通道数都是branch_features，卷积核大小为1，步长为1，填充为0，无偏置
        # 批量归一化层，输入和输出通道数都是branch_features
        # 激活函数层，使用ReLU函数，并在原地操作
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

        # 定义一个静态方法，名为depthwise_conv，用于实现深度卷积操作

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        # 返回一个nn.Conv2d对象，输入和输出通道数都是i，卷积核大小为kernel_size，步长为stride，填充为padding，
        # 无偏置（如果bias为False），分组数为i（即每个输入通道对应一个输出通道）
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    # 定义一个前向传播方法，接受一个参数：x（输入张量）
    def forward(self, x):
        # 如果步长等于1，执行以下操作
        if self.stride == 1:
            # 将输入张量沿着第二维（通道维）分成两部分，分别赋值给x1和x2
            x1, x2 = x.chunk(2, dim=1)
            # 将x1和第二个分支的输出在通道维上拼接起来，作为输出张量
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        # 否则，执行以下操作
        else:
            # 将第一个分支和第二个分支的输出在通道维上拼接起来，作为输出张量
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        # 对输出张量进行通道混洗操作，分组数为2
        out = channel_shuffle(out, 2)

        # 返回输出张量
        return out


# 定义一个类，名为FocalLoss，继承自nn.Module类，用于实现焦点损失函数
class FocalLoss(nn.Module):
    # 定义初始化方法，接受四个参数：weight（权重张量），reduction（损失函数的缩减方式），gamma（焦点系数），eps（防止除零的小数）
    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        # 调用父类的初始化方法
        super(FocalLoss, self).__init__()
        # 将gamma和eps赋值给类的属性
        self.gamma = gamma
        self.eps = eps
        # 定义一个交叉熵损失函数对象，使用给定的权重和缩减方式
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    # 定义一个前向传播方法，接受两个参数：input（输入张量），target（目标张量）
    def forward(self, input, target):
        # 使用交叉熵损失函数计算输入和目标之间的对数概率
        logp = self.ce(input, target)
        # 使用指数函数计算输入和目标之间的概率
        p = torch.exp(-logp)
        # 使用焦点损失函数的公式计算损失值，即(1 - p) ** gamma * logp
        loss = (1 - p) ** self.gamma * logp
        # 返回损失值的均值
        return loss.mean()


# 定义一个类，名为DUC，继承自nn.Module类，用于实现密集上采样卷积
class DUC(nn.Module):
    '''
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''

    # 定义初始化方法，接受三个参数：inplanes（输入通道数），planes（输出通道数），upscale_factor（上采样因子），默认为2
    def __init__(self, inplanes, planes, upscale_factor=2):
        # 调用父类的初始化方法
        super(DUC, self).__init__()
        # 定义一个卷积层，输入通道数为inplanes，输出通道数为planes，卷积核大小为3，填充为1，无偏置
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, bias=False)
        # 定义一个批量归一化层，输入和输出通道数都是planes，动量为0.1
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        # 定义一个激活函数层，使用ReLU函数，并在原地操作
        self.relu = nn.ReLU(inplace=True)
        # 定义一个像素重排层，上采样因子为upscale_factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    # 定义一个前向传播方法，接受一个参数：x（输入张量）
    def forward(self, x):
        # 依次通过卷积层、批量归一化层、激活函数层和像素重排层，得到输出张量
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        # 返回输出张量
        return x


class ShuffleNetV2HeatMap(nn.Module):  # 定义一个继承自nn.Module的类，用于生成热图
    def __init__(self, arg_dict):  # 定义类的初始化方法，接受一个字典作为参数
        super(ShuffleNetV2HeatMap, self).__init__()  # 调用父类的初始化方法
        self.kp_num = arg_dict['kp_num']  # 从字典中获取关键点的数量，赋值给类的属性
        width_mult = arg_dict['channel_ratio']  # 从字典中获取通道比例，赋值给局部变量
        inverted_residual = InvertedResidual  # 将InvertedResidual类赋值给局部变量，方便后续使用
        if width_mult == 0.5:  # 根据通道比例的不同，设置不同的网络层参数
            stages_repeats = [4, 8, 4]  # 每个阶段重复的次数
            stages_out_channels = [24, 48, 96, 192, 1024]  # 每个阶段输出的通道数
        elif width_mult == 1.0:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 244, 488, 976, 2048]
        elif width_mult == 0.25:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 28, 48, 96, 512]
        else:
            assert (False)  # 如果通道比例不在上述范围内，抛出异常

        if len(stages_repeats) != 3:  # 检查stages_repeats的长度是否为3，如果不是，抛出异常
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:  # 检查stages_out_channels的长度是否为5，如果不是，抛出异常
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels  # 将stages_out_channels赋值给类的属性

        input_channels = 3  # 设置输入通道数为3
        output_channels = self._stage_out_channels[0]  # 设置输出通道数为第一个阶段的输出通道数
        self.conv1 = nn.Sequential(  # 定义第一个卷积层，包含一个卷积操作，一个批归一化操作和一个激活函数
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),  # 卷积操作，卷积核大小为3，步长为2，填充为1，无偏置
            nn.BatchNorm2d(output_channels),  # 批归一化操作，对输出通道进行归一化
            nn.ReLU(inplace=True),  # 激活函数，使用ReLU函数，并在原地修改输入
        )
        input_channels = output_channels  # 更新输入通道数为输出通道数

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 定义最大池化层，池化核大小为3，步长为2，填充为1

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]  # 定义三个阶段的名称
        for name, repeats, output_channels in zip(  # 遍历三个阶段的名称，重复次数和输出通道数
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]  # 创建一个序列，包含一个反向残差模块，输入通道数和输出通道数由参数决定，步长为2
            for i in range(repeats - 1):  # 根据重复次数，往序列中添加更多的反向残差模块，步长为1
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))  # 将序列作为一个子模块赋值给类的属性，属性名由name决定
            input_channels = output_channels  # 更新输入通道数为输出通道数

        output_channels = self._stage_out_channels[-1]  # 设置输出通道数为最后一个阶段的输出通道数

        self.conv5 = nn.Sequential(  # 定义第五个卷积层，包含一个卷积操作，一个批归一化操作和一个激活函数
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),  # 卷积操作，卷积核大小为1，步长为1，无填充，无偏置
            nn.BatchNorm2d(output_channels),  # 批归一化操作，对输出通道进行归一化
            nn.ReLU(inplace=True),  # 激活函数，使用ReLU函数，并在原地修改输入
        )

        self.conv_compress = nn.Conv2d(1024, 256, 1, 1, 0,
                                       bias=False)  # 定义一个压缩卷积层，将1024个通道压缩到256个通道，卷积核大小为1，步长为1，无填充，无偏置
        self.duc1 = DUC(256, 512, upscale_factor=2)  # 定义一个密集上采样卷积层，将256个通道扩展到512个通道，并将特征图的尺寸放大2倍
        self.duc2 = DUC(128, 256, upscale_factor=2)  # 定义一个密集上采样卷积层，将128个通道扩展到256个通道，并将特征图的尺寸放大2倍
        # self.duc3 = DUC(64, 128, upscale_factor=2) # 定义一个密集上采样卷积层，将64个通道扩展到128个通道，并将特征图的尺寸放大2倍，但这一层被注释掉了
        self.conv_result = nn.Conv2d(64, self.kp_num, 1, 1, 0,
                                     bias=False)  # 定义一个结果卷积层，将64个通道压缩到关键点的数量，并输出热图，卷积核大小为1，步长为1，无填充，无偏置
        self.loss_func = torch.nn.MSELoss(size_average=None, reduce=None,
                                          reduction='sum')  # 定义一个损失函数，使用均方误差作为损失函数，并对所有元素求和

    def _forward_impl(self, x): # 定义一个内部的前向传播方法，接受一个输入张量x
        # See note [TorchScript super()]
        x = self.conv1(x) # 将输入张量通过第一个卷积层
        x = self.maxpool(x) # 将输出张量通过最大池化层
        x = self.stage2(x) # 将输出张量通过第二个阶段的子模块
        x = self.stage3(x) # 将输出张量通过第三个阶段的子模块
        x = self.stage4(x) # 将输出张量通过第四个阶段的子模块
        x = self.conv5(x) # 将输出张量通过第五个卷积层
        x = self.conv_compress(x) # 将输出张量通过压缩卷积层
        x = self.duc1(x) # 将输出张量通过第一个密集上采样卷积层
        x = self.duc2(x) # 将输出张量通过第二个密集上采样卷积层
        # x = self.duc3(x) # 这一行被注释掉了，不会将输出张量通过第三个密集上采样卷积层
        x = self.conv_result(x) # 将输出张量通过结果卷积层，得到热图
        return x # 返回热图

    def forward(self, x): # 定义类的前向传播方法，接受一个输入张量x
        x = self._forward_impl(x) # 调用内部的前向传播方法，得到热图
        if self.training: # 如果是在训练模式下，直接返回热图
            return x
        else: # 如果是在评估模式下，对热图应用sigmoid函数，将值映射到0到1之间，然后返回热图
            heatmaps = nn.functional.sigmoid(x)
            return heatmaps

    # train_step这个函数的具体作用是在训练过程中，对一批图片和对应的热图标签进行前向传播和损失函数的计算，以及可视化预测的热图和真实的热图
    def train_step(self, images, labels, local_rank=0):
        preds = self.forward(images)  # 前向传播，得到预测的热图
        preds = nn.functional.sigmoid(preds)  # 对预测的热图应用sigmoid函数，将值映射到0-1之间
        batch = labels.shape[0]  # 获取批次大小
        # visual
        epoch = int(os.environ['epoch'])  # 获取当前的训练轮数
        epoch_changed = os.environ['epoch_changed']  # 获取一个标志，表示是否刚刚开始新的一轮训练
        if epoch_changed == 'true' and local_rank == 0:  # 如果是新的一轮训练，并且是第一个进程（local_rank为0）
            visual_add_image_with_heatmap(images, preds, labels, mean=[103.53, 116.28, 123.675],
                                          std=[57.375, 57.12, 58.395],
                                          epoch=epoch)  # 调用一个函数，将原始图片、预测的热图和真实的热图叠加在一起，显示出来，方便观察效果
            os.environ['epoch_changed'] = 'false'  # 将标志设为false，表示已经处理过新的一轮训练
        preds_head = preds[:, 0, :, :]  # 取出预测的热图中第一个通道的内容，表示头部位置的热图
        preds_knee = preds[:, 1, :, :]  # 取出预测的热图中第二个通道的内容，表示膝盖位置的热图
        preds_loin = preds[:, 2, :, :]  # 取出预测的热图中第三个通道的内容，表示腰部位置的热图
        labels_head = labels[:, 0, :, :]  # 取出真实的热图中第一个通道的内容，表示头部位置的热图
        labels_knee = labels[:, 1, :, :]  # 取出真实的热图中第二个通道的内容，表示膝盖位置的热图
        labels_loin = labels[:, 2, :, :]  # 取出真实的热图中第三个通道的内容，表示腰部位置的热图
        head_pos_mask = labels_head > 0 # 得到一个布尔张量，表示真实头部位置热图中哪些像素值大于0（即有头部信息）
        head_neg_mask = labels_head == 0 # 得到一个布尔张量，表示真实头部位置热图中哪些像素值等于0（即没有头部信息）
        knee_pos_mask = labels_knee > 0 # 得到一个布尔张量，表示真实膝盖位置热图中哪些像素值大于0（即有膝盖信息）
        knee_neg_mask = labels_knee == 0 # 得到一个布尔张量，表示真实膝盖位置热图中哪些像素值等于0（即没有膝盖信息）
        loin_pos_mask = labels_loin > 0 # 得到一个布尔张量，表示真实腰部位置热图中哪些像素值大于0（即有腰部信息）
        loin_neg_mask = labels_loin == 0 # 得到一个布尔张量，表示真实腰部位置热图中哪些像素值等于0（即没有腰部信息）

        loss_head_pos = self.loss_func(preds_head[head_pos_mask], labels_head[head_pos_mask]) / labels.shape[0] # 计算预测头部位置热图和真实头部位置热图在有头部信息的像素上的损失函数，这里使用的是均方误差（MSE）损失函数，并且除以批次大小，得到每个样本的平均损失
        loss_head_neg = self.loss_func(preds_head[head_neg_mask], labels_head[head_neg_mask]) / labels.shape[0] # 计算预测头部位置热图和真实头部位置热图在没有头部信息的像素上的损失函数，同样使用MSE损失函数，并且除以批次大小
        loss_knee_pos = self.loss_func(preds_knee[knee_pos_mask], labels_knee[knee_pos_mask]) / labels.shape[0] # 计算预测膝盖位置热图和真实膝盖位置热图在有膝盖信息的像素上的损失函数，同样使用MSE损失函数，并且除以批次大小
        loss_knee_neg = self.loss_func(preds_knee[knee_neg_mask], labels_knee[knee_neg_mask]) / labels.shape[0] # 计算预测膝盖位置热图和真实膝盖位置热图在没有膝盖信息的像素上的损失函数，同样使用MSE损失函数，并且除以批次大小
        loss_loin_pos = self.loss_func(preds_loin[loin_pos_mask], labels_loin[loin_pos_mask]) / labels.shape[0] # 计算预测腰部位置热图和真实腰部位置热图在有腰部信息的像素上的损失函数，同样使用MSE损失函数，并且除以批次大小
        loss_loin_neg = self.loss_func(preds_loin[loin_neg_mask], labels_loin[loin_neg_mask]) / labels.shape[0] # 计算预测腰部位置热图和真实腰部位置热图在没有腰部信息的像素上的损失函数，同样使用MSE损失函数，并且除以批次大小

        loss_total = loss_head_pos + 2.0 * loss_knee_pos + loss_loin_pos + 0.1 * loss_head_neg + 0.2 * loss_knee_neg + 0.1 * loss_loin_neg  # 计算总的损失函数，这里对不同位置和不同像素的损失函数赋予了不同的权重，这是为了平衡不同位置和不同像素的重要性，这些权重是根据实验结果调整得到的
        return {'total': loss_total,
                'h_pos': loss_head_pos,
                'k_pos': loss_knee_pos,
                'l_pos': loss_loin_pos,
                'h_neg': loss_head_neg,
                'k_neg:': loss_knee_neg,
                'l_neg:': loss_loin_neg}  # 返回一个字典，包含总的损失函数和各个位置的正样本和负样本损失函数，这些值可以用于监控训练过程和评估模型性能

    def eval_step(self, images):  # 定义一个评估步骤的函数，输入是一批图片
        out = self.forward(images)  # 前向传播，得到预测的热图
        pass  # 这里没有写具体的评估逻辑，可能是因为还没有实现或者是省略了

