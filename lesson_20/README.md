# 第 20 课：Dataset、DataLoader、batch 与泛化

本课把第 19 课的全量 XOR 训练扩展为分 batch 训练。第一步只学习如何将 inputs 和 labels 组合为 Dataset，并由 DataLoader 按 batch 取出。

## Day 1：创建不打乱顺序的 DataLoader

只修改 `batch_data.py` 中的 `make_data_loader`：

1. 使用 `torch.utils.data.TensorDataset(inputs, labels)` 绑定样本和标签；
2. 使用 `torch.utils.data.DataLoader` 创建 loader；
3. `batch_size` 使用函数参数；
4. 当前阶段固定 `shuffle=False`；
5. 不设置 `drop_last=True`，因此最后不足一个 batch 的样本仍然保留；
6. 返回 DataLoader。

当样本数为 `10`、`batch_size=4` 时，batch 大小应为：

```text
4, 4, 2
```

DataLoader 默认不会给最后一个 batch 补零。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_20/test_batch_data.py -q -p no:cacheprovider
```

骨架预期先失败；实现完成后应为 `4 passed`。

## Day 2：可复现地打乱样本

扩展 `make_data_loader`：

1. 增加 `shuffle: bool = False` 参数；
2. 增加 `seed: int | None = None` 参数；
3. 当 seed 不为 `None` 时，创建 `torch.Generator().manual_seed(seed)`；
4. 将 shuffle 和 generator 传给 DataLoader；
5. 同一个 seed 创建的两个 loader，第一次遍历顺序应一致；
6. 不同 seed 的顺序可以不同，但输入和标签始终保持配对。

注意：同一个 loader 被重复遍历时，generator 状态会继续前进，因此不同 epoch 可以得到不同顺序。可复现指的是“相同代码、相同 seed、重新运行实验”得到相同序列。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_20/test_shuffle_data.py -q -p no:cacheprovider
```

当前实现预期先失败；完成后应为 `4 passed`。

## Day 3：用 batch 完成一个 epoch

只修改 `train_epoch.py` 中的 `train_one_epoch`：

1. 遍历 DataLoader 得到 `(batch_inputs, batch_labels)`；
2. 对每个 batch 调用第 19 课的 `train_one_step`；
3. 用 `batch_loss * batch_size` 累加 loss；
4. 用所有样本数加权计算 epoch 平均 loss；
5. 返回 Python `float`；
6. 不要把不同 batch 的 loss 直接无权平均，因为最后一个 batch 可能更小。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_20/test_train_epoch.py -q -p no:cacheprovider
```

当前实现预期先失败；完成后应为 `4 passed`。

## Day 4：可复现的训练集与验证集划分

只修改 `data_split.py` 中的 `split_train_validation`：

1. 使用 `torch.Generator().manual_seed(seed)` 创建随机生成器；
2. 使用 `torch.randperm(sample_count, generator=generator)` 生成随机索引；
3. 前 `int(sample_count * train_fraction)` 个索引作为训练集；
4. 剩余索引作为验证集；
5. inputs 和 labels 必须使用相同索引；
6. 返回 `(train_inputs, train_labels, validation_inputs, validation_labels)`；
7. 相同 seed 必须复现相同划分。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_20/test_data_split.py -q -p no:cacheprovider
```

当前骨架预期先失败；完成后应为 `4 passed`。

## Day 5：训练/验证闭环与指标历史

只修改 `train_validate.py` 中的 `train_and_validate`：

每个 epoch 严格执行：

```text
train_one_epoch(train_loader)
-> evaluate_classifier(validation_inputs, validation_labels)
-> 记录 train_loss、validation_loss、validation_accuracy
```

要求：

1. 训练阶段可以更新参数；
2. 验证阶段不能调用 `backward()` 或 optimizer；
3. 返回三个等长的 list；
4. `epochs=0` 时返回三个空列表。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_20/test_train_validate.py -q -p no:cacheprovider
```

当前骨架预期先失败；完成后应为 `3 passed`。

## Day 6：生成带噪声的二维 XOR 数据

只修改 `noisy_xor.py` 中的 `make_noisy_xor`：

1. 使用四个中心点 `[-1,-1]、[-1,1]、[1,-1]、[1,1]`；
2. 中心标签依次为 `[0,1,1,0]`；
3. 每个中心重复 `samples_per_corner` 次；
4. 使用固定 seed 的 `torch.Generator` 生成高斯噪声；
5. 噪声乘以 `noise_std` 后加到 inputs；
6. labels 不添加噪声；
7. 返回 float32 inputs 和 int64 labels。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_20/test_noisy_xor.py -q -p no:cacheprovider
```

当前骨架预期先失败；完成后应为 `4 passed`。

## Day 7：观察过拟合

运行独立实验入口：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m lesson_20.overfitting_experiment
```

实验故意组合少量数据、高噪声和较大模型。固定 seed 后，当前环境的结果约为：

```text
final train loss: 0.0130
final validation loss: 2.9863
final validation accuracy: 0.7500
```

训练 loss 很低而验证 loss 很高，说明模型记住了少量训练样本，但没有学到能够稳定迁移到未见样本的规律。
