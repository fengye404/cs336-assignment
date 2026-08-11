# Linear 实现清单

目标：实现作业 PDF 3.3.2 的无 bias `Linear` 模块，并通过对应测试。

## 类与接口

- [ ] 在自己的源码模块中声明 `Linear` 类。
- [ ] 继承 `torch.nn.Module`。
- [ ] `__init__` 接收 `in_features`、`out_features`、`device`、`dtype`。
- [ ] 在构造函数开头调用父类构造函数。
- [ ] 实现 `forward(x)`。

## 参数

- [ ] 创建并保存可训练权重 `W`（`nn.Parameter`）。
- [ ] `W` 的 shape 是 `(out_features, in_features)`。
- [ ] 不创建 bias 参数。
- [ ] 参数放在传入的 `device` 和 `dtype` 上。

## 初始化

- [ ] 计算 Linear 初始化所需的标准差：`sqrt(2 / (in_features + out_features))`。
- [ ] 用 PDF 指定的截断正态分布初始化 `W`。
- [ ] 截断范围是 `[-3 * std, 3 * std]`。

## Forward 行为

- [ ] 输入支持 shape `(..., in_features)`。
- [ ] 输出 shape 为 `(..., out_features)`。
- [ ] 使用无 bias 的线性变换；数学记号是 `y = W x`。
- [ ] PyTorch 的矩阵乘法方向与权重 shape 一致。

## 验证

- [ ] 用一个手算的小矩阵检查输出数值。
- [ ] 检查 `W` 能在 `model.parameters()` 中出现。
- [ ] 在 `tests/adapters.py` 中连接自己的实现。
- [ ] 运行 `uv run pytest tests/test_model.py -k linear`。
- [ ] 记录首个失败信息与修复结论。
