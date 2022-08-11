# 多任务预训练代码

## 运行

在tensorflow 1.15 + keras 2.3.1下训练，直接执行`python main.py`即可。

## 架构

每个任务一个脚本（如 t0201_iflytek.py ），脚本格式可以直接看示例；然后全部由 main.py 统一导入和管理。
