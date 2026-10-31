"""复制本文件建立自己的实验配置，再通过 --config 指定。"""

from dexmani_real.config.real import RealConfig

# 仅组装参数，不在实验文件中连接设备或加载模型。
# 修改嵌套参数可直接构造对应 dataclass，或使用 dataclasses.replace。
CONFIG = RealConfig()
