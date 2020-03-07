# ## TimeDataset

# 時系列データは`TimeDataset`クラスで作成できます。

from ivory.common.context import np
from ivory.common.dataset import TimeDataset

x = np.arange(128).reshape(-1, 4)
t = x[:, 0]
data = TimeDataset((x, t), batch_size=2, time_size=4)
data
for x, t in data:
    print(data.state)
    print(t)
print(data.shape, x.shape, t.shape)
