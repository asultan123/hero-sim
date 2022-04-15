import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ifmap_size = 10
kernel_size = 3
ofmap_size = ifmap_size-kernel_size+1
channel_count = 4

ifmap = np.arange(1, (ifmap_size**2)*channel_count+1).reshape(channel_count, ifmap_size, ifmap_size)
kernel = np.arange(1, ((kernel_size**2)*channel_count)+1).reshape(channel_count, kernel_size, kernel_size)
ifmap_window = sliding_window_view(ifmap, (channel_count, kernel_size, kernel_size)).reshape(-1, channel_count, kernel_size, kernel_size)
ofmap = np.sum(np.multiply(ifmap_window, kernel), axis=(1, 2, 3)).reshape(ofmap_size, ofmap_size)
print(ofmap)
