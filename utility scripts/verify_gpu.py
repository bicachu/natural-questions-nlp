import os
import tensorflow as tf

print(tf.compat.v1.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
))

# print('====== Your GPU info ======')
# print('name:\t\t', property.name)
# print('capability:\t', 'v{}.{}'.format(property.major, property.minor))
# print('memory:\t\t', round(property.total_memory / 1e9), 'Gb')
# print('processors:\t', property.multi_processor_count)
