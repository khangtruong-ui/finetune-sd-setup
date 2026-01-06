# utils/sharding.py
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

mesh = Mesh(np.array(jax.devices()).reshape((-1, 1)), ('data', 'kernel',))
kernel_sharding = NamedSharding(mesh, P(None, 'kernel'))
conv_sharding = NamedSharding(mesh, P(None, None, None, 'kernel'))
sharding = NamedSharding(mesh, P('data'))
non_sharding = no_sharding = NamedSharding(mesh, P())


def distribute_device(tensor, sharding, replicate=False):
    if type(tensor) in [int, float, str]:
        return tensor
    global_shape = (tensor.shape[0] * jax.process_count(),) + tensor.shape[1:] if not replicate else tensor.shape
    global_array = jax.make_array_from_process_local_data(sharding, tensor, global_shape)
    return global_array
