# utils/sharding.py
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

def setup_sharding():
    devices = np.array(jax.devices()).reshape(-1, 1)
    mesh = Mesh(devices, ('data', 'model'))
    return mesh, {
        "data": NamedSharding(mesh, P('data')),
        "model": NamedSharding(mesh, P(None, 'model')),
        "replicated": NamedSharding(mesh, P()),
    }

def shard_params(params, sharding_rules):
    def _shard(x):
        return jax.device_put(x, sharding_rules["replicated"])  # Customize per-param if needed
    return jax.tree_map(_shard, params)
