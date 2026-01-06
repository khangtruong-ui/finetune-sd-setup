import jax

def tree_map(f):

    def wrap_func(pytree, *args, **kwargs):
        return jax.tree.map(lambda x: f(x, *args, **kwargs), pytree)

    return wrap_func
