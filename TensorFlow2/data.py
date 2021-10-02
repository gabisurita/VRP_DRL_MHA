import tensorflow as tf


def generate_data(
    n_samples=1000, n_customer=20, seed=None, min_size=0.01, max_size=0.2
):
    if seed is None:
        g = tf.random.experimental.Generator.from_non_deterministic_state()
    else:
        g = tf.random.experimental.Generator.from_seed(seed)

    @tf.function
    def tf_rand():
        return [
            # Hub coords.
            g.uniform(shape=[n_samples, 2], minval=0, maxval=1),
            # Customer coords.
            g.uniform(shape=[n_samples, n_customer, 2], minval=0, maxval=1),
            # Customer demands.
            g.uniform(
                shape=[n_samples, n_customer],
                minval=min_size,
                maxval=max_size,
                dtype=tf.float32,
            ),
        ]

    return tf.data.Dataset.from_tensor_slices(tuple(tf_rand()))
