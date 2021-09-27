import tensorflow as tf


def generate_data(
    n_samples=1000, n_customer=20, seed=None, min_size=0.01, max_size=0.2
):
    g = tf.random.experimental.Generator.from_non_deterministic_state()
    if seed is not None:
        g = tf.random.experimental.Generator.from_seed(seed)

    @tf.function
    def tf_rand():
        return [
            g.uniform(shape=[n_samples, 2], minval=0, maxval=1),
            g.uniform(shape=[n_samples, n_customer, 2], minval=0, maxval=1),
            g.uniform(
                shape=[n_samples, n_customer],
                minval=min_size,
                maxval=max_size,
                dtype=tf.float32,
            ),
        ]

    return tf.data.Dataset.from_tensor_slices(tuple(tf_rand()))
