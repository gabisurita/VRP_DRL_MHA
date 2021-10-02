from dataclasses import dataclass

import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, embed_dim=128, ff_dim=512, **kwargs):
        super().__init__(**kwargs)

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
        )

        self.mlp = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )

        self.bn1 = tf.keras.layers.BatchNormalization(trainable=True)
        self.bn2 = tf.keras.layers.BatchNormalization(trainable=True)

    def call(self, x, mask=None, training=True):
        mha_out = self.mha(x, x, attention_mask=mask, training=training)
        bn1_out = self.bn1(x + mha_out, training=training)
        mlp_out = self.mlp(bn1_out, training=training)
        bn2_out = self.bn2(x + mlp_out, training=training)

        return bn2_out


class GraphTransformerEncoder(tf.keras.models.Model):
    def __init__(self, embed_dim=128, num_heads=8, n_layers=3, ff_dim=512):
        super().__init__()

        self.encode_depot = tf.keras.layers.Dense(embed_dim, use_bias=True)
        self.encode_customers = tf.keras.layers.Dense(embed_dim, use_bias=True)
        self.encoder_layers = [
            EncoderLayer(
                num_heads=num_heads,
                embed_dim=embed_dim,
                ff_dim=ff_dim,
            )
            for _ in range(n_layers)
        ]

    @tf.function
    def call(self, x, mask=None, training=True):
        depot_xy, customer_xy, demand = x

        x = tf.concat(
            [
                self.encode_depot(depot_xy)[:, None, :],
                self.encode_customers(
                    tf.concat([customer_xy, demand[:, :, None]], axis=-1)
                ),
            ],
            axis=1,
        )

        for layer in self.encoder_layers:
            x = layer(x, mask, training)

        return x


class PointerDecoder(tf.keras.models.Model):
    def __init__(self, embed_dim=128, num_heads=8, clip=10.0, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.clip = clip

        self.Wq_fixed = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.Wq_step = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.Wk = tf.keras.layers.Dense(embed_dim, use_bias=False)

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
        )

    @tf.function
    def call(self, node_embeddings, step_context, mask):
        Q_step = self.Wq_step(step_context)
        Q_fixed = self.Wq_fixed(tf.reduce_sum(node_embeddings, axis=1, keepdims=True))

        Q1 = Q_fixed + Q_step
        Q2 = self.mha(
            query=Q1,
            key=node_embeddings,
            value=node_embeddings,
            attention_mask=tf.transpose(mask, (0, 2, 1)),
        )
        K2 = self.Wk(node_embeddings)

        depth = tf.cast(self.embed_dim, tf.float32)
        logits = tf.matmul(Q2, K2, transpose_b=True) / tf.math.sqrt(depth)
        logits = self.clip * tf.math.tanh(logits)

        logits = tf.where(
            tf.transpose(mask, perm=(0, 2, 1)),
            tf.ones_like(logits) * -np.inf,
            logits,
        )

        return tf.squeeze(logits, axis=1)


class AttentionModel(tf.keras.models.Model):
    def __init__(
        self,
        embed_dim=128,
        n_encode_layers=3,
        num_heads=8,
        tanh_clipping=10.0,
        ff_dim=512,
    ):
        super().__init__()
        self.encoder = GraphTransformerEncoder(
            embed_dim, num_heads, n_encode_layers, ff_dim
        )
        self.decoder = PointerDecoder(embed_dim, num_heads, tanh_clipping)

    def encode(self, x, training=True):
        return self.encoder(x, training=training)

    def decode(self, x, node_embeddings, mask=None, training=True):
        return self.decoder(x, node_embeddings, mask, training=training)

    def call(self, x, **kwargs):
        return solve(self, x, **kwargs)


def get_costs(coords, route):
    d = tf.gather(coords, indices=route, batch_dims=1)
    return (
        tf.reduce_sum(tf.norm(d[:, 1:] - d[:, :-1], ord=2, axis=2), axis=1)
        + tf.norm(d[:, 0] - coords[:, 0], ord=2, axis=1)
        + tf.norm(d[:, -1] - coords[:, 0], ord=2, axis=1)
    )


def solve(net, problems, deterministic=False, training=True):
    depot_xy, customer_xy, demand = problems

    # Encode the problems.
    node_embeddings = net.encode(problems, training=training)

    bsz, n_nodes, _ = node_embeddings.shape
    demand = tf.concat([tf.zeros((bsz, 1)), demand], axis=-1)
    xy = tf.concat([depot_xy[:, None, :], customer_xy], 1)

    # Initialize visited state varible.
    visited = tf.concat(
        [
            tf.ones((bsz, 1), dtype=tf.bool),
            tf.zeros((bsz, n_nodes - 1), dtype=tf.bool),
        ],
        axis=-1,
    )

    # Capacities.
    capacities = tf.ones([bsz, 1], dtype=tf.float32)

    # Initial mask.
    mask = visited

    # Decoder context.
    step_context = tf.concat(
        [node_embeddings[:, 0, None], capacities[:, :, None]], axis=-1
    )

    log_ps = tf.TensorArray(
        dtype=tf.float32,
        size=0,
        dynamic_size=True,
        element_shape=(bsz, n_nodes),
    )

    tours = tf.TensorArray(
        dtype=tf.int32, size=0, dynamic_size=True, element_shape=(bsz,)
    )

    for i in range(n_nodes * 2):

        logits = net.decode(node_embeddings, step_context, mask[:, :, None])

        if deterministic:
            next_node = tf.math.top_k(logits, 1).indices
        else:
            next_node = tf.random.categorical(logits, 1, dtype=tf.int32)

        visited = visited | tf.cast(tf.one_hot(next_node, n_nodes)[:, 0, :], tf.bool)

        # If visiting the depot, refill capacity.
        is_next_depot = next_node == 0
        capacities = tf.where(
            is_next_depot, 1.0, capacities - tf.gather(demand, next_node, batch_dims=1)
        )

        capacity_over_customer = demand > capacities
        # mask_customer = capacity_over_customer[:, :, None] | visited_customer
        mask_customer = capacity_over_customer | visited

        mask_depot = is_next_depot[:, 0] & (
            tf.reduce_sum(tf.cast(mask_customer == False, tf.int32), axis=1) > 0
        )

        mask = tf.concat([mask_depot[:, None], mask_customer[:, 1:]], axis=-1)

        prev_node_embedding = tf.gather(
            node_embeddings, indices=next_node, batch_dims=1
        )

        step_context = tf.concat([prev_node_embedding, capacities[:, :, None]], axis=-1)

        tours = tours.write(i, tf.squeeze(next_node, axis=1))

        log_p = tf.nn.log_softmax(logits, axis=-1)
        log_ps = log_ps.write(i, log_p)

    def get_log_likelihood(log_p, pi):
        log_p = tf.gather_nd(log_p, tf.expand_dims(pi, axis=-1), batch_dims=2)
        return tf.reduce_sum(log_p, 1)

    routes = tf.transpose(tours.stack(), perm=(1, 0))

    ll = get_log_likelihood(tf.transpose(log_ps.stack(), perm=(1, 0, 2)), routes)
    cost = get_costs(xy, routes)

    return cost, ll, routes
