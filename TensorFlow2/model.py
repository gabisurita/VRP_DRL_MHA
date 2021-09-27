import tensorflow as tf
import tensorflow.keras.backend as K

from decoder_utils import TopKSampler, CategoricalSampler, Env
import numpy as np


class DotProductAttention(tf.keras.layers.Layer):
    def __init__(
        self, clip=None, return_logits=False, head_depth=16, inf=1e10, **kwargs
    ):
        super().__init__(**kwargs)
        self.clip = clip
        self.return_logits = return_logits
        self.inf = inf
        dk = tf.cast(head_depth, tf.float32)
        self.scale = tf.math.sqrt(dk)

    def call(self, x, mask=None):
        """Q: (batch, n_heads, q_seq(=n_nodes or =1), head_depth)
        K: (batch, n_heads, k_seq(=n_nodes), head_depth)
        logits: (batch, n_heads, q_seq(this could be 1), k_seq)
        mask: (batch, n_nodes, 1), e.g. tf.Tensor([[ True], [ True], [False]])
        mask[:,None,None,:,0]: (batch, 1, 1, n_nodes) ==> broadcast depending on logits shape
        [True] -> [1 * -np.inf], [False] -> [logits]
        """
        Q, K, V = x
        logits = tf.matmul(Q, K, transpose_b=True) / self.scale

        if self.clip is not None:
            logits = self.clip * tf.math.tanh(logits)

        if self.return_logits:
            if mask is not None:
                logits = tf.where(
                    tf.transpose(mask, perm=(0, 2, 1)),
                    tf.ones_like(logits) * (-np.inf),
                    logits,
                )
            return logits

        if mask is not None:
            logits = tf.where(
                mask[:, None, None, :, 0], tf.ones_like(logits) * (-np.inf), logits
            )

        probs = tf.nn.softmax(logits, axis=-1)
        return tf.matmul(probs, V)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        n_heads=8,
        embed_dim=128,
        clip=None,
        return_logits=None,
        need_W=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_depth = self.embed_dim // self.n_heads
        if self.embed_dim % self.n_heads != 0:
            raise ValueError("embed_dim = n_heads * head_depth")

        self.need_W = need_W
        self.attention = DotProductAttention(
            clip=clip, return_logits=return_logits, head_depth=self.head_depth
        )
        if self.need_W:
            # stdv = 1./tf.math.sqrt(tf.cast(embed_dim, tf.float32))
            # init = tf.keras.initializers.RandomUniform(minval = -stdv, maxval = stdv)# init = tf.random_uniform_initializer(minval = -stdv, maxval= stdv)
            self.Wk = tf.keras.layers.Dense(
                self.embed_dim, use_bias=False
            )  # torch.nn.Linear(embed_dim, embed_dim)
            self.Wv = tf.keras.layers.Dense(self.embed_dim, use_bias=False)
            self.Wq = tf.keras.layers.Dense(self.embed_dim, use_bias=False)
            self.Wout = tf.keras.layers.Dense(self.embed_dim, use_bias=False)

    def split_heads(self, T, batch):
        """https://qiita.com/halhorn/items/c91497522be27bde17ce
        T: (batch, n_nodes, self.embed_dim)
        T reshaped: (batch, n_nodes, self.n_heads, self.head_depth)
        return: (batch, self.n_heads, n_nodes, self.head_depth)
        """
        T = tf.reshape(T, (batch, -1, self.n_heads, self.head_depth))
        return tf.transpose(T, perm=(0, 2, 1, 3))

    def combine_heads(self, T, batch):
        """T: (batch, self.n_heads, n_nodes, self.head_depth)
        T transposed: (batch, n_nodes, self.n_heads, self.head_depth)
        return: (batch, n_nodes, self.embed_dim)
        """
        T = tf.transpose(T, perm=(0, 2, 1, 3))
        return tf.reshape(T, (batch, -1, self.embed_dim))

    def call(self, x, mask=None):
        """q, k, v = x
        encoder arg x: [x, x, x]
        shape of q: (batch, n_nodes, embed_dim)
        output[0] - output[h_heads-1]: (batch, n_nodes, head_depth)
        --> concat output: (batch, n_nodes, head_depth * h_heads)
        return output: (batch, n_nodes, embed_dim)
        """
        Q, K, V = x
        if self.need_W:
            Q, K, V = self.Wq(Q), self.Wk(K), self.Wv(V)

        batch = K.shape[0]
        output = self.attention(
            [self.split_heads(T, batch) for T in [Q, K, V]], mask=mask
        )
        output = self.combine_heads(output, batch)
        if self.need_W:
            return self.Wout(output)
        return output


class ResidualBlock_BN(tf.keras.layers.Layer):
    def __init__(self, MHA, BN, **kwargs):
        super().__init__(**kwargs)
        self.MHA = MHA
        self.BN = BN

    def call(self, x, mask=None, training=True):
        if mask is None:
            return self.BN(x + self.MHA(x), training=training)
        return self.BN(x + self.MHA(x, mask), training=training)


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, MHA, **kwargs):
        super().__init__(**kwargs)
        self.MHA = MHA

    def call(self, x, mask=None):
        return self.MHA([x, x, x], mask=mask)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, n_heads=8, FF_hidden=512, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.FF_hidden = FF_hidden
        self.activation = activation
        self.BN1 = tf.keras.layers.BatchNormalization(trainable=True)
        self.BN2 = tf.keras.layers.BatchNormalization(trainable=True)

    def build(self, input_shape):
        self.MHA_sublayer = ResidualBlock_BN(
            SelfAttention(
                MultiHeadAttention(
                    n_heads=self.n_heads, embed_dim=input_shape[2], need_W=True
                )  # input_shape[2] = embed_dim = 128
            ),
            self.BN1,
        )
        self.FF_sublayer = ResidualBlock_BN(
            tf.keras.models.Sequential(
                [
                    # tf.keras.layers.Dense(self.FF_hidden, use_bias = True, activation = self.activation, kernel_initializer = init, bias_initializer = init),
                    # tf.keras.layers.Dense(input_shape[2], use_bias = True, kernel_initializer = init, bias_initializer = init)
                    tf.keras.layers.Dense(
                        self.FF_hidden, use_bias=True, activation=self.activation
                    ),
                    tf.keras.layers.Dense(input_shape[2], use_bias=True),
                ]
            ),
            self.BN2,
        )
        super().build(input_shape)

    def call(self, x, mask=None, training=True):
        """arg x: (batch, n_nodes, embed_dim)
        return: (batch, n_nodes, embed_dim)
        """
        return self.FF_sublayer(
            self.MHA_sublayer(x, mask=mask, training=training), training=training
        )


class GraphAttentionEncoder(tf.keras.models.Model):
    def __init__(self, embed_dim=128, n_heads=8, n_layers=3, FF_hidden=512):
        super().__init__()
        # stdv = 1./tf.math.sqrt(tf.cast(embed_dim, tf.float32))
        # init = tf.keras.initializers.RandomUniform(minval = -stdv, maxval = stdv)
        # self.init_W_depot = tf.keras.layers.Dense(embed_dim, use_bias = True, kernel_initializer = init, bias_initializer = init)# torch.nn.Linear(2, embedding_dim)
        # self.init_W = tf.keras.layers.Dense(embed_dim, use_bias = True, kernel_initializer = init, bias_initializer = init)# torch.nn.Linear(3, embedding_dim)
        self.init_W_depot = tf.keras.layers.Dense(
            embed_dim, use_bias=True
        )  # torch.nn.Linear(2, embedding_dim)
        self.init_W = tf.keras.layers.Dense(
            embed_dim, use_bias=True
        )  # torch.nn.Linear(3, embedding_dim)
        self.encoder_layers = [
            EncoderLayer(n_heads, FF_hidden) for _ in range(n_layers)
        ]

    @tf.function
    def call(self, x, mask=None, training=True):
        """x[0] -- depot_xy: (batch, 2) --> embed_depot_xy: (batch, embed_dim)
        x[1] -- customer_xy: (batch, n_nodes-1, 2)
        x[2] -- demand: (batch, n_nodes-1)
        --> concated_customer_feature: (batch, n_nodes-1, 3) --> embed_customer_feature: (batch, n_nodes-1, embed_dim)
        embed_x(batch, n_nodes, embed_dim)

        return: (node embeddings(= embedding for all nodes), graph embedding(= mean of node embeddings for graph))
                =((batch, n_nodes, embed_dim), (batch, embed_dim))
        """
        x = tf.concat(
            [
                self.init_W_depot(x[0])[:, None, :],
                self.init_W(tf.concat([x[1], x[2][:, :, None]], axis=-1)),
            ],
            axis=1,
        )

        for layer in self.encoder_layers:
            x = layer(x, mask, training)

        return (x, tf.reduce_mean(x, axis=1))


class DecoderCell(tf.keras.models.Model):
    def __init__(self, embed_dim=128, n_heads=8, clip=10.0, **kwargs):
        super().__init__(**kwargs)

        self.Wk1 = tf.keras.layers.Dense(
            embed_dim, use_bias=False
        )  # torch.nn.Linear(embed_dim, embed_dim, bias = False)
        self.Wv = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.Wk2 = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.Wq_fixed = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.Wout = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.Wq_step = tf.keras.layers.Dense(embed_dim, use_bias=False)

        self.MHA = MultiHeadAttention(
            n_heads=n_heads, embed_dim=embed_dim, need_W=False
        )
        self.SHA = DotProductAttention(
            clip=clip, return_logits=True, head_depth=embed_dim
        )
        # SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
        self.env = Env

    # @tf.function
    def compute_static(self, node_embeddings, graph_embedding):
        Q_fixed = self.Wq_fixed(graph_embedding[:, None, :])
        K1 = self.Wk1(node_embeddings)
        V = self.Wv(node_embeddings)
        K2 = self.Wk2(node_embeddings)
        return Q_fixed, K1, V, K2

    @tf.function
    def _compute_mha(self, Q_fixed, step_context, K1, V, K2, mask):
        Q_step = self.Wq_step(step_context)
        Q1 = Q_fixed + Q_step
        Q2 = self.MHA([Q1, K1, V], mask=mask)
        Q2 = self.Wout(Q2)
        logits = self.SHA([Q2, K2, None], mask=mask)
        return tf.squeeze(logits, axis=1)

    # @tf.function
    def call(self, x, encoder_output, return_pi=False, decode_type="sampling"):
        """context: (batch, 1, 2*embed_dim+1)
        tf.concat([graph embedding[:,None,:], previous node embedding, remaining vehicle capacity[:,:,None]], axis = -1)
        encoder output
        ==> graph embedding: (batch, embed_dim)
        ==> node_embeddings: (batch, n_nodes, embed_dim)
        previous node embedding: (batch, n_nodes, embed_dim)
        remaining vehicle capacity(= D): (batch, 1)

        mask: (batch, n_nodes, 1), dtype = tf.bool, [True] --> [-inf], [False] --> [logits]
        context: (batch, 1, 2*embed_dim+1)

        squeezed logits: (batch, n_nodes), logits denotes the value before going into softmax
        next_node: (batch, 1), minval = 0, maxval = n_nodes-1, dtype = tf.int32
        log_p: (batch, n_nodes) <-- squeezed logits: (batch, n_nodes), log(exp(x_i) / exp(x).sum())
        """
        node_embeddings, graph_embedding = encoder_output
        Q_fixed, K1, V, K2 = self.compute_static(node_embeddings, graph_embedding)

        env = Env(x, node_embeddings)
        mask, step_context, D = env._create_t1()

        selecter = {"greedy": TopKSampler(), "sampling": CategoricalSampler()}.get(
            decode_type, None
        )
        log_ps = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True,
            element_shape=(env.batch, env.n_nodes),
        )
        tours = tf.TensorArray(
            dtype=tf.int32, size=0, dynamic_size=True, element_shape=(env.batch,)
        )

        for i in tf.range(env.n_nodes * 2):
            logits = self._compute_mha(Q_fixed, step_context, K1, V, K2, mask)
            log_p = tf.nn.log_softmax(logits, axis=-1)
            next_node = selecter(log_p)
            mask, step_context, D = env._get_step(next_node, D)

            tours = tours.write(i, tf.squeeze(next_node, axis=1))
            log_ps = log_ps.write(i, log_p)

        pi = tf.transpose(tours.stack(), perm=(1, 0))
        ll = env.get_log_likelihood(tf.transpose(log_ps.stack(), perm=(1, 0, 2)), pi)
        cost = env.get_costs(pi)

        if return_pi:
            return cost, ll, pi
        return cost, ll


class AttentionModel(tf.keras.models.Model):
    def __init__(
        self,
        embed_dim=128,
        n_encode_layers=3,
        n_heads=8,
        tanh_clipping=10.0,
        FF_hidden=512,
    ):
        super().__init__()

        self.Encoder = GraphAttentionEncoder(
            embed_dim, n_heads, n_encode_layers, FF_hidden
        )
        self.Decoder = DecoderCell(embed_dim, n_heads, tanh_clipping)

    def call(self, x, training=True, return_pi=False, decode_type="greedy"):
        encoder_output = self.Encoder(x, training=training)
        decoder_output = self.Decoder(
            x, encoder_output, return_pi=return_pi, decode_type=decode_type
        )
        if return_pi:
            cost, ll, pi = decoder_output
            return cost, ll, pi
        cost, ll = decoder_output
        return cost, ll
