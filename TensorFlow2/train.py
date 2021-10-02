from dataclasses import dataclass
from typing import Optional

import tensorflow as tf
from tqdm import tqdm
from time import time
from scipy.stats import ttest_rel

from model import AttentionModel, solve
from data import generate_data


@dataclass
class Config:
    seed: int = 123
    n_customer: int = 20
    batch: int = 512
    batch_steps: int = 2500
    batch_verbose: int = 10
    n_rollout_samples: int = 10_000
    epochs: int = 20
    embed_dim: int = 128
    num_heads: int = 8
    tanh_clipping: float = 10.0
    n_encode_layers: int = 3
    lr: float = 1e-4
    warmup_epochs: int = 1
    weight_dir: str = "./Weights/"
    task: str = "VRP50_finetune"
    dump_date: str = "0726_03_41"
    n_samples: int = 1_280_000

    load_path: Optional[str] = None
    """Preload a previous checkpoint."""

    # min_size
    # max_size


def eval(model, dataset, batch=1000):
    costs_list = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True, element_shape=(batch,)
    )

    for i, inputs in enumerate(dataset.batch(batch)):
        cost, _, _ = solve(model, inputs, deterministic=True, training=False)
        costs_list = costs_list.write(i, cost)

    return tf.reshape(costs_list.stack(), (-1,))


def train(cfg):

    # Initialize the candidate model.
    model = AttentionModel(
        cfg.embed_dim, cfg.n_encode_layers, cfg.num_heads, cfg.tanh_clipping
    )

    # Initialize the baseline model with random weights.
    baseline = AttentionModel(
        cfg.embed_dim, cfg.n_encode_layers, cfg.num_heads, cfg.tanh_clipping
    )

    # If a checkpoint is passed, preload weights from it.
    if cfg.load_path:

        # We must call the model at least once before loading.
        load_dataset = generate_data(n_samples=1, n_customer=cfg.n_customer)

        eval(model, load_dataset, 1)
        eval(baseline, load_dataset, 1)

        model.load_weights(cfg.load_path)
        baseline.load_weights(cfg.load_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

    # Initialize the running Keras metrics.
    ave_loss = tf.keras.metrics.Mean()
    ave_L = tf.keras.metrics.Mean()

    # Sample a random evaluation dataset.
    eval_dataset = generate_data(
        n_samples=cfg.n_rollout_samples, n_customer=cfg.n_customer
    )

    t1 = time()
    for epoch in range(cfg.epochs):

        # Sample the training data.
        dataset = (
            generate_data(cfg.n_samples, cfg.n_customer)
            .batch(cfg.batch)
            .prefetch(tf.data.AUTOTUNE)
        )

        for t, inputs in enumerate(dataset):

            # # Warmup mean baseline.
            if cfg.load_path is None and epoch < cfg.warmup_epochs:
                # bs, _ = model(inputs, decode_type="sampling")
                bs, _, _ = solve(model, inputs)
                bs = tf.reduce_mean(bs)

            # Greedy rollout baseline.
            else:
                # bs, _ = baseline(inputs, decode_type="greedy")
                bs, _, _ = solve(baseline, inputs, deterministic=True, training=False)

            # Really make sure there's no gradient in the baseline.
            bs = tf.stop_gradient(bs)

            with tf.GradientTape() as tape:
                # Forward pass.
                # L, ll = model(inputs, decode_type="sampling", training=True)
                L, ll, _ = solve(model, inputs)

                # Compute REINFORCE loss.
                loss = tf.reduce_mean((L - bs) * ll)

            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)

            L_mean = tf.reduce_mean(L)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            ave_loss.update_state(loss)
            ave_L.update_state(L_mean)

            if t % (cfg.batch_verbose) == 0:
                t2 = time()
                print(
                    "Epoch %d (batch = %d): Loss: %1.3f L: %1.3f, %dmin%dsec"
                    % (
                        epoch,
                        t,
                        ave_loss.result().numpy(),
                        ave_L.result().numpy(),
                        (t2 - t1) // 60,
                        (t2 - t1) % 60,
                    )
                )
                t1 = time()

        # Evaluate candidate.
        candidate_vals = eval(model, eval_dataset)
        candidate_mean = tf.reduce_mean(candidate_vals)

        baseline_vals = eval(baseline, eval_dataset)
        baseline_mean = tf.reduce_mean(baseline_vals)

        # Check if the candidate mean value is better.
        candidate_better = candidate_mean < baseline_mean

        # Check if improvement is significant with a one-sided t-test.
        _, p = ttest_rel(candidate_vals, baseline_vals)
        statistically_significant = (p / 2) < 0.05

        print(
            f"Epoch {epoch} candidate mean {candidate_mean}, baseline mean {baseline_mean}, p-value {p / 2}"
        )

        # If model is statiscally better than the baseline, copy model parameters.
        if candidate_better and statistically_significant:
            print("New baseline!")
            for a, b in zip(baseline.variables, model.variables):
                a.assign(b)

            eval_dataset = generate_data(
                n_samples=cfg.n_rollout_samples, n_customer=cfg.n_customer
            )

        # Save the model checkpoint.
        model.save_weights(
            "%s%s_epoch%s.h5" % (cfg.weight_dir, cfg.task, epoch), save_format="h5"
        )

        ave_loss.reset_states()
        ave_L.reset_states()


if __name__ == "__main__":
    cfg = Config()
    cfg.embed_dim = 128
    cfg.n_customer = 50
    # cfg.n_rollout_samples = 10_000

    # n_samples / batch must be integer.
    cfg.n_samples = 12_800  # 100_000
    cfg.batch_verbose = 10
    cfg.batch = 128
    cfg.batch_steps = 2500
    cfg.learning_rate = 5e-4
    cfg.task = "VRP50_refac"
    # cfg.load_path = "./Weights/VRP50_large_pretrained.h5"

    print("Start train")
    print(cfg.__dict__)
    train(cfg)
