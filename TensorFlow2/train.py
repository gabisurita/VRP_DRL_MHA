from dataclasses import dataclass

import tensorflow as tf
from tqdm import tqdm
from time import time

from model import AttentionModel
from baseline import RolloutBaseline
from data import generate_data


def train(cfg, log_path=None):
    def rein_loss(model, inputs, bs, t):
        L, ll = model(inputs, decode_type="sampling", training=True)
        b = bs[t] if bs is not None else baseline.eval(inputs, L)
        b = tf.stop_gradient(b)
        return tf.reduce_mean((L - b) * ll), tf.reduce_mean(L)

    def grad_func(model, inputs, bs, t):
        with tf.GradientTape() as tape:
            loss, L_mean = rein_loss(model, inputs, bs, t)
        return loss, L_mean, tape.gradient(loss, model.trainable_variables)

    model = AttentionModel(
        cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping
    )
    baseline = RolloutBaseline(
        model,
        cfg.task,
        cfg.weight_dir,
        cfg.n_rollout_samples,
        cfg.embed_dim,
        cfg.n_customer,
        cfg.warmup_beta,
        cfg.wp_epochs,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
    ave_loss = tf.keras.metrics.Mean()
    ave_L = tf.keras.metrics.Mean()

    t1 = time()
    for epoch in range(cfg.epochs):
        dataset = generate_data(cfg.n_samples, cfg.n_customer)

        bs = baseline.eval_all(dataset, cfg.batch)
        bs = (
            tf.reshape(bs, (-1, cfg.batch)) if bs is not None else None
        )  # bs: (cfg.batch_steps, cfg.batch) or None

        for t, inputs in enumerate(dataset.batch(cfg.batch)):

            loss, L_mean, grads = grad_func(model, inputs, bs, t)

            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )  # optimizer.step

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
                if cfg.islogger:
                    if log_path is None:
                        log_path = "%s%s_%s.csv" % (
                            cfg.log_dir,
                            cfg.task,
                            cfg.dump_date,
                        )  # cfg.log_dir = ./Csv/
                        with open(log_path, "w") as f:
                            f.write("time,epoch,batch,loss,cost\n")
                    with open(log_path, "a") as f:
                        f.write(
                            "%dmin%dsec,%d,%d,%1.3f,%1.3f\n"
                            % (
                                (t2 - t1) // 60,
                                (t2 - t1) % 60,
                                epoch,
                                t,
                                ave_loss.result().numpy(),
                                ave_L.result().numpy(),
                            )
                        )
                t1 = time()

        baseline.epoch_callback(model, epoch)
        model.save_weights(
            "%s%s_epoch%s.h5" % (cfg.weight_dir, cfg.task, epoch), save_format="h5"
        )  # cfg.weight_dir = ./Weights/

        ave_loss.reset_states()
        ave_L.reset_states()


@dataclass
class Config:
    mode: str = "train"
    seed: int = 123
    n_customer: int = 20
    batch: int = 512
    batch_steps: int = 2500
    batch_verbose: int = 10
    n_rollout_samples: int = 10_000
    epochs: int = 20
    embed_dim: int = 128
    n_heads: int = 8
    tanh_clipping: float = 10.0
    n_encode_layers: int = 3
    lr: float = 0.0001
    warmup_beta: float = 0.8
    wp_epochs: int = 1
    islogger: bool = True
    log_dir: str = "./Csv/"
    weight_dir: str = "./Weights/"
    pkl_dir: str = "./Pkl/"
    cuda_dv: int = 0
    task: str = "VRP20_train"
    dump_date: str = "0726_03_41"
    pkl_path: str = "./Pkl/VRP20_train.pkl"
    n_samples: int = 1_280_000
    # min_size
    # max_size


if __name__ == "__main__":
    cfg = Config()
    # cfg.embed_dim = 32
    cfg.n_customer = 50
    cfg.n_rollout_samples = 1000

    # n_samples / batch must be integer.
    cfg.n_samples = 100_000
    cfg.batch = 500
    cfg.batch_steps = 2500
    cfg.learning_rate = 1e-3

    print("Start train")
    print(cfg.__dict__)
    train(cfg)
