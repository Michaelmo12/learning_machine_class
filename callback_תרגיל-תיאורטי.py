import numpy as np
import keras

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when there is suspected overfitting:
    the gap (val_loss - loss) stays above max_gap for 'patience' epochs.
    """

    def __init__(self, patience=0, max_gap=0.05):
        super().__init__()
        self.patience = patience
        self.max_gap = float(max_gap)
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0              # כמה אפוקים המתנו מאז שהיה בסדר
        self.stopped_epoch = 0
        self.best = np.inf         # נשמור את val_loss הטוב ביותר

    def on_epoch_end(self, epoch, logs=None):
        current_train = logs.get("loss")
        current_val = logs.get("val_loss")
        if current_train is None or current_val is None:
            return  # בלי val_loss אין בדיקת overfitting

        gap = current_val - current_train

        # עדכון נקודת השיא לפי val_loss
        if np.less(current_val, self.best):
            self.best = current_val
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            # אם הפער גדול מהסף – מונים; אחרת מאפסים
            if gap > self.max_gap:
                self.wait += 1
            else:
                self.wait = 0

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.best_weights is not None:
                print("Restoring model weights from the end of the best epoch (by val_loss).")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")


def get_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(10,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")
        msg = f"epoch {epoch + 1}: loss={loss:.4f}"
        if acc is not None:
            msg += f", acc={acc:.4f}"
        if val_loss is not None:
            msg += f", val_loss={val_loss:.4f}"
        if val_acc is not None:
            msg += f", val_acc={val_acc:.4f}"
        print(msg)


# reproducibility
SEED = 1337
rng = np.random.default_rng(SEED)

# create a learnable synthetic dataset instead of random labels
# so that accuracy can meaningfully improve

def make_dataset(n=4000, d=10, noise=0.6):
    X = rng.normal(size=(n, d))
    w = rng.normal(size=(d,))
    b = rng.normal()
    logits = X @ w + b + rng.normal(scale=noise, size=n)
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(np.int32)
    return X, y.reshape(-1, 1)

X, y = make_dataset(n=4000, d=10, noise=0.6)

# train/val split
idx = rng.permutation(len(X))
train_sz = int(0.8 * len(X))
train_idx, val_idx = idx[:train_sz], idx[train_sz:]
x_train, y_train = X[train_idx], y[train_idx]
x_val, y_val = X[val_idx], y[val_idx]



model = get_model()
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=30,
    verbose=0,
    callbacks=[
        LossAndErrorPrintingCallback(),
        EarlyStoppingAtMinLoss(patience=5, max_gap=0.05),
    ],
)