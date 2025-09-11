import numpy as np
from keras.callbacks import Callback


class EarlyStoppingOnOverfitting(Callback):
    """
    Stop training when there is suspicion of overfitting,
    i.e., when val_loss is much greater than training loss.
    
    Arguments:
        threshold: Maximum allowed difference between val_loss and loss.
                   If exceeded, training stops.
    """
    def __init__(self, threshold=0.5):
        super(EarlyStoppingOnOverfitting, self).__init__()
        self.threshold = threshold
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        if loss is not None and val_loss is not None:
            if (val_loss - loss) > self.threshold:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f"\nEpoch {epoch+1}: early stopping due to suspected overfitting "
                      f"(val_loss={val_loss:.4f}, loss={loss:.4f}, diff={val_loss-loss:.4f})")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Training stopped at epoch {self.stopped_epoch+1} due to overfitting suspicion.")

