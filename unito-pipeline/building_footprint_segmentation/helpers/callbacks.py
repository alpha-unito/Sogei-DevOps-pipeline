import datetime
import logging
import os
import shutil

import cv2
import torch
import warnings
import time
import json
import csv
import numpy as np

from py_oneliner import one_liner

from building_footprint_segmentation.utils import date_time
from building_footprint_segmentation.utils.operations import (
    is_overridden_func,
    make_directory,
)
from building_footprint_segmentation.utils.py_network import (
    adjust_model,
    gpu_variable,
    convert_tensor_to_numpy,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("segmentation")


class CallbackList(object):
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        if len(callbacks) != 0:
            [
                logger.debug("Registered {}".format(c.__class__.__name__))
                for c in callbacks
            ]

    def append(self, callback):
        logger.debug("Registered {}".format(callback.__class__.__name__))
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Epoch Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_epoch_begin):
                logger.debug(
                    "Nothing Registered On Epoch Begin {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Epoch End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_epoch_end):
                logger.debug(
                    "Nothing Registered On Epoch End {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            logger.debug("On Batch Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_batch_begin):
                logger.debug(
                    "Nothing Registered On Batch Begin {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):

        for callback in self.callbacks:
            logger.debug("On Batch End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_batch_end):
                logger.debug(
                    "Nothing Registered On Batch End {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_batch_end(batch, logs)

    def on_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_begin):
                logger.debug(
                    "Nothing Registered On Begin {}".format(callback.__class__.__name__)
                )
            callback.on_begin(logs)

    def on_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_end):
                logger.debug(
                    "Nothing Registered On End {}".format(callback.__class__.__name__)
                )
            callback.on_end(logs)

    def interruption(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("Interruption {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.interruption):
                logger.debug(
                    "Nothing Registered On Interruption {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.interruption(logs)

    def update_params(self, params):
        for callback in self.callbacks:
            if not is_overridden_func(callback.update_params):
                logger.debug(
                    "Nothing Registered On Update param {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.update_params(params)

    def __iter__(self):
        return iter(self.callbacks)


# class Callback(object):
#     def __init__(self, log_dir):
#         self.log_dir = os.path.join(
#             log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         )
class Callback(object):
    def __init__(self, log_dir, end_epoch=None, batch_size=None):
        folder_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if end_epoch is not None and batch_size is not None:
            folder_name = f"epoch{end_epoch}_batch{batch_size}_" + folder_name
        self.log_dir = os.path.join(log_dir, folder_name)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_begin(self, logs=None):
        pass

    def on_end(self, logs=None):
        pass

    def interruption(self, logs=None):
        pass

    def update_params(self, params):
        pass


# class TrainStateCallback(Callback):
#     """
#     Save the training state
#     """

#     def __init__(self, log_dir):
#         super().__init__(log_dir)
#         timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         state = make_directory(self.log_dir, "state")
#         self.chk = os.path.join(state, "default.pt")
#         self.best = os.path.join(state, "best.pt")

#         self.previous_best = None

class TrainStateCallback(Callback):
    """
    Save the training state
    """
    def __init__(self, log_dir, end_epoch=None, batch_size=None):
        super().__init__(log_dir, end_epoch, batch_size)
        state = make_directory(self.log_dir, "state")
        self.chk = os.path.join(state, "default.pt")
        self.best = os.path.join(state, "best.pt")

        self.previous_best = None


    def on_epoch_end(self, epoch, logs=None):
        valid_loss = logs["valid_loss"]
        my_state = logs["state"]
        if self.previous_best is None or valid_loss < self.previous_best:
            self.previous_best = valid_loss
            torch.save(my_state, str(self.best))
        torch.save(my_state, str(self.chk))
        logger.debug(
            "Successful on Epoch End {}, Saved State".format(self.__class__.__name__)
        )

    def interruption(self, logs=None):
        my_state = logs["state"]

        torch.save(my_state, str(self.chk))
        logger.debug(
            "Successful on Interruption {}, Saved State".format(self.__class__.__name__)
        )


class TensorBoardCallback(Callback):
    """
    Log tensor board events
    """

    def __init__(self, log_dir, end_epoch=None, batch_size=None):
        super().__init__(log_dir, end_epoch, batch_size)
        self.writer = SummaryWriter(make_directory(self.log_dir, "events"))

    def plt_scalar(self, y, x, tag):
        if type(y) is dict:
            self.writer.add_scalars(tag, y, global_step=x)
            self.writer.flush()
        else:
            self.writer.add_scalar(tag, y, global_step=x)
            self.writer.flush()

    def plt_images(self, img, global_step, tag):
        self.writer.add_image(tag, img, global_step)
        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        lr = logs["lr"]
        train_loss = logs["train_loss"]
        valid_loss = logs["valid_loss"]

        train_metric = logs["train_metric"]
        valid_metric = logs["valid_metric"]

        self.plt_scalar(lr, epoch, "LR/Epoch")
        self.plt_scalar(
            {"train_loss": train_loss, "valid_loss": valid_loss}, epoch, "Loss/Epoch"
        )

        metric_keys = list(train_metric.keys())
        for key in metric_keys:
            self.plt_scalar(
                {
                    "Train_{}".format(key): train_metric[key],
                    "Valid_{}".format(key): valid_metric[key],
                },
                epoch,
                "{}/Epoch".format(key),
            )

        logger.debug(
            "Successful on Epoch End {}, Data Plot".format(self.__class__.__name__)
        )

    def on_batch_end(self, batch, logs=None):
        img_data = logs["plt_img"] if "plt_img" in logs else None
        data = logs["plt_lr"]

        if img_data is not None:
            # self.plt_images(to_tensor(np.moveaxis(img_data["img"], -1, 0)), batch, img_data["tag"])
            pass

        self.plt_scalar(data["data"], batch, data["tag"])
        logger.debug(
            "Successful on Batch End {}, Data Plot".format(self.__class__.__name__)
        )


class SchedulerCallback(Callback):
    def __init__(self, scheduler):
        super().__init__(None)
        self.scheduler = scheduler

    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.step(epoch)
        logger.debug(
            "Successful on Epoch End {}, Lr Scheduled".format(self.__class__.__name__)
        )


class TimeCallback(Callback):
    def __init__(self, log_dir, end_epoch=None, batch_size=None):
        super().__init__(log_dir, end_epoch, batch_size)
        self.start_time = None

    def on_begin(self, logs=None):
        self.start_time = time.time()

    def on_end(self, logs=None):
        end_time = time.time()
        total_time = date_time.get_time(end_time - self.start_time)
        one_liner.one_line(
            tag="Run Time",
            tag_data=f"{total_time}",
            tag_color="cyan",
            to_reset_data=True,
            to_new_line_data=True,
        )

    def interruption(self, logs=None):
        end_time = time.time()
        total_time = date_time.get_time(end_time - self.start_time)
        one_liner.one_line(
            tag="Run Time",
            tag_data=f"{total_time}",
            tag_color="cyan",
            to_reset_data=True,
            to_new_line_data=True,
        )


class TrainChkCallback(Callback):
    def __init__(self, log_dir, end_epoch=None, batch_size=None):
        super().__init__(log_dir, end_epoch, batch_size)
        self.chk = os.path.join(make_directory(self.log_dir, "chk_pth"), "chk_pth.pt")

    def on_epoch_end(self, epoch, logs=None):
        my_state = logs["state"]
        torch.save(adjust_model(my_state["model"]), str(self.chk))
        logger.debug(
            "Successful on Epoch End {}, Chk Saved".format(self.__class__.__name__)
        )

    def interruption(self, logs=None):
        my_state = logs["state"]
        torch.save(adjust_model(my_state["model"]), str(self.chk))
        logger.debug(
            "Successful on interruption {}, Chk Saved".format(self.__class__.__name__)
        )


class TestDuringTrainingCallback(Callback):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.test_path = os.path.join(self.log_dir, "test_on_epoch_end")

    def on_epoch_end(self, epoch, logs=None):
        model = logs["model"]
        test_loader = logs["test_loader"]
        model.eval()
        try:
            if os.path.exists(self.test_path):
                shutil.rmtree(self.test_path)

            for i, test_data in enumerate(test_loader):
                self.inference(
                    model,
                    gpu_variable(test_data["images"]),
                    test_data["file_name"],
                    make_directory(
                        os.path.dirname(self.test_path),
                        os.path.basename(self.test_path),
                    ),
                    epoch,
                )
                break
        except Exception as ex:
            logger.exception("Skipped Exception in {}".format(self.__class__.__name__))
            logger.exception("Exception {}".format(ex))
            pass

    def inference(self, model, image, file_name, save_path, index):
        pass


class BinaryTestCallback(TestDuringTrainingCallback):
    def __init__(self, log_dir, threshold: float = 0.20):
        super().__init__(log_dir)
        self._threshold = threshold

    @torch.no_grad()
    def inference(self, model, image, file_name, save_path, index):
        """

        :param model: the model used for training
        :param image: the images loaded by the test loader
        :param file_name: the file name of the test image
        :param save_path: path where to save the image
        :param index:
        :return:
        """
        prediction = model(image)
        prediction = prediction.sigmoid()
        prediction[prediction >= self._threshold] = 1
        prediction[prediction < self._threshold] = 0

        batch, _, h, w = prediction.shape
        for i in range(batch):
            prediction_numpy = convert_tensor_to_numpy(prediction[i])
            prediction_numpy = prediction_numpy.reshape((h, w))
            cv2.imwrite(
                os.path.join(save_path, f"{file_name[i]}.png"), prediction_numpy * 255
            )


def load_default_callbacks(log_dir: str):
    return [
        TrainChkCallback(log_dir),
        TimeCallback(log_dir),
        TensorBoardCallback(log_dir),
        TrainStateCallback(log_dir),
    ]



# def load_callback(log_dir: str, callback: str) -> Callback:
#     """
#     :param log_dir:
#     :param callback:
#     :return:
#     """
#     return eval(callback)(log_dir)

def load_callback(log_dir: str, callback: str, end_epoch: int = None, batch_size: int = None) -> Callback:
    """
    Dynamically load a callback class and initialize it.

    :param log_dir: Base output directory
    :param callback: Name of the callback class
    :param end_epoch: Training end epoch (optional)
    :param batch_size: Training batch size (optional)
    :return: Initialized callback object
    """
    return eval(callback)(log_dir, end_epoch, batch_size)




# class MetricsLoggerCallback(Callback):
#     """
#     Save training metrics (accuracy, loss, etc.) into a log.txt file in log_dir.
#     """
#     def __init__(self, log_dir, end_epoch=None, batch_size=None):
#         self.log_dir = os.path.abspath(log_dir)
#         os.makedirs(self.log_dir, exist_ok=True)

#         self.log_file = os.path.join(self.log_dir, "log.txt")
#         with open(self.log_file, "w") as f:
#             f.write(f"# Training log (batch_size={batch_size}, end_epoch={end_epoch})\n")
#             f.write("# Columns: epoch, train_loss, valid_loss, accuracy, precision, recall, f1, iou\n")

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
        
#         metrics = ["train_loss", "valid_loss", "accuracy", "precision", "recall", "f1", "iou"]
#         safe_logs = {"epoch": epoch}

#         train_metric = logs.get("train_metric", {})
#         valid_metric = logs.get("valid_metric", {})

#         for m in metrics:
#             # Prefer direct logs, then train_metric, then valid_metric
#             v = logs.get(m)
#             if v is None:
#                 v = train_metric.get(m)
#             if v is None:
#                 v = valid_metric.get(m)
#             if isinstance(v, (np.integer, np.floating)):
#                 v = v.item()
#             safe_logs[m] = v

#         with open(self.log_file, "a") as f:
#             f.write(f"{json.dumps(safe_logs)}\n")





# class MetricsLoggerCallback(Callback):
#     """
#     Save training metrics into a CSV file in log_dir.
#     First column is a string with end_epoch and batch_size info.
#     """
#     def __init__(self, log_dir, end_epoch=None, batch_size=None):
#         self.log_dir = os.path.abspath(log_dir)
#         os.makedirs(self.log_dir, exist_ok=True)

#         self.end_epoch = end_epoch
#         self.batch_size = batch_size

#         self.log_file = os.path.join(self.log_dir, "metrics_log.csv")
#         # Define columns
#         self.columns = ["run", "epoch", "train_loss", "valid_loss", "accuracy", "precision", "recall", "f1", "iou"]

#         # Write header
#         with open(self.log_file, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=self.columns)
#             writer.writeheader()

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}

#         run_id = f"epoch{self.end_epoch}_batch{self.batch_size}"
#         train_metric = logs.get("train_metric", {})
#         valid_metric = logs.get("valid_metric", {})

#         row = {"run": run_id, "epoch": epoch}
#         for col in self.columns[2:]:  # skip run and epoch
#             v = logs.get(col)
#             if v is None:
#                 v = train_metric.get(col)
#             if v is None:
#                 v = valid_metric.get(col)
#             if isinstance(v, (np.integer, np.floating)):
#                 v = v.item()
#             row[col] = v

#         # Append row to CSV
#         with open(self.log_file, "a", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=self.columns)
#             writer.writerow(row)


#ONE CSV FILE WHERE ALL SCATTERS WERE WRITING

# class MetricsLoggerCallback(Callback):
#     """
#     Save training metrics into a CSV file in log_dir.
#     First column is a string with end_epoch and batch_size info.
#     Appends to existing CSV if present.
#     """
#     def __init__(self, log_dir, end_epoch=None, batch_size=None):
#         self.log_dir = os.path.abspath(log_dir)
#         os.makedirs(self.log_dir, exist_ok=True)

#         self.end_epoch = end_epoch
#         self.batch_size = batch_size

#         self.log_file = os.path.join(self.log_dir, "metrics_log.csv")
#         # Define columns
#         self.columns = ["run", "epoch", "train_loss", "valid_loss", "accuracy", "precision", "recall", "f1", "iou"]

#         # Write header only if file doesn't exist
#         if not os.path.exists(self.log_file):
#             with open(self.log_file, "w", newline="") as f:
#                 writer = csv.DictWriter(f, fieldnames=self.columns)
#                 writer.writeheader()

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}

#         run_id = f"epoch{self.end_epoch}_batch{self.batch_size}"
#         train_metric = logs.get("train_metric", {})
#         valid_metric = logs.get("valid_metric", {})

#         row = {"run": run_id, "epoch": epoch}
#         for col in self.columns[2:]:  # skip run and epoch
#             v = logs.get(col)
#             if v is None:
#                 v = train_metric.get(col)
#             if v is None:
#                 v = valid_metric.get(col)
#             if isinstance(v, (np.integer, np.floating)):
#                 v = v.item()
#             row[col] = v

#         # Append row to CSV
#         with open(self.log_file, "a", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=self.columns)
#             writer.writerow(row)


class MetricsLoggerCallback(Callback):
    """
    Save training metrics into a CSV file in log_dir.
    First column is a string with end_epoch and batch_size info.
    Appends to existing CSV if present.
    """
    def __init__(self, log_dir, end_epoch=None, batch_size=None):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        self.end_epoch = end_epoch
        self.batch_size = batch_size

        # Unique CSV per run
        self.log_file = os.path.join(self.log_dir,
            f"metrics_epoch{self.end_epoch}_batch{self.batch_size}.csv"
        )
        # Define columns
        self.columns = ["run", "epoch", "train_loss", "valid_loss", "accuracy", "precision", "recall", "f1", "iou"]

        # Always create new file with header
        with open(self.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        run_id = f"epoch{self.end_epoch}_batch{self.batch_size}"
        train_metric = logs.get("train_metric", {})
        valid_metric = logs.get("valid_metric", {})

        row = {"run": run_id, "epoch": epoch}
        for col in self.columns[2:]:  # skip run and epoch
            v = logs.get(col)
            if v is None:
                v = train_metric.get(col)
            if v is None:
                v = valid_metric.get(col)
            if isinstance(v, (np.integer, np.floating)):
                v = v.item()
            row[col] = v

        # Append row to CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)