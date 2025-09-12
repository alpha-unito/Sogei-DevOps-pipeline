import os
import sys
import json
import albumentations as A
import argparse

#ho aggiunto questo per avere end_epoch e batch come parametro (anche import argpars è per questo) 
parser = argparse.ArgumentParser()
parser.add_argument("--end_epoch", type=int, default=1, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size for the data loader")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to store outputs")
args = parser.parse_args()

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from building_footprint_segmentation.segmentation import init_segmentation, read_trainer_config
from building_footprint_segmentation.helpers.callbacks import CallbackList, load_callback
from building_footprint_segmentation.trainer import Trainer

segmentation = init_segmentation("binary")


augmenters = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)
])

model = segmentation.load_model(name="ReFineNet")
criterion = segmentation.load_criterion(name="BinaryCrossEntropy")
loader = segmentation.load_loader(
    root_folder=r"/Users/dorianaunito/PostDoc/PROJECTS/SOGEI-DevOps/venv/building-footprint-segmentation/data",
    image_normalization="divide_by_255",
    label_normalization="binary_label",
    augmenters=augmenters,
    batch_size=args.batch_size,
)
metrics = segmentation.load_metrics(
    data_metrics=["accuracy", "precision", "f1", "recall", "iou"]
)

optimizer = segmentation.load_optimizer(model, name="Adam")

output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
os.makedirs(output_dir, exist_ok=True)

# callbacks = CallbackList()
# # Ouptut from all the callbacks caller will be stored at the path specified in log_dir
# for caller in  ["TrainChkCallback", "TimeCallback", "TensorBoardCallback", "TrainStateCallback"]:
#     # callbacks.append(load_callback(r"/Users/dorianaunito/PostDoc/PROJECTS/SOGEI-DevOps/venv/building-footprint-segmentation/out_data", caller))
#     callbacks.append(load_callback(output_dir, caller))



# callbacks = CallbackList()
# for caller in ["TrainChkCallback", "TimeCallback", "TensorBoardCallback", "TrainStateCallback"]:
#     callbacks.append(load_callback(output_dir, caller, args.end_epoch, args.batch_size))

#adding loggs
callbacks = CallbackList()
for caller in ["TrainChkCallback", "TimeCallback", "TensorBoardCallback", "TrainStateCallback", "MetricsLoggerCallback"]:
    callbacks.append(load_callback(output_dir, caller, args.end_epoch, args.batch_size))



trainer = Trainer(
    model=model,
    criterion=criterion,
    loader=loader,
    metrics=metrics,
    callbacks=callbacks,
    optimizer=optimizer,
    scheduler=None,
)
#Ho cambiato qua per avere end_epoch come parametro
trainer.train(start_epoch=0, end_epoch=args.end_epoch)



