import fiftyone as fo
import fiftyone.zoo as foz
#you can change the number of images per your choice to train
#Load 1000 training images from COCO-2017 dataset
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    max_samples=1000,
    dataset_dir="coco_1000",  # Folder where data will be stored
)

# Export images to a flat image folder
dataset.export(
    export_dir="coco_1000/images",
    dataset_type=fo.types.ImageDirectory,
)

print("Download complete! Images saved to ./coco_1000/images")
