import os
import logging

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)


def _get_bdd100k_metadata():
    BDD100k_STUFF_CLASSES = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]
    BDD100k_STUFF_COLORS = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
    # fmt: on
    return {
        "stuff_classes": BDD100k_STUFF_CLASSES,
        "stuff_colors": BDD100k_STUFF_COLORS,
    }


def _get_bdd100k_files(image_dir, label_dir):
    files = []
    # scan through the directory
    videos = PathManager.ls(image_dir)
    logger.info(f"{len(videos)} videos found in '{image_dir}'.")
    for video in videos:
        video_img_dir = os.path.join(image_dir, video)
        video_label_dir = os.path.join(label_dir, video)
        for basename in PathManager.ls(video_img_dir):
            image_file = os.path.join(video_img_dir, basename)

            suffix = ".jpg"
            assert basename.endswith(suffix), basename
            basename = basename[: -len(suffix)]

            label_file = os.path.join(video_label_dir, basename + ".png")

            files.append((image_file, label_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_bdd100k_semantic(image_dir, gt_dir):
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, label_file in _get_bdd100k_files(image_dir, gt_dir):
        ret.append(
            {
                "file_name": image_file,
                "image_id": os.path.basename(image_file),
                "sem_seg_file_name": label_file,
                "height": 720,
                "width": 1280,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    return ret


def register_bdd100k(root):
    SPLITS = {
        "bdd100k_sem_seg_train": (
            "bdd100k/seg/images/train/",
            "bdd100k/seg/labels/train/",
        ),
        "bdd100k_sem_seg_val": ("bdd100k/seg/images/val/", "bdd100k/seg/labels/val/"),
    }
    for key, (image_dir, label_dir) in SPLITS.items():
        meta = _get_bdd100k_metadata()
        image_dir = os.path.join(root, image_dir)
        label_dir = os.path.join(root, label_dir)

        DatasetCatalog.register(
            key, lambda x=image_dir, y=label_dir: load_bdd100k_semantic(x, y)
        )
        MetadataCatalog.get(key).set(
            image_dir=image_dir,
            label_dir=label_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


if __name__.endswith(".bdd100k"):
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_bdd100k(_root)
