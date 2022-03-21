# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import os
import PIL.Image as Image

from detectron2.utils.file_io import PathManager
from detectron2.evaluation import SemSegEvaluator as BaseEvaluator


class SemSegEvaluator(BaseEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
    ):
        super().__init__(
            dataset_name,
            distributed,
            output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        self._reset_num = 0

    def reset(self):
        super().reset()
        suffix = f"sem_seg_{self._reset_num}"
        self._working_dir = os.path.join(self._output_dir, f"{suffix}")
        PathManager.mkdirs(self._working_dir)
        self._logger.info(f"call reset, and mkdir {self._working_dir}")
        self._reset_num += 1

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            input_gt_file = self.input_file_to_gt_file[input["file_name"]]
            with PathManager.open(input_gt_file, "rb") as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
            file_name = os.path.basename(input["file_name"])
            file_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(self._working_dir, f"{file_name}_sem_seg.png")
            # print(file_path, pred.shape, pred.dtype)
            Image.fromarray(pred.astype(np.uint8)).save(file_path)
