from collections import defaultdict
import datetime
import json
import logging
import os

import numpy as np
import openpifpaf

LOG = logging.getLogger(__name__)


class PoseTrackMetric(openpifpaf.metric.Metric):
    def __init__(self, *, images, categories, output_format='2018'):
        super().__init__()

        self.images_by_file = images
        self.categories_by_file = categories
        self.output_format = output_format

        self.predictions_by_file = defaultdict(list)
        self.output_dir_suffix = '{}'.format(
            datetime.datetime.now().strftime('%y%m%d-%H%M%S'),
        )

    def stats(self):
        return {
            'stats': [],
            'text_labels': [],
        }

    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        annotation_file = image_meta['annotation_file']

        # make sure an entry for this annotation file is created
        # even when a file does not have any predictions
        if annotation_file not in self.predictions_by_file:
            self.predictions_by_file[annotation_file] = []

        for ann in predictions:
            keypoints = np.copy(ann.data)

            # cleanup
            keypoints[:, 2] = np.clip(keypoints[:, 2], 0.0, 1.0)
            keypoints[keypoints[:, 2] == 0.0, :2] = 0.0

            bbox = [float(v) for v in ann.bbox()]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            bbox = [round(v, 2) for v in bbox]

            self.predictions_by_file[annotation_file].append({
                'bbox': bbox,
                'keypoints': [round(k, 2) for k in keypoints.reshape(-1).tolist()],
                'track_id': ann.id_,
                'image_id': image_meta['image_id'],
                'image_file': image_meta['file_name'],
                'category_id': 1,
                'scores': [round(s, 2) for s in keypoints[:, 2].tolist()],
                'score': max(0.01, round(ann.score(), 2)),
            })

    def _write2018(self, output_dir, annotation_file, *, additional_data=None):
        sequence_name = os.path.basename(annotation_file)
        out_name = '{}/{}'.format(output_dir, sequence_name)
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        LOG.info('writing %s', out_name)

        data = {
            'images': self.images_by_file[annotation_file],
            'annotations': self.predictions_by_file[annotation_file],
            'categories': self.categories_by_file[annotation_file],
        }
        if additional_data:
            data = dict(**data, **additional_data)
        with open(out_name, 'w') as f:
            json.dump(data, f)
        LOG.info('wrote %s', out_name)

    def write_predictions(self, filename, *, additional_data=None):
        output_dir = '{}-{}'.format(filename, self.output_dir_suffix)
        for annotation_file in self.predictions_by_file.keys():
            if self.output_format == '2018':
                self._write2018(output_dir, annotation_file, additional_data=additional_data)
            else:
                raise NotImplementedError
