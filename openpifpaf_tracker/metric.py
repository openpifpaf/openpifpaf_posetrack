from collections import defaultdict
import datetime
import json
import logging
import os
import subprocess

import numpy as np
import openpifpaf

LOG = logging.getLogger(__name__)


class Posetrack(openpifpaf.metric.Base):
    def __init__(self, *, images, categories,
                 ground_truth=None, output_format='2018'):
        super().__init__()

        self.images_by_file = images
        self.categories_by_file = categories
        self.ground_truth_directory = ground_truth
        self.output_format = output_format

        self.predictions_by_file = defaultdict(list)
        self.output_dir_suffix = '{}'.format(
            datetime.datetime.now().strftime('%y%m%d-%H%M%S'),
        )

        self._written_mot_stats_file = None
        self._written_ap_stats_file = None

    def stats(self):
        if self._written_ap_stats_file is None \
           or self._written_mot_stats_file is None:
            return {
                'stats': [],
                'text_labels': [],
            }

        with open(self._written_mot_stats_file, 'r') as f_mot:
            mot_stats = json.load(f_mot)
        with open(self._written_ap_stats_file, 'r') as f_ap:
            ap_stats = json.load(f_ap)

        mot_index_by_name = {n: int(i) for i, n in mot_stats['names'].items()}
        ap_index_by_name = {n: int(i) for i, n in ap_stats['names'].items()}

        return {
            'stats': [
                mot_stats['mota'][mot_index_by_name['total']],
                0.5 * (ap_stats['ap'][ap_index_by_name['right_wrist']]
                       + ap_stats['ap'][ap_index_by_name['left_wrist']]),
                0.5 * (ap_stats['ap'][ap_index_by_name['right_ankle']]
                       + ap_stats['ap'][ap_index_by_name['left_ankle']]),
                ap_stats['ap'][ap_index_by_name['total']],
            ],
            'text_labels': [
                'MOTA', 'AP_wrists', 'AP_ankles', 'AP',
            ],
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

        # use poseval to evaluate right away
        if self.ground_truth_directory is not None:
            gt_dir = os.path.dirname(self.ground_truth_directory)
            if not gt_dir.endswith('/'):
                gt_dir = gt_dir + '/'

            pred_dir = output_dir
            if not pred_dir.endswith('/'):
                pred_dir = pred_dir + '/'

            out_dir = output_dir
            if out_dir.endswith('/'):
                out_dir = out_dir[:-1]
            out_dir = out_dir + '-poseval/'

            cmd = [
                'python', '-m', 'poseval.evaluate',
                '--groundTruth', gt_dir,
                '--predictions', pred_dir,
                '--outputDir', out_dir,
                '--evalPoseTracking',
                '--evalPoseEstimation',
                '--saveEvalPerSequence',
            ]
            LOG.info('eval command: %s', ' '.join(cmd))
            subprocess.run(cmd, check=True)

            self._written_mot_stats_file = os.path.join(out_dir, 'total_MOT_metrics.json')
            self._written_ap_stats_file = os.path.join(out_dir, 'total_AP_metrics.json')
