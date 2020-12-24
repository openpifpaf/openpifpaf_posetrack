import logging

import openpifpaf

LOG = logging.getLogger(__name__)


class MultiTracking(openpifpaf.visualizer.Base):
    trail_length = 10

    def __init__(self, meta: openpifpaf.headmeta.Caf):
        super().__init__('multi_' + meta.name)
        LOG.debug('vis %s', 'multi_' + meta.name)
        self.meta = meta
        self.annotation_painter = openpifpaf.show.AnnotationPainter()

        self.anns_trail = []

    def predicted(self, anns):
        if not self.indices:
            return

        self.anns_trail.append(anns)
        if len(self.anns_trail) > self.trail_length:
            self.anns_trail.pop(0)

        current_ids = {ann.id_ for ann in self.anns_trail[-1]}
        with self.image_canvas(self._image) as ax:
            for frame_i, frame_anns in enumerate(self.anns_trail):
                # only show trails for poses that are in the current frame
                frame_anns = [ann for ann in frame_anns if ann.id_ in current_ids]

                # only show trails for poses that have confidence > 0.01
                frame_anns = [ann for ann in frame_anns if ann.score() > 0.01]

                alpha = 0.5**(len(self.anns_trail) - 1 - frame_i)
                if self._image_meta is not None:
                    frame_anns = openpifpaf.transforms.Preprocess.annotations_inverse(
                        frame_anns, self._image_meta)
                self.annotation_painter.annotations(ax, frame_anns, alpha=alpha)
