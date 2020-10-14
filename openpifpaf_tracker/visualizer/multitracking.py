import logging

import openpifpaf

LOG = logging.getLogger(__name__)


class MultiTracking(openpifpaf.visualizer.Base):
    show = False

    instance_threshold = 0.2
    trail_length = 10

    def __init__(self, meta: openpifpaf.headmeta.Caf):
        super().__init__(meta.name)
        self.meta = meta
        self.annotation_painter = openpifpaf.show.AnnotationPainter()

        self.anns = []

    def predicted(self, anns):
        if not self.show:
            return

        self.anns.append([ann for ann in anns if ann.score() > self.instance_threshold])
        if len(self.anns) > self.trail_length:
            self.anns.pop(0)

        with self.image_canvas(self._processed_image) as ax:
            for frame_anns in self.anns:
                self.annotation_painter.annotations(ax, frame_anns)
