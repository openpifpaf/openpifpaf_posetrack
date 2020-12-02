import logging

import openpifpaf

LOG = logging.getLogger(__name__)


class MultiTracking(openpifpaf.visualizer.Base):
    show = False
    trail_length = 10

    def __init__(self, meta: openpifpaf.headmeta.Caf):
        super().__init__(meta.name)
        self.meta = meta
        self.annotation_painter = openpifpaf.show.AnnotationPainter()

        self.anns = []

    def predicted(self, anns):
        if not self.show:
            return

        self.anns.append(anns)
        if len(self.anns) > self.trail_length:
            self.anns.pop(0)

        current_ids = {ann.id_ for ann in self.anns[-1]}
        with self.image_canvas(self._image) as ax:
            for frame_i, frame_anns in enumerate(self.anns):
                # only show trails for poses that are in the current frame
                frame_anns = [ann for ann in frame_anns if ann.id_ in current_ids]

                alpha = 0.5**(len(self.anns) - 1 - frame_i)
                if self._image_meta is not None:
                    frame_anns = openpifpaf.transforms.Preprocess.annotations_inverse(
                        frame_anns, self._image_meta)
                self.annotation_painter.annotations(ax, frame_anns, alpha=alpha)
