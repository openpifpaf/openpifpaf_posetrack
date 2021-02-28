import openpifpaf

from .. import headmeta


class Tcaf(openpifpaf.visualizer.Base):
    def __init__(self, meta: headmeta.Tcaf):
        super().__init__(meta.name)
        self.meta = meta
        self.caf_visualizer = openpifpaf.visualizer.Caf(meta)

    def targets(self, field, *, annotation_dicts):
        self.caf_visualizer.targets(field, annotation_dicts=annotation_dicts[0])
        self.caf_visualizer.targets(field, annotation_dicts=annotation_dicts[1])

    def predicted(self, field):
        self.caf_visualizer.predicted(field)
