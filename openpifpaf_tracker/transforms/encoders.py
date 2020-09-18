import openpifpaf


class Encoders(openpifpaf.transforms.Preprocess):
    def __init__(self, encoders):
        self.encoders = encoders

    def __call__(self, image, anns, meta):
        anns = [enc(image, anns, meta) for enc in self.encoders]
        for image_meta in meta:
            image_meta['head_indices'] = [enc.meta.head_index for enc in self.encoders]
        return image, anns, meta
