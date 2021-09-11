import typing as t


class Compose(object):
    def __init__(self, transforms: t.List):
        assert transforms
        self.transforms = transforms.copy()

    def __call__(self, data):
        for tr in self.transforms:
            data = tr(data)
        return data
