from typewriter.values.values import END_OF_TEXT


class MarkEnd(object):
    def __init__(self, marker=None):
        self.marker = marker or END_OF_TEXT

    def __call__(self, text: str):
        return text + self.marker
