class Queue(object):
    def __init__(self):
        self.pairs = []

    def put(self, item, value):
        self.pairs.append((item, value))
        self.pairs.sort(key=lambda pair: pair[1])

    def get(self):
        return self.pairs.pop(0)[0]

    def contains(self, item):
        items = [pair[0] for pair in self.pairs]
        return item in items

    def is_empty(self):
        return not self.pairs

    def size(self):
        return len(self.pairs)