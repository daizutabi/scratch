import numpy

numpy.scatter_add = numpy.add.at

try:
    import cupy
except ImportError:
    cupy = None
else:
    pass
    # cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)


class Context:
    def __init__(self):
        self.context = "cpu"

    def __getattr__(self, name: str):
        if self.context == "cpu":
            return getattr(numpy, name)
        else:
            return getattr(cupy, name)


np = Context()
