import os
import tempfile

import nnabla.experimental.viewers as V
from IPython.display import Image


def create_graph(node, width=None, format="png"):
    graph = V.SimpleGraph(verbose=False)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "tmp")
        graph.save(node, path, format=format)
        return Image(".".join([path, format]), width=width)
