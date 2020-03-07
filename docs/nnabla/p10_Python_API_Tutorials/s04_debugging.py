# # Debugging (https://nnabla.readthedocs.io/en/latest/python/tutorial/debugging.html)

import nnabla as nn
import nnabla.experimental.viewers as V
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from IPython.display import Image
from nnabla.ext_utils import get_extension_context
from nnabla.utils.profiler import GraphProfiler, GraphProfilerCsvWriter

from ivory.utils.path import cache_file


def block(x, maps, test=False, name="block"):
    h = x
    with nn.parameter_scope(name):
        with nn.parameter_scope("in-block-1"):
            h = PF.convolution(h, maps, kernel=(3, 3), pad=(1, 1), with_bias=False)
            h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)
        with nn.parameter_scope("in-block-2"):
            h = PF.convolution(h, maps // 2, kernel=(3, 3), pad=(1, 1), with_bias=False)
            h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)
        with nn.parameter_scope("in-block-3"):
            h = PF.convolution(h, maps, kernel=(3, 3), pad=(1, 1), with_bias=False)
            h = PF.batch_normalization(h, batch_stat=not test)

        if h.shape[1] != x.shape[1]:
            with nn.parameter_scope("skip"):
                s = PF.convolution(x, maps, kernel=(3, 3), pad=(1, 1), with_bias=False)
                s = PF.batch_normalization(s, batch_stat=not test)

    return F.relu(h + s)


def network(x, maps=16, test=False):
    h = x
    h = PF.convolution(
        h, maps, kernel=(3, 3), pad=(1, 1), name="first-conv", with_bias=False
    )
    h = PF.batch_normalization(h, batch_stat=not test, name="first-bn")
    h = F.relu(h)
    for l in range(4):
        h = block(h, maps * 2 ** (l + 1), name="block-{}".format(l))
        h = F.max_pooling(h, (2, 2))
    h = F.average_pooling(h, h.shape[2:])
    pred = PF.affine(h, 100, name="pred")
    return pred


# ## Visit Method
class PrintFunc:
    def __call__(self, nnabla_func):
        print("==========")
        print(nnabla_func.info.type_name)
        print(nnabla_func.inputs)
        print(nnabla_func.outputs)
        print(nnabla_func.info.args)


# -
nn.clear_parameters()  # this call is just in case to do the following code again
x = nn.Variable([4, 3, 128, 128])
pred = network(x)
pred.visit(PrintFunc())

# ## Simple Graph Viewer
# !Create graph again just in case
nn.clear_parameters()  # call this in case you want to run the following code agian
x = nn.Variable([4, 3, 128, 128])
pred = network(x)
# -
graph = V.SimpleGraph(verbose=False)
path = cache_file("nnabla/tutorial/debugging/graph")
graph.save(pred, path, cleanup=True, format="png")
Image(path + ".png", width=500)

# ## Profiling utils
# !Create graph again just in case
nn.clear_parameters()  # call this in case you want to run the following code agian

# !Context
device = "cudnn"
ctx = get_extension_context(device)
nn.set_default_context(ctx)

# !Network
x = nn.Variable([4, 3, 128, 128])
t = nn.Variable([4, 1])
pred = network(x)
loss = F.mean(F.softmax_cross_entropy(pred, t))

# !Solver
solver = S.Momentum()
solver.set_parameters(nn.get_parameters())

# !Profiler
B = GraphProfiler(loss, solver=solver, device_id=0, ext_name=device, n_run=100)
B.run()
print("Profile finished.")

# !Report
path = cache_file("nnabla/tutorial/debugging/profile.csv")
with open(path, "w") as f:
    writer = GraphProfilerCsvWriter(B, file=f)
    writer.write()
