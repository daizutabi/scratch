# # Graph Converter for Inference
# # (https://nnabla.readthedocs.io/en/latest/python/tutorial/
# # graph_converter_for_inference.html)

import nnabla as nn
import nnabla.experimental.graph_converters as GC
import nnabla.experimental.viewers as V
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

from ivory.utils.nnabla.graph import create_graph


# -
# !LeNet
def LeNet(image, test=False):
    h = PF.convolution(image, 16, (5, 5), (1, 1), with_bias=False, name="conv1")
    h = PF.batch_normalization(h, batch_stat=not test, name="conv1-bn")
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=True, name="conv2")
    h = PF.batch_normalization(h, batch_stat=not test, name="conv2-bn")
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.affine(h, 10, with_bias=False, name="fc1")
    h = PF.batch_normalization(h, batch_stat=not test, name="fc1-bn")
    h = F.relu(h)

    pred = PF.affine(h, 10, with_bias=True, name="fc2")
    return pred


x = nn.Variable.from_numpy_array(np.random.rand(4, 3, 28, 28))
y = LeNet(x, test=True)
create_graph(y, width=300)

# ## BatchNormalizationLinearConverter
converter = GC.BatchNormalizationLinearConverter(name="bn-linear-lenet")
z = converter.convert(y, [x])
create_graph(z, width=300)
# ## BatchNormalizationFoldedConverter
converter = GC.BatchNormalizationFoldedConverter(name="bn-folded-lenet")
z = converter.convert(y, [x])
create_graph(z, width=300)
# ## FixedPointWeightConverter
converter = GC.FixedPointWeightConverter(name="fixed-point-weight-lenet")
z = converter.convert(y, [x])
create_graph(z, width=300)
# ## FixedPointActivationConverter
converter = GC.FixedPointActivationConverter(name="fixed-point-activation-lenet")
z = converter.convert(y, [x])
create_graph(z, width=300)
# -
converter_w = GC.FixedPointWeightConverter(name="fixed-point-lenet")
converter_a = GC.FixedPointActivationConverter(name="fixed-point-lenet")
converter = GC.SequentialConverter([converter_w, converter_a])
z = converter.convert(y, [x])
create_graph(z, width=300)
