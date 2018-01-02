import mxnet as mx
from mxnet.test_utils import *
import numpy as np
from SigmoidCrossEntropy import *

def test_sigmoid_crossentropy():
    batch_size = 10
    num_classes = 20
    x = np.random.random((batch_size, num_classes))
    y = np.random.randint(0, 2, size = x.shape)

    data = mx.symbol.Variable("data")
    label = mx.symbol.Variable("label")
    sym = mx.sym.Custom(data = data, label = label, op_type = "SigmoidCrossEntropy")

    f = 1.0 / (1 + np.exp(-x))
    loss = np.sum(- (y * np.log(f) + (1 - y) * np.log(1 - f) )) / batch_size
    loss = np.array([loss])
    check_symbolic_forward(sym, [x, y], [loss])

    dx = (f - y) / batch_size

    check_symbolic_backward(sym, [x, y], [np.ones(x.shape)], [dx])


if __name__ == "__main__":
    test_sigmoid_crossentropy()
