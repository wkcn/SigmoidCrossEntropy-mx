import mxnet as mx

class SigmoidCrossEntropy(mx.operator.CustomOp):
    def __init__(self):
        super(SigmoidCrossEntropy, self).__init__()
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        y = in_data[1]
        m = mx.nd.relu(-x)
        loss = mx.nd.mean(mx.nd.sum(x - x * y + m + mx.nd.log(mx.nd.exp(-m) + mx.nd.exp(-x - m)), axis = 1))
        self.assign(out_data[0], req[0], loss)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dx = mx.nd.sigmoid(in_data[0]) - in_data[1]
        self.assign(in_grad[0], req[0], dx)
        
@mx.operator.register("SigmoidCrossEntropy")
class SigmoidCrossEntropyProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SigmoidCrossEntropyProp, self).__init__(need_top_grad = False)
    def list_arguments(self):
        return ["data", "label"]
    def list_outputs(self):
        return ["loss"]
    def infer_shape(self, in_shape):
        return [in_shape[0], in_shape[0]], [(1, )], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype], [dtype], []
    def create_operator(self, ctx, shapes, dtypes):
        return SigmoidCrossEntropy()
