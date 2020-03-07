import sys

from ivory.common.context import np

np.context = "gpu"

from ivory.core.optimizer import Adam  # isort:skip
from ivory.utils.repository import repo_directory  # isort:skip

sys.path.append(repo_directory("scratch2"))
from dataset import sequence  # isort:skip

(x_train, t_train), (x_test, t_test) = sequence.load_data("addition.txt")
x_train = np.asarray(x_train)
t_train = np.asarray(t_train)
x_train = x_train[:, ::-1]
x_test = x_test[:, ::-1]

char_to_id, id_to_char = sequence.get_vocab()

# モデルの形状を設定します。
vocab_size = len(char_to_id)
wordvec_size = 16
hideen_size = 128

# モデルを作成します。
from ivory.core.model import branch, Model  # isort:skip

net_encoder = [
    ("input", vocab_size),
    ("embedding", wordvec_size),
    ("lstm", hideen_size, "select"),
]

net_decoder = [
    ("input", vocab_size),
    ("embedding", wordvec_size),
    ("lstm", hideen_size),
    ("affine", vocab_size, "softmax_cross_entropy"),
]

encoder = branch(net_encoder)
decoder = branch(net_decoder)

# エンコーダの出力をデコーダのLSTMレイヤに入力します。
v = encoder[-1].y.set_variable()
decoder[1].h.set_variable(v)  # type:ignore
print(v)
# デコーダの損失パラメータを起点にしてモデルを構築します。
model = Model([decoder[-1].loss])  # type:ignore
for k, layer in enumerate(model.layers):
    print(f"layers[{k}]", layer)
for k, v in enumerate(model.data_input_variables):
    print(f"inputs[{k}]", v)

# 重みの初期値を設定します。
from ivory.common.context import np  # isort:skip

model.init(std="xavier")
for p in [encoder[0].W, decoder[0].W]:  # type:ignore
    p.variable.data = p.init(std=0.01)
for p in model.weights:
    if p.name != "b":
        std1 = f"{float(p.d.std()):.03f}"
        std2 = f"{float(np.sqrt(1/p.d.shape[0])):.03f}"
        print(p.layer.name, p.name, std1, std2, type(p.d), p.d.dtype)

# エンコーダの`stateful`を`False`に設定します。
encoder[1].stateful.d = False  # type:ignore


# 答えを生成する関数を定義します。
def generate(question, char_id, length):
    vs = model.data_input_variables
    vs[0].data = question.reshape(1, -1)
    for layer in encoder:
        for output in layer.outputs:
            output.variable.data = None
        layer.forward()
    chars = []
    for i in range(length):
        vs[1].data = np.array(char_id).reshape(1, 1)
        for layer in decoder[:-1]:
            for output in layer.outputs:
                output.variable.data = None
            layer.forward()
        char_id = np.argmax(layer.y.d.flatten())
        chars.append(int(char_id))
    return chars


# 正解率を計算する関数を定義します。
def evaluate(question, answer):
    generated = generate(question, answer[0], answer.shape[0] - 1)
    return 1 if list(answer[1:]) == generated else 0


max_grad = 5.0

optimizer = Adam()
optimizer.set_model(model)

batch_size = 128
data_size = len(x_train)
max_iters = data_size // batch_size

total_loss = 0.0
count = 0

for epoch in range(25):
    idx = np.random.permutation(np.arange(data_size))
    x_train = x_train[idx]
    t_train = t_train[idx]

    for iters in range(max_iters):
        batch_x = x_train[iters * batch_size : (iters + 1) * batch_size]
        batch_t = t_train[iters * batch_size : (iters + 1) * batch_size]

        model.set_data(batch_x, batch_t[:, :-1], batch_t[:, 1:])
        model.forward()
        total_loss += model.loss
        count += 1
        model.backward()
        model.clip_grads(max_grad)
        optimizer.update()

        if iters % 20 == 0:
            avg_loss = float(total_loss / count)
            print(epoch + 1, iters + 1, f"{avg_loss:.02f}")
            total_loss, count = 0, 0

        model.reset_state()
