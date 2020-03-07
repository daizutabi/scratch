# # 7.3 seq2seqの実装

# 足し算データセットを読み出します。
from ivory.utils.repository import import_module  # isort:skip

sequence = import_module("scratch2/dataset/sequence")
x_train, t_train = sequence.load_data("addition.txt")[0]
char_to_id, id_to_char = sequence.get_vocab()

# モデルの形状を設定します。
vocab_size = len(char_to_id)
wordvec_size = 16
hideen_size = 128


# 「ゼロから作るDeep Learning ❷」のモデルを読み込みます。
sequence = import_module("scratch2/ch07/seq2seq")
seq2seq = sequence.Seq2seq(vocab_size, wordvec_size, hideen_size)
for p in seq2seq.params:
    print(p.shape)

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
# 重みの初期値を「ゼロから作るDeep Learning」と同じにします。
for p, data in zip(model.weights, seq2seq.params):
    p.variable.data = data.copy()
    print(p.layer.name, p.name, p.d.shape, p.d.dtype)

# エンコーダの`stateful`を`False`に設定します。
encoder[1].stateful.d = False  # type:ignore

# データを用意します。
batch_size = 2
x, t = x_train[:batch_size, ::-1], t_train[:batch_size]

# モデルに代入し、「ゼロから作るDeep Learning」の結果と比較します。

# 順伝搬
import numpy as np  # isort:skip

model.reset_state()
model.set_data(x, t[:, :-1], t[:, 1:])
for layer in encoder:
    layer.clear_data()
    layer.forward()  # type:ignore

xs = seq2seq.encoder.embed.forward(x)
hs = seq2seq.encoder.lstm.forward(xs)
h = seq2seq.encoder.forward(x)
print(np.allclose(xs, encoder[0].y.d))
print(np.allclose(hs, encoder[1].y.d))
print(np.allclose(h, encoder[2].y.d))

for layer in decoder:
    layer.clear_data()
    layer.forward()  # type:ignore

seq2seq.decoder.lstm.set_state(h)
out = seq2seq.decoder.embed.forward(t[:, :-1])
out2 = seq2seq.decoder.lstm.forward(out)
score = seq2seq.decoder.affine.forward(out2)
loss = seq2seq.softmax.forward(score, t[:, 1:])
print(np.allclose(out, decoder[0].y.d))
print(np.allclose(out2, decoder[1].y.d))
print(np.allclose(score, decoder[2].y.d))
print(np.allclose(loss, model.loss))

# 逆伝搬
for layer in decoder[::-1]:
    layer.clear_grad()
    layer.backward()  # type:ignore

dscore = seq2seq.softmax.backward()
dout2 = seq2seq.decoder.affine.backward(dscore)
dout = seq2seq.decoder.lstm.backward(dout2)
seq2seq.decoder.embed.backward(dout)
dh = seq2seq.decoder.lstm.dh
print(np.allclose(dout, decoder[1].x.g))
print(np.allclose(dout2, decoder[2].x.g))
print(np.allclose(dscore, decoder[3].x.g))
print(np.allclose(dh, decoder[1].h.g))  # type:ignore
print(np.allclose(dh, encoder[2].y.g))

for layer in encoder[::-1]:
    layer.clear_grad()
    layer.backward()  # type:ignore

dhs = np.zeros_like(seq2seq.encoder.hs)
dhs[:, -1, :] = dh
dout = seq2seq.encoder.lstm.backward(dhs)
seq2seq.encoder.embed.backward(dout)
print(np.allclose(dout, encoder[1].x.g))
print(np.allclose(dhs, encoder[2].x.g))

# 勾配を比較します。
for p, grad in zip(model.weights, seq2seq.grads):
    print(p.layer.name, p.name, np.allclose(p.variable.grad, grad))

# 重みの更新
optimizer = import_module("scratch2/common/optimizer")
util = import_module("scratch2/common/util")
max_grad = 5.0
util.clip_grads(seq2seq.grads, max_grad)
adam_scratch = optimizer.Adam()
adam_scratch.update(seq2seq.params, seq2seq.grads)


from ivory.core.optimizer import Adam  # isort:skip

model.clip_grads(max_grad)
adam = Adam()
adam.set_model(model)
adam.update()


# 更新された重みを比較します。
for p, data in zip(model.weights, seq2seq.params):
    print(p.layer.name, p.name, np.allclose(p.variable.data, data))


# モデル経由で訓練を実施します。
seq2seq = sequence.Seq2seq(vocab_size, wordvec_size, hideen_size)
for p, data in zip(model.weights, seq2seq.params):
    p.variable.data = data.copy()

data_size = len(x_train)
batch_size = 128

for iters in range(5):
    model.reset_state()
    batch_x = x_train[iters * batch_size : (iters + 1) * batch_size]
    batch_t = t_train[iters * batch_size : (iters + 1) * batch_size]

    seq2seq.forward(batch_x, batch_t)
    seq2seq.backward()
    util.clip_grads(seq2seq.grads, max_grad)
    adam_scratch.update(seq2seq.params, seq2seq.grads)

    model.set_data(batch_x, batch_t[:, :-1], batch_t[:, 1:])
    model.forward()
    model.backward()
    model.clip_grads(max_grad)
    adam.update()

    print('grad ', end='')
    for p, grad in zip(model.weights, seq2seq.grads):
        print(np.allclose(p.variable.grad, grad), end=', ')
    print('\ndata ', end='')
    for p, data in zip(model.weights, seq2seq.params):
        print(np.allclose(p.variable.data, data), end=', ')
    print()
