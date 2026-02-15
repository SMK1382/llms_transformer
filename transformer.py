import numpy as np
from word_embedding import embedding
from vocabulary import vocabulary
import random


#####################################################################################
def positional_encoding(X):
    batch, seq_len, d_model = X.shape
    for i in range(batch):
        for pos in range(seq_len):
            for j in range(0, d_model, 2):
                X[i, pos, j] += np.sin(pos / (10000 ** (2 * j / d_model)))
            for j in range(1, d_model, 2):
                X[i, pos, j] += np.cos(pos / (10000 ** (2 * (j-1) / d_model)))
    return X


#####################################################################################
class softmax:
    def __init__(self):
        self.probs = None

    def softmax_forward(self, x, axis=-1):
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        x_shifted = np.clip(x_shifted, -500, 500)
        exp_x = np.exp(x_shifted)
        self.probs = exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-10)
        return self.probs

    def softmax_backward(self, grad_x):
        temp = np.sum(self.probs * grad_x, axis=-1, keepdims=True)
        dS = self.probs * (grad_x - temp)
        return dS


#####################################################################################
class add_and_norm:
    def __init__(self, d_model, eps=1e-5, learning_rate=0.01):
        self.X = None
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
        self.alpha = learning_rate

        self.X_hat = None
        self.mean = None
        self.var = None
        self.Z = None

    def get_input(self, X):
        self.X = X
        self.batch_size, self.seq_len, self.d_model = X.shape

    def forward(self, sublayer_out):
        Z = self.X + sublayer_out

        mean = np.mean(Z, axis=-1, keepdims=True)
        var = np.var(Z, axis=-1, keepdims=True)

        X_hat = (Z - mean) / np.sqrt(var + self.eps)
        out = self.gamma * X_hat + self.beta

        self.Z = Z
        self.mean = mean
        self.var = var
        self.X_hat = X_hat

        return out

    def backward(self, d_out):
        d_gamma = np.sum(d_out * self.X_hat, axis=(0, 1))
        d_beta = np.sum(d_out, axis=(0, 1))

        dX_hat = d_out * self.gamma
        std_inv = 1.0 / np.sqrt(self.var + self.eps)

        d_var = np.sum(dX_hat * (self.Z - self.mean) * -0.5 * std_inv**3, axis=-1, keepdims=True)
        d_mean = np.sum(dX_hat * -std_inv, axis=-1, keepdims=True) + d_var * np.mean(-2.0 * (self.Z - self.mean), axis=-1, keepdims=True)
        d_Z = dX_hat * std_inv + d_var * 2.0 * (self.Z - self.mean) / self.d_model + d_mean / self.d_model

        d_X = d_Z
        d_Sublayer = d_Z

        self.gamma -= self.alpha * d_gamma
        self.beta -= self.alpha * d_beta

        return d_X, d_Sublayer


#####################################################################################
class multi_head_attention:
    def __init__(self, d_model, n_head, learning_rate=0.01, decoder=False, mask=False):
        scale = 0.01
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        self.alph = learning_rate
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.mask = mask
        self.decoder = decoder
        self.softmax = softmax()

        self.X = None
        self.Q = None
        self.K = None
        self.V = None
        self.attn = None
        self.out_heads = None
        self.out_cat = None

    def get_input(self, X):
        self.X = X
        if self.decoder and isinstance(X, (list, tuple)):
            self.batch_size, self.seq_len, self.d_model = X[0].shape
        else:
            self.batch_size, self.seq_len, self.d_model = X.shape

    def attention_forward(self, padding_mask):
        if self.decoder and isinstance(self.X, (list, tuple)):
            Q = self.X[0] @ self.W_Q
            K = self.X[0] @ self.W_K
            V = self.X[1] @ self.W_V
        else:
            Q = self.X @ self.W_Q
            K = self.X @ self.W_K
            V = self.X @ self.W_V

        Q = Q.reshape(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)

        if padding_mask is not None:
            mask_expanded = padding_mask[:, np.newaxis, np.newaxis, :]
            scores = np.where(mask_expanded == 1, scores, -1e9)

        if self.mask:
            causal_mask = np.tril(np.ones((self.seq_len, self.seq_len)))
            scores = np.where(causal_mask == 1, scores, -1e9)

        attn = self.softmax.softmax_forward(scores, axis=-1)
        out_heads = attn @ V
        out_cat = out_heads.transpose(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.d_model)
        out = out_cat @ self.W_O

        self.Q = Q
        self.K = K
        self.V = V
        self.attn = attn
        self.out_heads = out_heads
        self.out_cat = out_cat

        return out

    def attention_backward(self, d_out):
        B, N, D = self.batch_size, self.seq_len, self.d_model
        H = self.n_head
        Dh = self.d_head
        scale = np.sqrt(self.d_head)

        d_out_cat = d_out @ self.W_O.T
        dW_O = (self.out_cat.reshape(B * N, D).T @ d_out.reshape(B * N, D))

        d_out_heads = d_out_cat.reshape(B, N, H, Dh).transpose(0, 2, 1, 3)

        d_V = self.attn.transpose(0, 1, 3, 2) @ d_out_heads
        d_attn = d_out_heads @ self.V.transpose(0, 1, 3, 2)

        d_scores = self.softmax.softmax_backward(d_attn)

        d_Q = (d_scores @ self.K) / scale
        d_K = (d_scores.transpose(0, 1, 3, 2) @ self.Q) / scale

        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(B * N, D)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(B * N, D)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(B * N, D)

        X_flat = self.X.reshape(B * N, D) if not isinstance(self.X, (list, tuple)) else self.X[0].reshape(B * N, D)

        dW_Q = X_flat.T @ d_Q
        dW_K = X_flat.T @ d_K
        dW_V = X_flat.T @ d_V

        self.W_Q -= self.alph * dW_Q
        self.W_K -= self.alph * dW_K
        self.W_V -= self.alph * dW_V
        self.W_O -= self.alph * dW_O

        d_X = (d_Q @ self.W_Q.T + d_K @ self.W_K.T + d_V @ self.W_V.T).reshape(B, N, D)

        if self.decoder:
            return (d_Q @ self.W_Q.T + d_K @ self.W_K.T).reshape(B, N, D), (d_V @ self.W_V.T).reshape(B, N, D)

        return d_X


#####################################################################################
class feed_forward:
    def __init__(self, d_model, d_ff=8, learning_rate=0.01):
        scale = 0.01
        self.W_1 = np.random.randn(d_model, d_ff) * scale
        self.b_1 = np.zeros(d_ff)
        self.W_2 = np.random.randn(d_ff, d_model) * scale
        self.b_2 = np.zeros(d_model)

        self.learning_rate = learning_rate
        self.X = None

        self.Z1 = None
        self.A1 = None

    def get_input(self, X):
        self.X = X
        self.batch_size, self.seq_len, self.d_model = X.shape

    def forward(self):
        X_flat = self.X.reshape(-1, self.d_model)

        self.Z1 = X_flat @ self.W_1 + self.b_1
        self.A1 = np.maximum(0, self.Z1)

        out = self.A1 @ self.W_2 + self.b_2
        out = out.reshape(self.batch_size, self.seq_len, self.d_model)

        return out

    def backward(self, dOut):
        B, N, D = self.batch_size, self.seq_len, self.d_model

        dOut_flat = dOut.reshape(-1, D)

        dW_2 = self.A1.T @ dOut_flat
        db_2 = np.sum(dOut_flat, axis=0)

        dA1 = dOut_flat @ self.W_2.T
        dZ1 = dA1 * (self.Z1 > 0)

        X_flat = self.X.reshape(-1, D)
        dW_1 = X_flat.T @ dZ1
        db_1 = np.sum(dZ1, axis=0)

        dX = dZ1 @ self.W_1.T
        dX = dX.reshape(B, N, D)

        self.W_1 -= self.learning_rate * dW_1
        self.b_1 -= self.learning_rate * db_1
        self.W_2 -= self.learning_rate * dW_2
        self.b_2 -= self.learning_rate * db_2

        return dX


#####################################################################################
class Linear:
    def __init__(self, d_model, d_out, learning_rate=0.01):
        scale = 0.01
        self.W = np.random.randn(d_model, d_out) * scale
        self.b = np.zeros(d_out)
        self.lr = learning_rate

        self.X = None

    def forward(self, X):
        self.X = X
        self.batch_size, self.seq_len, self.d_model = X.shape

        X_flat = X.reshape(-1, self.d_model)
        out = X_flat @ self.W + self.b
        out = out.reshape(self.batch_size, self.seq_len, -1)

        return out

    def backward(self, d_out):
        B, N, D_out = d_out.shape
        D_in = self.W.shape[0]

        d_out_flat = d_out.reshape(-1, D_out)
        X_flat = self.X.reshape(-1, D_in)

        dW = X_flat.T @ d_out_flat
        db = np.sum(d_out_flat, axis=0)
        dX = d_out_flat @ self.W.T

        self.W -= self.lr * dW
        self.b -= self.lr * db

        return dX.reshape(B, N, D_in)


#####################################################################################
def padding(x):
    temp = []
    max_len = max(len(s) for s in x)
    for i in x:
        temp.append(i + ['<pad>'] * (max_len - len(i)))
    return np.array(temp), np.where(np.array(temp) == '<pad>', 0, 1)


def cross_entropy_with_grad(vocabulary, target_model, output_model):
    B, T, V = output_model.shape

    target_index = []
    for i in target_model:
        for j in i:
            target_index.append(np.where(vocabulary == j)[0][0])
    target_index = np.array(target_index)

    output_flat = output_model.reshape(B * T, V)
    output_flat = np.clip(output_flat, 1e-10, 1.0)

    loss = -np.sum(np.log(output_flat[np.arange(B * T), target_index] + 1e-10))

    grad = output_flat.copy()
    grad[np.arange(B * T), target_index] -= 1
    grad = grad / (B * T)
    grad = grad.reshape(B, T, V)

    return loss, grad


def input_embedding(input_model, embedding):
    result = np.zeros((input_model.shape[0], input_model.shape[1], len(list(embedding.values())[0])))
    for b in range(input_model.shape[0]):
        for s in range(input_model.shape[1]):
            word = input_model[b, s]
            for key, value in embedding.items():
                if word == key:
                    result[b, s] = value
                    break
    return result


#####################################################################################
def forward_train(X, target_model, vocabulary, padding_matrix,
                  multi_head_attention_1, multi_head_attention_2, masked_multi_head_attention,
                  feed_forward_1, feed_forward_2, norm_1, norm_2, norm_3, norm_4, norm_5,
                  linear, softmax_layer):

    # Encoder
    X_1 = X_2 = positional_encoding(X.copy())
    multi_head_attention_1.get_input(X_2)
    X_2 = multi_head_attention_1.attention_forward(padding_matrix)
    norm_1.get_input(X_2)
    X_1 = X_2 = norm_1.forward(X_1)

    feed_forward_1.get_input(X_2)
    X_2 = feed_forward_1.forward()
    norm_2.get_input(X_2)
    output_encoder = norm_2.forward(X_1)

    # Decoder
    X_1 = X_2 = positional_encoding(X.copy())
    masked_multi_head_attention.get_input(X_2)
    X_2 = masked_multi_head_attention.attention_forward(padding_matrix)
    norm_3.get_input(X_2)
    X_1 = X_2 = norm_3.forward(X_1)

    multi_head_attention_2.get_input([output_encoder, X_2])
    X_2 = multi_head_attention_2.attention_forward(padding_matrix)
    norm_4.get_input(X_2)
    X_1 = X_2 = norm_4.forward(X_1)

    feed_forward_2.get_input(X_2)
    X_2 = feed_forward_2.forward()
    norm_5.get_input(X_2)
    X_2 = norm_5.forward(X_1)

    X_1 = linear.forward(X_2)
    X_1 = softmax_layer.softmax_forward(X_1)

    return cross_entropy_with_grad(vocabulary, target_model, X_1)


def backward_train(grad, 
                   multi_head_attention_1, multi_head_attention_2, masked_multi_head_attention,
                   feed_forward_1, feed_forward_2, norm_1, norm_2, norm_3, norm_4, norm_5,
                   linear, softmax_layer):

    X = softmax_layer.softmax_backward(grad)
    X = linear.backward(X)

    X, sub_layer = norm_5.backward(X)
    X = feed_forward_2.backward(X)
    X, sub_layer = norm_4.backward(X + sub_layer)
#--------------------------------------------------------------------------------------
    #X, sub_layer_2 = multi_head_attention_2.attention_backward(X + sub_layer)
    #X, sub_layer = norm_3.backward(X + sub_layer_2)

    #X = masked_multi_head_attention.attention_backward(X + sub_layer)
    #X, sub_layer = norm_2.backward(X)

    #X = feed_forward_1.backward(X)
    #X, sub_layer = norm_1.backward(X + sub_layer)

    #X = multi_head_attention_1.attention_backward(X + sub_layer)
#--------------------------------------------------------------------------------------
    X_1, sub_layer_2 = multi_head_attention_2.attention_backward(X)

    X, sub_layer = norm_3.backward(sub_layer + sub_layer_2)
    X = masked_multi_head_attention.attention_backward(X)
    
    X, sub_layer = norm_2.backward(X_1)
    X = feed_forward_1.backward(X)
    X, sub_layer = norm_1.backward(X + sub_layer)
    X = multi_head_attention_1.attention_backward(X)


#####################################################################################
class transformer:
    def __init__(self, d_model, n_head, vocabulary, learning_rate=0.01):
        self.d_model = d_model
        self.n_head = n_head
        self.learning_rate = learning_rate

        self.multi_head_attention_1 = multi_head_attention(d_model, n_head, learning_rate)
        self.multi_head_attention_2 = multi_head_attention(d_model, n_head, learning_rate, decoder=True)
        self.masked_multi_head_attention = multi_head_attention(d_model, n_head, learning_rate, mask=True)

        self.feed_forward_1 = feed_forward(d_model, learning_rate=learning_rate)
        self.feed_forward_2 = feed_forward(d_model, learning_rate=learning_rate)

        self.norm_1 = add_and_norm(d_model, learning_rate=learning_rate)
        self.norm_2 = add_and_norm(d_model, learning_rate=learning_rate)
        self.norm_3 = add_and_norm(d_model, learning_rate=learning_rate)
        self.norm_4 = add_and_norm(d_model, learning_rate=learning_rate)
        self.norm_5 = add_and_norm(d_model, learning_rate=learning_rate)

        self.linear = Linear(d_model, len(vocabulary), learning_rate)
        self.softmax = softmax()

    def train(self, X, target_model, vocabulary, padding_matrix, epochs):
        for epoch in range(epochs):
            loss, grad = forward_train(
                X, target_model, vocabulary, padding_matrix,
                self.multi_head_attention_1, self.multi_head_attention_2, self.masked_multi_head_attention,
                self.feed_forward_1, self.feed_forward_2,
                self.norm_1, self.norm_2, self.norm_3, self.norm_4, self.norm_5,
                self.linear, self.softmax
            )

            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f}")

            backward_train(
                grad,
                self.multi_head_attention_1, self.multi_head_attention_2, self.masked_multi_head_attention,
                self.feed_forward_1, self.feed_forward_2,
                self.norm_1, self.norm_2, self.norm_3, self.norm_4, self.norm_5,
                self.linear, self.softmax
            )

    def inference(self, X):
        pass


#####################################################################################
if __name__ == "__main__":
    batch_size = 3
    d_model = 512
    n_head = 8

    with open("samples.txt", 'r') as file:
        samples = file.read().lower().split(".\n")
        file.close()

    X = []
    targets = []
    for i in range(len(samples) - 1):
        X.append(["<bos>"] + samples[i].split())
        targets.append(samples[i].split() + ["<eos>"])

    input_model = []
    target_model = []
    for i in range(batch_size):
        temp = random.randint(0, min(99, len(X) - 1))
        input_model.append(X[temp])
        target_model.append(targets[temp])

    input_model, padding_matrix = padding(input_model)
    target_model, _ = padding(target_model)

    input_model = input_embedding(input_model, embedding).reshape(input_model.shape[0], input_model.shape[1], d_model)

    transformer_model = transformer(d_model, n_head, vocabulary, learning_rate=0.1)
    transformer_model.train(input_model, target_model, vocabulary, padding_matrix, 50)
