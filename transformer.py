import numpy as np



def positional_encoding(X_input):
    batch, seq_len, d_model = X_input.shape
    for i in range(batch):
        for pos in range(seq_len):
            for j in range(0, d_model, 2):
                X_input[i, pos, j] += np.sin(pos / (10000 ** (2 * j / d_model)))
            for j in range(1, d_model, 2):
                X_input[i, pos, j] += np.cos(pos / (10000 ** (2 * (j-1) / d_model)))
    return X_input




def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def attention_matrix_generate(d_model, i):
    np.seed(i)
    V = np.random.random(d_model, d_model)
    Q = np.random.random(d_model, d_model)
    K = np.random.random(d_model, d_model)
    return (K, Q, V)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(A, dA):
    # A, dA: (self.batch_size, self.n_head, self.seq_len, self.d_head)
    temp = np.sum(A * dA, axis=-1, keepdims=True)
    dS = A * (dA - temp)
    return dS


class multi_head_attention:
    def init(self, X, W_Q, W_K, W_V, W_O, n_head, learning_rate = 0.1 mask=None):
        self.X = X                                  # (self.batch_size, self.seq_len, d_model)
        self.W_Q = W_Q                              # (d_model, d_model)
        self.W_K = W_K
        self.W_V = W_V
        self.W_O = W_O
        self.alph = learning_rate
        self.batch_size, self.seq_len, self.d_model = X.shape
        self.n_head = n_head
        self.d_head = self.d_model // self.n_head
        self.mask = mask

        self.Q = None
        self.K = None
        self.V = None
        self.attn = None
        self.out_heads = None
        self.out_cat = None
    # ==================================================
    # Forward
    # ==================================================
    def attention_forward(self):
        self.batch_size, self.seq_len, D = self.batch_size, self.seq_len, self.d_model
        self.n_head, self.d_head = self.n_head, self.d_head

        Q = self.X @ self.W_Q # (self.batch_size, self.seq_len, D)
        K = self.X @ self.W_K
        V = self.X @ self.W_V

        Q = Q.reshape(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)  # (self.batch_size, H, self.seq_len, self.d_head)
        K = K.reshape(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)  # (self.batch_size, self.n_head, self.seq_len, self.d_head)

        if self.mask is not self.seq_lenone:
            scores = np.where(self.mask == 0, -1e9, scores)

        attn = softmax(scores, axis=-1)             # (self.batch_size, self.n_head, self.seq_len, self.d_head)
        out_heads = attn @ V                        # (self.batch_size, self.n_head, self.seq_len, self.d_head)

        out_cat = out_heads.transpose(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.d_model)
        out = out_cat @ self.W_O                    # (self.batch_size, self.seq_len, self.d_model)

        # save cache
        self.Q = Q
        self.K = K
        self.V = V
        self.attn = attn
        self.out_heads = out_heads
        self.out_cat = out_cat

        return out

    # ==================================================
    # Backward
    # ==================================================
    def attention_backward(self, d_out):
        self.batch_size, self.seq_len, D = self.batch_size, self.seq_len, self.d_model
        self.n_head, self.d_head = self.n_head, self.d_head
        scale = np.sqrt(self.d_head)

        # ---------- W_O ----------
        d_out_cat = d_out @ self.W_O.T                         # (bath, seq_len, d_model)
        dW_O = (self.out_cat.reshape(self.batch_size * self.seq_len, self.d_model).T @ d_out.reshape(batch_size, self.seq_len, self.d_model))

        d_out_heads = d_out_cat.reshape(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)

        # ---------- V & Attention ----------
        d_V = self.attn.transpose(0, 1, 3, 2) @ d_out_heads # (batch, n_head, seq_len, d_head)

        d_attn = d_out_heads @ self.V.transpose(0, 1, 3, 2) # (batch, n_head, seq_len, d_head)

        # ---------- Softmax ----------
        d_Scores = softmax_backward(self.attn, d_attn)

        # ---------- Q & K ----------
        d_Q = (dScores @ self.K) / scale # (batch, n_head, seq_len, d_head)
        d_K = (dScores.transpose(0, 1, 3, 2) @ self.Q) / scale

        # ---------- reshape ----------
        d_Q = dQ.transpose(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.d_model)
        d_K = dK.transpose(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.d_model)
        d_V = dV.transpose(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.d_model)

        X_flat = self.X.reshape(self.batch_size * self.seq_len, self.d_model)

        # ---------- W_Q, W_K, W_V ----------
        d_W_Q = X_flat.T @ d_Q
        d_W_K = X_flat.T @ d_K
        d_W_V = X_flat.T @ d_V

        # ---------- dX ----------
        d_X = (
            d_Q @ self.W_Q.T + d_K @ self.W_K.T + d_V @ self.W_V.T
        ).reshape(self.batch_size, self.seq_len, self.d_model)

        self.W_Q = self.W_Q - self.alph * d_W_Q                             # (d_model, d_model)
        self.W_K = self.W_K - self.alph * d_W_K
        self.W_V = self.W_V - self.alph * d_W_V
        self.W_O = self.W_O - self.alph * d_W_O

        return d_X


def masked_attention_process(X_input, Q, K, V, batch, d_model, seq_len, n_head, d_attention):
    pass


def add_norm(X_input, Attention):
    pass


def neural_network_matrix_generate(layer_number, d_model, i):
    np.seed(i)
    return np.random.random(layer_number, d_model, d_model)


# fully_connected
def neural_network(neural_network, X_input):
    X_output = X_input.T
    for i in range(neural_network.shape[0]):
        X_output = neural_network[i, :, :].dot(X_output)
    return X_output



def Encoder(X_input, batch, d_model, n_head, K, Q, V, neural_network):
    d_attention = d_model / n_head
    X_input = positional_enconding(X_input)
    Attention = attention_process(X_input, Q, K, V, batch, d_model, seq_len, n_head, d_attention)
    X_input = add_norm(X_input, Attention)
    output_neural_network = neural_network(neural_network, X_input)
    X_input = add_norm(X_input, output_neural_network)
    return X_input


def Decoder(mode, output_Encoder, batch, d_model, n_head, K, Q, V, neural_network, token_right):
    if mode == 0:
        d_attention = d_model / n_head
        output_Encoder = positional_enconding(X_input)
        Attention = attention_process(output_Encoder, Q, K, V, batch, d_model, seq_len, n_head, d_attention)
        output_Encoder = add_norm(output_Encoder, Attention)
    pass


def feed_forward(mode, X_input, batch, d_model, seq_len, n_head, K_1, Q_1, V_1, K_2, Q_2, V_2, neural_network_1, neural_network_2):
    for i in range(X_input.shape[0]):
        output_Encoder = Encoder(X_input[i], batch, d_model, seq_len, n_head, K_1, Q_1, V_1, neural_network_1)
        output_Decoder = Decoder(mode, output_Encoder, batch, d_model, seq_len, n_head, K_2, Q_2, V_2, neural_network_2)
        """به ازای هر batch یک درایه به loss اضافه میشود."""
        loss = []
    return loss


def back_propagation():
    pass
#####################################################################
def transformer(X_input, mode, epoch):
    batch = X_input.shape[0]
    seq_len = X_input.shape(0)
    d_model = 512
    n_head = 8
    K_1, Q_1, V_1 = attention_matrix_generate(d_model, 1)
    neural_network_1 = neural_network_matrix_generate(8, d_model, 1)
    K_2, Q_2, V_2 = attention_matrix_generate(d_model, 2)
    m_K_2, m_Q_2, m_V_2 = attention_matrix_generate(d_model, 2)
    neural_network_2 = neural_network_matrix_generate(8, d_model, 2)
    if mode = 0:
        for i in range(epoch):
            #loss = feed_forward()
            loss = feed_forward(mode, X_input, batch, d_model, seq_len, n_head, K_1, Q_1, V_1, K_2, Q_2, V_2, neural_network_1, neural_network_2):
            #back_propagation(loss)




