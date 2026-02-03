import numpy as np



def positional_encoding(X):
    batch, seq_len, d_model = X.shape
    for i in range(batch):
        for pos in range(seq_len):
            for j in range(0, d_model, 2):
                X[i, pos, j] += np.sin(pos / (10000 ** (2 * j / d_model)))
            for j in range(1, d_model, 2):
                X[i, pos, j] += np.cos(pos / (10000 ** (2 * (j-1) / d_model)))
    return X


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(A, dA):
    # A, dA: (self.batch_size, self.n_head, self.seq_len, self.d_head)
    temp = np.sum(A * dA, axis=-1, keepdims=True)
    dS = A * (dA - temp)
    return dS
#############################################################################################

class add_and_norm:
    def init(self, d_model, eps=1e-5,learning_rate = 0.1):
        """
        X     : (B, N, d_model)   residual input
        gamma : (d_model,)
        beta  : (d_model,)
        """
        self.X = None
        self.batch_size, self.seq_len, self.d_model = X.shape
        self.gamma = np.ones(d_model)
        self.beta = np.ones(d_model)
        self.eps = eps
        self.alpha = learning_rate

        # cache
        self.X_hat = None
        self.mean = None
        self.var = None
        self.Z = None   # X + sublayer output


    def get_input(self, X):
        self.X = X
    # ==================================================
    # Forward
    # ==================================================
    def forward(self, sublayer_out):
        """
        sublayer_out : (B, N, d_model)
        """
        # Residual connection
        Z = self.X + sublayer_out          

        # LayerNorm (per token, over D)
        mean = np.mean(Z, axis=-1, keepdims=True)
        var = np.var(Z, axis=-1, keepdims=True)

        X_hat = (Z - mean) / np.sqrt(var + self.eps)
        out = self.gamma * X_hat + self.beta

        # cache
        self.Z = Z
        self.mean = mean
        self.var = var
        self.X_hat = X_hat

        return out

    # ==================================================
    # Backward
    # ==================================================
    def backward(self, d_out):
        """
        d_Out : (B, N, d_model)
        """

        # ---------- gamma, beta ----------
        d_gamma = np.sum(d_out * self.X_hat, axis=(0, 1))
        d_beta = np.sum(d_out, axis=(0, 1))

        # ---------- LayerNorm backward ----------
        d_X_hat = d_out * self.gamma                  # (B, N, D)

        std_inv = 1.0 / np.sqrt(self.var + self.eps)

        d_var = np.sum(d_X_hat * (self.Z - self.mean) * -0.5 * std_inv**3, axis=-1, keepdims=True)

        d_mean = np.sum(d_X_hat * -std_inv, axis=-1, keepdims=True) + d_var * np.mean(-2.0 * (self.Z - self.mean), axis=-1, keepdims=True)

        d_Z = dX_hat * std_inv + d_var * 2.0 * (self.Z - self.mean) / self.d_model + d_mean / self.d_model

        # ---------- Residual split ----------
        d_X = d_Z
        d_Sublayer = d_Z

        self.gamma = self.gamma - self.alpha * d_gamma
        self.beta = self.beta - self.alpha * d_beta

        return d_X, d_Sublayer




class multi_head_attention:
    def init(self, d_model, n_head, learning_rate = 0.1, encoder=False, mask=False):
        self.W_Q = np.random.random(d_model, d_model)                              # (d_model, d_model)
        self.W_K = np.random.random(d_model, d_model)
        self.W_V = np.random.random(d_model, d_model)
        self.W_O = np.random.random(d_model, d_model)
        self.alph = learning_rate
        self.d_head = self.d_model // self.n_head
        self.n_head = n_head
        self.mask = mask
        self.encoder = encoder


        self.X = None                                 # (self.batch_size, self.seq_len, d_model)
        self.Q = None
        self.K = None
        self.V = None
        self.attn = None
        self.out_heads = None
        self.out_cat = None

    def get_input(self, X):
        self.X = X
        self.batch_size, self.seq_len, self.d_model = X.shape
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

        if self.mask is not False:
            mask = np.tril(np.ones(self.seq_len, self.seq_len), k=0)
            scores = np.where(self.mask == 1, scores, -1e9)

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
        dW_O = (self.out_cat.reshape(self.batch_size * self.seq_len, self.d_model).T @ d_out.reshape(batch_size * self.seq_len, self.d_model))

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
        d_Q = dQ.transpose(0, 2, 1, 3).reshape(self.batch_size * self.seq_len, self.d_model)
        d_K = dK.transpose(0, 2, 1, 3).reshape(self.batch_size * self.seq_len, self.d_model)
        d_V = dV.transpose(0, 2, 1, 3).reshape(self.batch_size * self.seq_len, self.d_model)

        X_flat = self.X.reshape(self.batch_size * self.seq_len, self.d_model)

        # ---------- W_Q, W_K, W_V ----------
        d_W_Q = X_flat.T @ d_Q
        d_W_K = X_flat.T @ d_K
        d_W_V = X_flat.T @ d_V

        self.W_Q -= self.alph * d_W_Q                           # (d_model, d_model)
        self.W_K -= self.alph * d_W_K
        self.W_V -= self.alph * d_W_V
        self.W_O -= self.alph * d_W_O


        # ---------- dX ----------
        d_X = (d_Q @ self.W_Q.T + d_K @ self.W_K.T + d_V @ self.W_V.T).reshape(self.batch_size, self.seq_len, self.d_model)

        if encoder:
            return (d_Q @ self.W_Q.T + d_K @ self.W_K.T).reshape(self.batch_size, self.seq_len, self.d_model),  (d_V @ self.W_V.T).reshape(self.batch_size, self.seq_len, self.d_model)

        return d_X





#####################################################################################

class feed_forward:
    def __init__(self, d_model):
        """
        X   : (B, N, d_model)
        W_1 : (d_model, d_ff)
        b_1 : (d_ff,)
        W_2 : (d_ff, d_model)
        b_2 : (d_model,)
        """
        self.W_1 = np.random.random(d_model, 8)
        self.b_1 = np.random.random(8)
        self.W_2 = np.random.random(8, d_model)
        self.b_2 = np.random.random(d_model)
        self.X = None

        # cache
        self.Z1 = None
        self.A1 = None
    
    def get_input(self, X):
        self.X = X
        self.batch_size, self.seq_len, self.d_model = X.shape
    # ==================================================
    # Forward
    # ==================================================
    def forward(self):

        X_flat = self.X.reshape(self.batch_size * self.seq_len, self.d_model)

        Z1 = X_flat @ self.W_1 + self.b_1           # (B*N, d_ff)

        A1 = np.maximum(0, Z1)                      # (B*N, d_ff)

        out = A1 @ self.W_2 + self.b_2              # (B*N, d_model)

        out = out.reshape(self.batch_size, self.seq_len, self.d_model)

        self.Z1 = Z1
        self.A1 = A1

        return out

    def backward(self, dOut):
        """
        dOut : (B, N, d_model)
        """
        B, N, D = self.batch_size, self.seq_len, self.d_model

        dOut_flat = dOut.reshape(self.batch_size * self.seq_len, self.d_model)

        # ---------- Linear 2 ----------
        dW_2 = self.A1.T @ dOut_flat                # (d_ff, d_model)
        db_2 = np.sum(dOut_flat, axis=0)            # (d_model,)

        dA1 = dOut_flat @ self.W_2.T                # (B*N, d_ff)

        # ---------- ReLU ----------
        dZ1 = dA1 * (self.Z1 > 0)                   # (B*N, d_ff)

        # ---------- Linear 1 ----------
        X_flat = self.X.reshape(self.batch_size * self.seq_len, self.d_model)

        dW_1 = X_flat.T @ dZ1                       # (d_model, d_ff)
        db_1 = np.sum(dZ1, axis=0)                  # (d_ff,)

        dX = dZ1 @ self.W_1.T                       # (B*N, d_model)
        dX = dX.reshape(self.batch_size * self.seq_len, self.d_model)

        return dX, dW_1, db_1, dW_2, db_2

class Linear:
    def __init__(self, d_in, d_out, learning_rate=0.1):
        self.W = np.random.randn(d_in, d_out) * 0.1
        self.b = np.zeros(d_out)
        self.lr = learning_rate

        # cache
        self.X = None

    # =====================================
    # Forward
    # =====================================
    def forward(self, X):
        """
        X : (B, N, d_in)
        return : (B, N, d_out)
        """
        self.X = X
        B, N, D_in = X.shape

        X_flat = X.reshape(B * N, D_in)
        out = X_flat @ self.W + self.b

        return out.reshape(B, N, -1)

    # =====================================
    # Backward
    # =====================================
    def backward(self, d_out):
        """
        d_out : (B, N, d_out)
        return dX : (B, N, d_in)
        """
        B, N, D_out = d_out.shape
        D_in = self.W.shape[0]

        d_out_flat = d_out.reshape(B * N, D_out)
        X_flat = self.X.reshape(B * N, D_in)

        # gradients
        dW = X_flat.T @ d_out_flat               # (D_in, D_out)
        db = np.sum(d_out_flat, axis=0)          # (D_out,)
        dX = d_out_flat @ self.W.T               # (B*N, D_in)

        # update
        self.W -= self.lr * dW
        self.b -= self.lr * db

        return dX.reshape(B, N, D_in)

#####################################################################################


def Encoder(X_input, n_head, K, Q, V, neural_network):
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


def forward(X, multi_head_attention_1, multi_head_attention_2, masked_multi_head_attention, feed_forward_1, feed_forward_2, norm_1, norm_2, norm_3, norm_4, norm_5):
    #Encoder
    X_1 = X_2 = positional_encoding(X)
    multi_head_attention_1.get_input(X_2)
    X_2 = multi_head_attention_1.forward()
    norm_1.get_input(X_2)
    X_1 = X_2 = norm_1.forward(X_1)
    feed_forward.get_input(X_2)
    X_2 = feed_forward.forward()
    norm_2.get_input(X_2)
    X_1 = X_2 = norm_2.forward(X_1)
    #======================================

#####################################################################
def transformer(X_input, n_head, epoch):
    batch, seq_len, d_model = X_input.shape
    multi_head_attention_1 = multi_head_attention(d_model, n_head)
    multi_head_attention_2 = multi_head_attention(d_model, n_head, encoder=True)
    masked_multi_head_attention = multi_head_attention(d_model, n_head, mask=True)
    feed_forward_1 = feed_forward(d_model)
    feed_forward_2 = feed_forward(d_model)
    norm_1 = add_and_norm(d_model)
    norm_2 = add_and_norm(d_model)
    norm_3 = add_and_norm(d_model)
    norm_4 = add_and_norm(d_model)
    norm_5 = add_and_norm(d_model)
    for i in range(epoch):
        loss = forward(X, n_head, multi_head_attention_1, multi_head_attention_2, masked_multi_head_attention, feed_forward_1, feed_forward_2, norm_1, norm_2, norm_3, norm_4, norm_5)
        backward(loss)
            #back_propagation(loss)




