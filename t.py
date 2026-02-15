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
    def __init__(self, d_model, eps=1e-5,learning_rate = 0.1):
        """
        X     : (B, N, d_model)   residual input
        gamma : (d_model,)
        beta  : (d_model,)
        """
        self.X = None
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
        self.batch_size, self.seq_len, self.d_model = X.shape
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
        dX_hat = d_out * self.gamma                  # (B, N, D)

        std_inv = 1.0 / np.sqrt(self.var + self.eps)

        d_var = np.sum(dX_hat * (self.Z - self.mean) * -0.5 * std_inv**3, axis=-1, keepdims=True)

        d_mean = np.sum(dX_hat * -std_inv, axis=-1, keepdims=True) + d_var * np.mean(-2.0 * (self.Z - self.mean), axis=-1, keepdims=True)

        d_Z = dX_hat * std_inv + d_var * 2.0 * (self.Z - self.mean) / self.d_model + d_mean / self.d_model

        # ---------- Residual split ----------
        d_X = d_Z
        d_Sublayer = d_Z

        self.gamma = self.gamma - self.alpha * d_gamma
        self.beta = self.beta - self.alpha * d_beta

        return d_X, d_Sublayer


#####################################################################################
class multi_head_attention:
    def __init__(self, d_model, n_head, softmax, learning_rate = 0.1, decoder=False, mask=False):
        self.W_Q = np.random.random((d_model, d_model))                              # (d_model, d_model)
        self.W_K = np.random.random((d_model, d_model))
        self.W_V = np.random.random((d_model, d_model))
        self.W_O = np.random.random((d_model, d_model))
        self.alph = learning_rate
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.mask = mask
        self.decoder = decoder
        self.softmax = softmax()

        self.X = None                                 # (self.batch_size, self.seq_len, d_model)
        self.Q = None
        self.K = None
        self.V = None
        self.attn = None
        self.out_heads = None
        self.out_cat = None

    def get_input(self, X):
        self.X = X
        if self.decoder:
            self.batch_size, self.seq_len, self.d_model = X[0].shape
        else:
            self.batch_size, self.seq_len, self.d_model = X.shape
    # ==================================================
    # Forward
    # ==================================================
    def attention_forward(self, padding_mask):
        #self.batch_size, self.seq_len,  = self.batch_size, self.seq_len, self.d_model
        #self.n_head, self.d_head = self.n_head, self.d_head
        
        if self.decoder:
            Q = self.X[0] @ self.W_Q # (self.batch_size, self.seq_len, D)
            K = self.X[0] @ self.W_K
            V = self.X[1] @ self.W_V

        else:
            Q = self.X @ self.W_Q # (self.batch_size, self.seq_len, D)
            K = self.X @ self.W_K
            V = self.X @ self.W_V

        Q = Q.reshape(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)  # (self.batch_size, H, self.seq_len, self.d_head)
        K = K.reshape(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)  # (self.batch_size, self.n_head, self.seq_len, self.d_head)


        if padding_mask is not None:
            mask_expanded = padding_mask[:, np.newaxis, np.newaxis, :]
            scores = np.where(mask_expanded == 1, scores, -1e9)


        if self.mask is not False:
            mask = np.tril(np.ones((self.seq_len, self.seq_len)), k=0)
            scores = np.where(self.mask == 1, scores, -1e9)

        attn = self.softmax.softmax_forward(scores, axis=-1)             # (self.batch_size, self.n_head, self.seq_len, self.d_head)
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
        d_Scores = self.softmax.softmax_backward(self.attn, d_attn)

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

        if self.decoder:
            return (d_Q @ self.W_Q.T + d_K @ self.W_K.T).reshape(self.batch_size, self.seq_len, self.d_model),  (d_V @ self.W_V.T).reshape(self.batch_size, self.seq_len, self.d_model)

        return d_X


#####################################################################################
class feed_forward:
    def __init__(self, d_model, learning_rate=0.1):
        """
        X   : (B, N, d_model)
        W_1 : (d_model, d_ff)
        b_1 : (d_ff,)
        W_2 : (d_ff, d_model)
        b_2 : (d_model,)
        """
        self.W_1 = np.random.random((d_model, 8))
        self.b_1 = np.random.random((8))
        self.W_2 = np.random.random((8, d_model))
        self.b_2 = np.random.random((d_model))

        self.learning_rate = learning_rate
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

        self.W_1 -= (self.learning_rate * dW_1)
        self.b_1 -= (self.learning_rate * db_1)
        self.W_2 -= (self.learning_rate * dW_2)
        self.b_2 -= (self.learning_rate * db_2)

        return dX.reshape(B, N, D)
#####################################################################################
class Linear:
    def __init__(self, d_model, d_out, learning_rate=0.1):
        self.W = np.random.randn(d_model, d_out) * 0.1
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
        self.batch_size, self.seq_len, self.d_model = X.shape

        X_flat = X.reshape(self.batch_size * self.seq_len, self.d_model)
        out = X_flat @ self.W + self.b
        out = out.reshape(self.batch_size, self.seq_len, -1)

        return out 

    # =====================================
    # Backward
    # =====================================
    def backward(self, d_out):
        B, N, D_out = d_out.shape
        D_in = self.W.shape[0]

        d_out_flat = d_out.reshape(B*N, D_out)
        X_flat = self.X.reshape(B*N, D_in)

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
    return (np.array(temp), np.where(np.array(temp)=='<pad>', 0, 1))



def cross_entropy_with_grad(vocabulary, target_model, output_model):
    B, T, V = output_model.shape

    target_index = np.array([], dtype='uint8')
    for i in target_model:
        for j in i:
            target_index = np.append(target_index, np.where(vocabulary == j)[0][0])

    output_flat = output_model.reshape(B*T, V)

    loss = -np.sum(np.log(output_flat[np.arange(B*T), target_index] + 1e-9))
    
    grad = output_flat.copy()
    #print('-' * 100)
    #print(target_index)
    #print('-' * 100)
    #print(grad)
    #print('-' * 100)

    grad[np.arange(B*T), target_index] -= 1
    grad = grad / (B*T)

    grad = grad.reshape(B, T, V)

    return loss, grad


def input_embedding(input_model, embedding):
    temp = np.array([])
    for i in input_model:
        for j in i:
            for key, value in embedding.items():
                if j == key:
                    temp = np.append(temp, value)
    return temp
#####################################################################################
def forward_train(X, vocabulary, padding, multi_head_attention_1, multi_head_attention_2, masked_multi_head_attention, feed_forward_1, feed_forward_2, norm_1, norm_2, norm_3, norm_4, norm_5, linear, softmax):
    #Encoder
    X_1 = X_2 = positional_encoding(X)
    multi_head_attention_1.get_input(X_2)
    X_2 = multi_head_attention_1.attention_forward(padding_matrix)
    norm_1.get_input(X_2)
    X_1 = X_2 = norm_1.forward(X_1)
    feed_forward_1.get_input(X_2)
    X_2 = feed_forward_1.forward()
    norm_2.get_input(X_2)
    output_encoder = norm_2.forward(X_1)
    #Decoder
    X_1 = X_2 = positional_encoding(X)
    masked_multi_head_attention.get_input(X_2)
    X_2 = masked_multi_head_attention.attention_forward(padding_matrix)
    norm_3.get_input(X_2)
    X_1 = X_2 = norm_3.forward(X_1)
    multi_head_attention_2.get_input(np.array([output_encoder, X_2]))
    X_2 = multi_head_attention_2.attention_forward(padding_matrix)
    norm_4.get_input(X_2)
    X_1 = X_2 = norm_4.forward(X_1)
    feed_forward_2.get_input(X_2)
    X_2 = feed_forward_2.forward()
    norm_5.get_input(X_2)
    X_2 = norm_5.forward(X_1)
    X_1 = linear.forward(X_2)
    X_1 = softmax.softmax_forward(X_1)
    #print(np.max(X_1[0, 0]))#temporary
    #print(X_1.shape)
    return cross_entropy_with_grad(vocabulary, target_model, X_1)


    #======================================
def backward_train(loss, multi_head_attention_1, multi_head_attention_2, masked_multi_head_attention, feed_forward_1, feed_forward_2, norm_1, norm_2, norm_3, norm_4, norm_5, linear, softmax):
    X = softmax.softmax_backward(loss)
    X = linear.backward(X)
    X, sub_layer = norm_5.backward(X)
    X = feed_forward_2.backward(X)
    X, sub_layer = norm_4.backward(X + sub_layer)
    print(X.shape)
    print(sub_layer.shape)
#####################################################################################
class transformer:
    def __init__(self, d_model, n_head, softmax, vocabulary):
        self.d_model = d_model
        self.n_head = n_head
        self.multi_head_attention_1 = multi_head_attention(d_model=d_model, n_head=n_head, softmax=softmax)
        self.multi_head_attention_2 = multi_head_attention(d_model=d_model, n_head=n_head, softmax=softmax, decoder=True)
        self.masked_multi_head_attention = multi_head_attention(d_model, n_head, softmax=softmax, mask=True)
        self.feed_forward_1 = feed_forward(d_model)
        self.feed_forward_2 = feed_forward(d_model)
        self.norm_1 = add_and_norm(d_model)
        self.norm_2 = add_and_norm(d_model)
        self.norm_3 = add_and_norm(d_model)
        self.norm_4 = add_and_norm(d_model)
        self.norm_5 = add_and_norm(d_model)
        self.linear = Linear(d_model, len(vocabulary))
        self.softmax = softmax()
               


    def train(self, X, vocabulary, padding, epoch):
        for i in range(epoch):
            loss, grad = forward_train(X, vocabulary, padding, self.multi_head_attention_1, self.multi_head_attention_2, self.masked_multi_head_attention, self.feed_forward_1, self.feed_forward_2, self.norm_1, self.norm_2, self.norm_3, self.norm_4, self.norm_5, self.linear, self.softmax)
            print(f"epoch: {i+1} | loss => ", loss)
            print("-" * 100)
            backward_train(grad, self.multi_head_attention_1, self.multi_head_attention_2, self.masked_multi_head_attention, self.feed_forward_1, self.feed_forward_2, self.norm_1, self.norm_2, self.norm_3, self.norm_4, self.norm_5, self.linear, self.softmax)

    def inference(self, X):
        pass
    
#=======================================

batch_size = 3
d_model = 512
n_head = 8


with open("samples.txt", 'r') as file:
    samples = file.read().lower().split(".\n")
    file.close()


X = []
targets = []
for i in range(0, len(samples) - 1):
    X.append(["<bos>"] + samples[i].split(" "))
    targets.append(samples[i].split(" ") + ["<eos>"])



input_model = []
target_model = []
for i in range(batch_size):
    temp = random.randint(0, 99)
    input_model.append(X[temp])
    target_model.append(targets[temp])

input_model, padding_matrix = padding(input_model)
target_model, _ = padding(target_model)
print("=" * 100)


input_model = input_embedding(input_model, embedding).reshape(input_model.shape[0], input_model.shape[1], d_model)

transformer = transformer(d_model, n_head, softmax, vocabulary)
transformer.train(input_model, vocabulary, padding_matrix, 200)
