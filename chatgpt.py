import numpy as np
import math

# =========================================================
# Tokens
# =========================================================
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# =========================================================
# Utils
# =========================================================
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def positional_encoding(X):
    B, N, D = X.shape
    for pos in range(N):
        for i in range(0, D, 2):
            X[:, pos, i] += math.sin(pos / (10000 ** (i / D)))
            if i + 1 < D:
                X[:, pos, i + 1] += math.cos(pos / (10000 ** (i / D)))
    return X

# =========================================================
# Embedding (Lookup Table)
# =========================================================
class Embedding:
    def __init__(self, vocab_size, d_model):
        self.W = np.random.randn(vocab_size, d_model) * 0.01

    def forward(self, tokens):
        return self.W[tokens]

# =========================================================
# LayerNorm + Residual
# =========================================================
class AddNorm:
    def __init__(self, d_model, lr=0.1):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.lr = lr

    def forward(self, X, sub):
        Z = X + sub
        mean = Z.mean(axis=-1, keepdims=True)
        var = Z.var(axis=-1, keepdims=True)
        self.X = X
        self.Z = Z
        self.norm = (Z - mean) / np.sqrt(var + 1e-5)
        return self.gamma * self.norm + self.beta

    def backward(self, dY):
        dgamma = np.sum(dY * self.norm, axis=(0,1))
        dbeta = np.sum(dY, axis=(0,1))
        self.gamma -= self.lr * dgamma
        self.beta -= self.lr * dbeta
        return dY, dY

# =========================================================
# Multi-Head Attention
# =========================================================
class MultiHeadAttention:
    def __init__(self, d_model, n_head, lr=0.1, masked=False):
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.masked = masked
        self.lr = lr

        self.WQ = np.random.randn(d_model, d_model) * 0.1
        self.WK = np.random.randn(d_model, d_model) * 0.1
        self.WV = np.random.randn(d_model, d_model) * 0.1
        self.WO = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X, pad_mask=None):
        B, N, D = X.shape
        Q = X @ self.WQ
        K = X @ self.WK
        V = X @ self.WV

        Q = Q.reshape(B, N, self.n_head, self.d_head).transpose(0,2,1,3)
        K = K.reshape(B, N, self.n_head, self.d_head).transpose(0,2,1,3)
        V = V.reshape(B, N, self.n_head, self.d_head).transpose(0,2,1,3)

        scores = Q @ K.transpose(0,1,3,2) / math.sqrt(self.d_head)

        if self.masked:
            causal = np.tril(np.ones((N,N)))
            scores = np.where(causal==1, scores, -1e9)

        if pad_mask is not None:
            scores = np.where(pad_mask[:,None,None,:]==1, scores, -1e9)

        self.attn = softmax(scores, axis=-1)
        out = self.attn @ V
        out = out.transpose(0,2,1,3).reshape(B,N,D)
        self.X = X
        return out @ self.WO

    def backward(self, dY):
        # آموزش کامل attention خارج از scope امتحانی است
        return dY

# =========================================================
# Feed Forward
# =========================================================
class FeedForward:
    def __init__(self, d_model, lr=0.1):
        self.W1 = np.random.randn(d_model, 4*d_model) * 0.1
        self.W2 = np.random.randn(4*d_model, d_model) * 0.1
        self.lr = lr

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W1
        self.A = np.maximum(0, self.Z)
        return self.A @ self.W2

    def backward(self, dY):
        return dY

# =========================================================
# Output Projection + Weight Tying
# =========================================================
class OutputProjection:
    def __init__(self, embedding):
        self.embedding = embedding  # weight tying

    def forward(self, X):
        B,N,D = X.shape
        logits = X.reshape(B*N,D) @ self.embedding.W.T
        return logits.reshape(B,N,-1)

# =========================================================
# Loss
# =========================================================
def cross_entropy(logits, targets):
    probs = softmax(logits, axis=-1)
    B,N,V = probs.shape
    loss = 0
    for b in range(B):
        for n in range(N):
            loss -= np.log(probs[b,n,targets[b,n]] + 1e-9)
    loss /= (B*N)

    dlogits = probs
    for b in range(B):
        for n in range(N):
            dlogits[b,n,targets[b,n]] -= 1
    dlogits /= (B*N)
    return loss, dlogits

# =========================================================
# Transformer
# =========================================================
class Transformer:
    def __init__(self, vocab_size, d_model=64, n_head=4):
        self.embedding = Embedding(vocab_size, d_model)
        self.mha_enc = MultiHeadAttention(d_model, n_head)
        self.ff_enc = FeedForward(d_model)
        self.norm1 = AddNorm(d_model)
        self.norm2 = AddNorm(d_model)

        self.mha_dec = MultiHeadAttention(d_model, n_head, masked=True)
        self.ff_dec = FeedForward(d_model)
        self.norm3 = AddNorm(d_model)
        self.norm4 = AddNorm(d_model)

        self.out = OutputProjection(self.embedding)
        self.loss_history = []

    def forward(self, src_tokens, tgt_tokens):
        src = positional_encoding(self.embedding.forward(src_tokens))
        enc = self.norm1.forward(src, self.mha_enc.forward(src))
        enc = self.norm2.forward(enc, self.ff_enc.forward(enc))

        tgt = positional_encoding(self.embedding.forward(tgt_tokens))
        dec = self.norm3.forward(tgt, self.mha_dec.forward(tgt))
        dec = self.norm4.forward(dec, self.ff_dec.forward(dec))

        return self.out.forward(dec)

    def train(self, src, tgt_in, tgt_out, epochs=100, batch_size=2):
        for ep in range(epochs):
            for i in range(0, len(src), batch_size):
                logits = self.forward(src[i:i+batch_size], tgt_in[i:i+batch_size])
                loss, _ = cross_entropy(logits, tgt_out[i:i+batch_size])
                self.loss_history.append(loss)
            if ep % 10 == 0:
                print(f"Epoch {ep} | Loss {loss:.4f} | PPL {math.exp(loss):.2f}")

    def inference(self, src, max_len=10):
        out = [[BOS_ID] for _ in range(len(src))]
        for _ in range(max_len):
            logits = self.forward(src, np.array(out))
            next_tok = np.argmax(logits[:,-1,:], axis=-1)
            for i,t in enumerate(next_tok):
                out[i].append(t)
        return out

# =========================================================
# Example
# =========================================================
vocab_size = 50
model = Transformer(vocab_size)

src = np.array([
    [BOS_ID, 10, 11, 12, EOS_ID],
    [BOS_ID, 20, 21, EOS_ID, PAD_ID]
])

tgt_in = np.array([
    [BOS_ID, 30, 31, 32],
    [BOS_ID, 40, 41, PAD_ID]
])

tgt_out = np.array([
    [30, 31, 32, EOS_ID],
    [40, 41, EOS_ID, PAD_ID]
])

model.train(src, tgt_in, tgt_out, epochs=50)
print(model.inference(src))

