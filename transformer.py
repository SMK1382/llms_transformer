import numpy as np

def attention_matrix_generate(d_model):
    np.seed(1)
    V = np.random.random(d_model, d_model)
    Q = np.random.random(d_model, d_model)
    K = np.random.random(d_model, d_model)
    return (K, Q, V)


def positional_encoding(X_input):
    X_input[[:],[::2]] += np.sin(np.range(0, X_input.shape[-1], 2)/np.pow(1000, 2 * np.range(0, X_input.shape[-1], 2)))
    X_input[[:],[1::2]] += np.sin(np.range(1, X_input.shape[-1], 2)/np.pow(1000, 2 * np.range(1, X_input.shape[-1], 2)))
    return X_input


def add_norm(X_input, Attention):
    pass


def feed_forward_neural_network(Attention):
    


def attention_process(X_input, Q, K, V, batch, d_model, seq_len, n_head, d_attention):
    #Permutation
    XQ = (X.dot(Q)).reshape(batch * n_head * seq_len * d_attention)
    XK = (X.dot(K)).reshape(batch * n_head * seq_len  * d_attention)
    XV = (X.dot(V)).reshape(batch * n_head * seq_len * d_attention)
    Attention = np.softmax(XQ.dot(XK.T)/ np.sqrt(d_model)) * XV
    #UnPermutation
    return Attention.reshape(batch * seq_len * d_model)


def transformer_1(X_input):
    batch = 1
    seq_len = X_input.shape(0)
    d_model = 512
    n_head = 8
    d_attention = d_model / n_head
    X_input = positional_enconding(X_input)
    K, Q, V = attention_matrix_generate(d_model)
    Attention = attention_process(X_input, Q, K, V, batch, d_model, seq_len, n_head, d_attention)
    X_input = add_norm(X_input, Attention)
    X_input = feed_forward_neural_network(X_input)

#####################################################################
def transformer(X_input):
