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


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def attention_process(X_input, Q, K, V, batch, d_model, seq_len, n_head, d_attention):
    attention_all = np.array([])
    for i in range(X_input.shape[0]):
        #Permutation
        XQ = (X[i].dot(Q)).reshape(n_head , seq_len , d_attention)
        XK = (X[i].dot(K)).reshape(n_head , seq_len  , d_attention)
        XV = (X[i].dot(V)).reshape(n_head , seq_len , d_attention)
        np.append(attention_all, np.softmax(XQ.dot(XK.T)/ np.sqrt(d_model)) * XV)
    #UnPermutation
    return attention_all.reshape(batch , seq_len , d_model)


def attention_matrix_generate(d_model, i):
    np.seed(i)
    V = np.random.random(d_model, d_model)
    Q = np.random.random(d_model, d_model)
    K = np.random.random(d_model, d_model)
    return (K, Q, V)


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




