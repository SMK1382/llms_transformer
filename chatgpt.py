import numpy as np




def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
        for i in range(1, d_model, 2):
            pe[pos, i] = np.cos(pos / (10000 ** ((i-1) / d_model)))
    return pe




def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)


class multi_head_attention():
    def __init__(self, X, W_Q, W_K, W_V, W_O, n_head, mask=None):
        self.batch_size, self.seq_len, self.d_model = X.shape
        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V
        self.W_O = W_O
        self.n_head = n_head
        self.d_head = self.d_model // self.n_head
        self.mask = mask
    def attention_compute(self):   
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
    
        Q = Q.reshape(batch_size, seq_len, n_head, d_head).transpose(0,2,1,3)
        K = K.reshape(batch_size, seq_len, n_head, d_head).transpose(0,2,1,3)
        V = V.reshape(batch_size, seq_len, n_head, d_head).transpose(0,2,1,3)
    
        scores = Q @ K.transpose(0,1,3,2) / np.sqrt(d_head)
        if mask is not None:
            scores = np.where(mask==0, -1e9, scores)
    
        attn = softmax(scores, axis=-1)
        out = attn @ V
    
        out = out.transpose(0,2,1,3).reshape(batch_size, seq_len, d_model)
    
        out_put = out @ W_O
        return out_put
    def attention_weigh_aupdate(self, loss):
        """
        d_L / d_output = loss
        d_L / d_W_O = d_L / d_output * d_output / d_W_O = out.T
        d_L / d_out = d_L / d_output * d_output / d_out = W_O.T
        # out = attn @ V
        d_L / d_V = d_L / d_out * d_out / d_V = attn.T
        d_L / d_attn = d_L / d_out * d_out / d_attn = V.T
        # attn  = softmax((Q @ K.T) / np.sqrt(d_head))
        # softmax backpropagation:

        """
        
        


class feed_forward_neural_network(X, W1, b1, W2, b2):
    return (np.maximum(0, X @ W1 + b1)) @ W2 + b2


def future_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len)))


def encoder_layer(X, params, n_head):
    attn_out = multi_head_attention(X, params['W_Q'], params['W_K'], params['W_V'], params['W_O'], n_head)
    X = layer_norm(X + attn_out)
    ff_out = feed_forward(X, params['W1'], params['b1'], params['W2'], params['b2'])
    X = layer_norm(X + ff_out)
    return X


def decoder_layer(X, enc_output, params, n_head, mask=None):
    # Masked Self-Attentiodecoder(X_input, enc_output, n_layer, n_head, params_list, mode=0):
    batch_size, seq_len, d_model = X_input.shape
    X_input = X_input + positional_encoding(seq_len, d_model)
    
    if mode == 0:
        mask = future_mask(seq_len)[None,None,:,:]
    else:
        mask = None
    
    for l in range(n_layer):
        X_input = decoder_layer(X_input, enc_output, params_list[l], n_head, mask)
    return X_input


def transformer_decoder_autoregressive(enc_output, start_token, max_len, n_layer, n_head, params_list, d_model):
    batch_size = enc_output.shape[0]
    generated = start_token
    
    for t in range(max_len):
        pos_enc = positional_encoding(generated.shape[1], d_model)
        dec_input = generated + pos_enc
        
        X = dec_input
        for l in range(n_layer):
            X = decoder_layer(X, enc_output, params_list[l], n_head, mask=future_mask(X.shape[1])[None,None,:,:])
        
        next_token = X[:, -1:, :]
        generated = np.concatenate([generated, next_token], axis=1)
    
    return generated


def transformer(X_input, target_input=None, n_layer=2, n_head=8, d_model=512, d_ff=2048, mode=0, max_len=10):
    enc_params = [init_params(d_model, n_head, d_ff) for _ in range(n_layer)]
    dec_params = [init_params(d_model, n_head, d_ff) for _ in range(n_layer)]
    
    enc_output = transformer_encoder(X_input, n_layer, n_head, enc_params)
    
    if mode == 0 and target_input is not None:
        dec_output = transformer_decoder(target_input, enc_output, n_layer, n_head, dec_params, mode)
    else:
        start_token = target_input  # (batch,1,d_model)
        dec_output = transformer_decoder_autoregressive(enc_output, start_token, max_len, n_layer, n_head, dec_params, d_model)
    
    return enc_output, dec_output


if __name__ == "__main__":
    batch_size = 2
    seq_len_src = 10
    seq_len_tgt = 1 
    d_model = 32

    X_input = np.random.randn(batch_size, seq_len_src, d_model)
    start_token = np.random.randn(batch_size, seq_len_tgt, d_model)

    enc_out, dec_out = transformer(X_input, start_token, n_layer=2, n_head=4, d_model=d_model, d_ff=64, mode=1, max_len=8)
    print("Encoder output shape:", enc_out.shape)
    print("Decoder generated sequence shape:", dec_out.shape)
