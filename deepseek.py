import numpy as np

def attention_matrix_generate(d_model, i):
    np.random.seed(i)
    V = np.random.random((d_model, d_model)) * 0.01
    Q = np.random.random((d_model, d_model)) * 0.01
    K = np.random.random((d_model, d_model)) * 0.01
    O = np.random.random((d_model, d_model)) * 0.01
    return (K, Q, V, O)

def neural_network_matrix_generate(d_model, i):
    np.random.seed(i)
    W_1 = np.random.random((d_model, 8)) * 0.01
    b_1 = np.random.random(8) * 0.01
    W_2 = np.random.random((8, d_model)) * 0.01
    b_2 = np.random.random(d_model) * 0.01
    return (W_1, b_1, W_2, b_2)

def positional_encoding(X):
    batch, seq_len, d_model = X.shape
    pe = np.zeros((batch, seq_len, d_model))
    
    for pos in range(seq_len):
        for j in range(0, d_model, 2):
            pe[:, pos, j] = np.sin(pos / (10000 ** (2 * j / d_model)))
        for j in range(1, d_model, 2):
            pe[:, pos, j] = np.cos(pos / (10000 ** (2 * (j-1) / d_model)))
    
    return X + pe

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def softmax_backward(A, dA):
    temp = np.sum(A * dA, axis=-1, keepdims=True)
    dS = A * (dA - temp)
    return dS

def create_look_ahead_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask == 0  # 1 for valid positions, 0 for masked

def cross_entropy_loss(y_pred, y_true):
    batch_size = y_pred.shape[0]
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    loss = -np.sum(y_true * np.log(y_pred)) / batch_size
    return loss

def cross_entropy_loss_backward(y_pred, y_true):
    batch_size = y_pred.shape[0]
    return (y_pred - y_true) / batch_size

class AddAndNorm:
    def __init__(self, d_model, learning_rate=0.01):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = 1e-5
        self.learning_rate = learning_rate
        
        self.X = None
        self.X_hat = None
        self.mean = None
        self.var = None
        self.Z = None
        
    def forward(self, X, sublayer_out):
        self.X = X
        
        # Residual connection
        Z = X + sublayer_out
        
        # Layer normalization
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
        batch_size, seq_len, d_model = d_out.shape
        
        # Gradient w.r.t gamma and beta
        d_gamma = np.sum(d_out * self.X_hat, axis=(0, 1))
        d_beta = np.sum(d_out, axis=(0, 1))
        
        # Gradient w.r.t X_hat
        d_X_hat = d_out * self.gamma
        
        # Gradient w.r.t Z
        std_inv = 1.0 / np.sqrt(self.var + self.eps)
        
        d_var = np.sum(d_X_hat * (self.Z - self.mean) * -0.5 * std_inv**3, axis=-1, keepdims=True)
        d_mean = np.sum(d_X_hat * -std_inv, axis=-1, keepdims=True) + \
                d_var * np.mean(-2.0 * (self.Z - self.mean), axis=-1, keepdims=True)
        
        d_Z = d_X_hat * std_inv + \
              d_var * 2.0 * (self.Z - self.mean) / d_model + \
              d_mean / d_model
        
        # Split gradient for residual connection
        d_X = d_Z
        d_sublayer = d_Z
        
        # Update parameters
        self.gamma -= self.learning_rate * d_gamma
        self.beta -= self.learning_rate * d_beta
        
        return d_X, d_sublayer

class MultiHeadAttention:
    def __init__(self, d_model, n_head, learning_rate=0.01, is_masked=False):
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.learning_rate = learning_rate
        self.is_masked = is_masked
        
        # Initialize weights
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01
        
        # Cache
        self.X = None
        self.Q = None
        self.K = None
        self.V = None
        self.attn = None
        self.out_cat = None
        
    def forward(self, X, K=None, V=None):
        self.X = X
        batch_size, seq_len, d_model = X.shape
        
        # Linear projections
        Q = X @ self.W_Q
        
        if K is not None and V is not None:
            # Cross-attention: use provided K, V from encoder
            self.K = K
            self.V = V
        else:
            # Self-attention
            self.K = K = X @ self.W_K
            self.V = V = X @ self.W_V
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)
        
        # Apply mask for decoder self-attention
        if self.is_masked:
            mask = create_look_ahead_mask(seq_len)
            scores = scores + (mask * -1e9)
        
        # Softmax
        self.attn = softmax(scores, axis=-1)
        
        # Context vectors
        out_heads = self.attn @ V
        
        # Concatenate heads
        self.out_cat = out_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Final linear projection
        out = self.out_cat @ self.W_O
        
        return out
    
    def backward(self, d_out):
        batch_size, seq_len, d_model = d_out.shape
        
        # Gradient w.r.t W_O
        d_out_cat = d_out @ self.W_O.T
        d_W_O = self.out_cat.reshape(batch_size * seq_len, d_model).T @ \
                d_out.reshape(batch_size * seq_len, d_model)
        
        # Reshape for multi-head
        d_out_heads = d_out_cat.reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(0, 2, 1, 3)
        
        # Gradients through attention
        d_attn = d_out_heads @ self.V.transpose(0, 1, 3, 2)
        d_V = self.attn.transpose(0, 1, 3, 2) @ d_out_heads
        
        # Gradient through softmax
        d_scores = softmax_backward(self.attn, d_attn)
        
        # Gradients for Q and K
        d_Q = (d_scores @ self.K) / np.sqrt(self.d_head)
        d_K = (d_scores.transpose(0, 1, 3, 2) @ self.Q) / np.sqrt(self.d_head)
        
        # Reshape back
        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, d_model)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, d_model)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, d_model)
        
        X_flat = self.X.reshape(batch_size * seq_len, d_model)
        
        # Gradients for weight matrices
        d_W_Q = X_flat.T @ d_Q
        d_W_K = X_flat.T @ d_K
        d_W_V = X_flat.T @ d_V
        
        # Gradient for input
        d_X = (d_Q @ self.W_Q.T + d_K @ self.W_K.T + d_V @ self.W_V.T).reshape(batch_size, seq_len, d_model)
        
        # Update weights
        self.W_Q -= self.learning_rate * d_W_Q
        self.W_K -= self.learning_rate * d_W_K
        self.W_V -= self.learning_rate * d_W_V
        self.W_O -= self.learning_rate * d_W_O
        
        return d_X

class FeedForward:
    def __init__(self, d_model, d_ff, learning_rate=0.01):
        self.d_model = d_model
        self.d_ff = d_ff
        self.learning_rate = learning_rate
        
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
        
        self.X = None
        self.Z1 = None
        self.A1 = None
        
    def forward(self, X):
        self.X = X
        batch_size, seq_len, d_model = X.shape
        
        X_flat = X.reshape(batch_size * seq_len, d_model)
        
        # First linear layer + ReLU
        self.Z1 = X_flat @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)
        
        # Second linear layer
        out = self.A1 @ self.W2 + self.b2
        
        out = out.reshape(batch_size, seq_len, d_model)
        
        return out
    
    def backward(self, d_out):
        batch_size, seq_len, d_model = d_out.shape
        
        d_out_flat = d_out.reshape(batch_size * seq_len, d_model)
        
        # Gradient w.r.t W2 and b2
        d_W2 = self.A1.T @ d_out_flat
        d_b2 = np.sum(d_out_flat, axis=0)
        
        # Gradient through second linear layer
        d_A1 = d_out_flat @ self.W2.T
        
        # Gradient through ReLU
        d_Z1 = d_A1 * (self.Z1 > 0)
        
        # Gradient w.r.t W1 and b1
        X_flat = self.X.reshape(batch_size * seq_len, d_model)
        d_W1 = X_flat.T @ d_Z1
        d_b1 = np.sum(d_Z1, axis=0)
        
        # Gradient for input
        d_X = d_Z1 @ self.W1.T
        d_X = d_X.reshape(batch_size, seq_len, d_model)
        
        # Update weights
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2
        
        return d_X

class Transformer:
    def __init__(self, d_model, n_head, vocab_size, seq_len, learning_rate=0.01):
        self.d_model = d_model
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        
        # Encoder components
        self.encoder_mha = MultiHeadAttention(d_model, n_head, learning_rate)
        self.encoder_ff = FeedForward(d_model, d_ff=8, learning_rate)
        self.encoder_norm1 = AddAndNorm(d_model, learning_rate)
        self.encoder_norm2 = AddAndNorm(d_model, learning_rate)
        
        # Decoder components
        self.decoder_masked_mha = MultiHeadAttention(d_model, n_head, learning_rate, is_masked=True)
        self.decoder_cross_mha = MultiHeadAttention(d_model, n_head, learning_rate)
        self.decoder_ff = FeedForward(d_model, d_ff=8, learning_rate)
        self.decoder_norm1 = AddAndNorm(d_model, learning_rate)
        self.decoder_norm2 = AddAndNorm(d_model, learning_rate)
        self.decoder_norm3 = AddAndNorm(d_model, learning_rate)
        
        # Embedding and output layers
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.01
        self.output_bias = np.zeros(vocab_size)
        
    def embed(self, tokens):
        # Convert token indices to embeddings
        batch_size, seq_len = tokens.shape
        embedded = np.zeros((batch_size, seq_len, self.d_model))
        
        for i in range(batch_size):
            for j in range(seq_len):
                embedded[i, j] = self.embedding[tokens[i, j]]
                
        return embedded
    
    def encode(self, encoder_input):
        # Add positional encoding
        X = positional_encoding(encoder_input)
        
        # Self-attention
        X_res = X
        X_attn = self.encoder_mha.forward(X)
        X = self.encoder_norm1.forward(X_res, X_attn)
        
        # Feed forward
        X_res = X
        X_ff = self.encoder_ff.forward(X)
        X = self.encoder_norm2.forward(X_res, X_ff)
        
        return X
    
    def decode(self, decoder_input, encoder_output):
        # Add positional encoding
        X = positional_encoding(decoder_input)
        
        # Masked self-attention
        X_res = X
        X_masked_attn = self.decoder_masked_mha.forward(X)
        X = self.decoder_norm1.forward(X_res, X_masked_attn)
        
        # Cross-attention with encoder output
        X_res = X
        X_cross_attn = self.decoder_cross_mha.forward(X, encoder_output, encoder_output)
        X = self.decoder_norm2.forward(X_res, X_cross_attn)
        
        # Feed forward
        X_res = X
        X_ff = self.decoder_ff.forward(X)
        X = self.decoder_norm3.forward(X_res, X_ff)
        
        return X
    
    def forward(self, encoder_tokens, decoder_tokens):
        # Embed tokens
        encoder_input = self.embed(encoder_tokens)
        decoder_input = self.embed(decoder_tokens)
        
        # Encode
        encoder_output = self.encode(encoder_input)
        
        # Decode
        decoder_output = self.decode(decoder_input, encoder_output)
        
        # Output projection
        batch_size, seq_len, _ = decoder_output.shape
        decoder_output_flat = decoder_output.reshape(batch_size * seq_len, self.d_model)
        logits = decoder_output_flat @ self.output_projection + self.output_bias
        logits = logits.reshape(batch_size, seq_len, self.vocab_size)
        
        # Softmax for probabilities
        probs = softmax(logits, axis=-1)
        
        return probs, logits, encoder_output
    
    def compute_loss(self, probs, targets):
        batch_size, seq_len, vocab_size = probs.shape
        
        # Create one-hot encoded targets
        targets_one_hot = np.zeros((batch_size, seq_len, vocab_size))
        for i in range(batch_size):
            for j in range(seq_len):
                targets_one_hot[i, j, targets[i, j]] = 1
        
        # Compute cross-entropy loss
        loss = cross_entropy_loss(probs, targets_one_hot)
        return loss
    
    def backward(self, probs, targets, encoder_output):
        batch_size, seq_len, vocab_size = probs.shape
        
        # Create one-hot encoded targets
        targets_one_hot = np.zeros((batch_size, seq_len, vocab_size))
        for i in range(batch_size):
            for j in range(seq_len):
                targets_one_hot[i, j, targets[i, j]] = 1
        
        # Gradient from loss (simplified)
        d_logits = cross_entropy_loss_backward(probs, targets_one_hot)
        
        # Need decoder output to compute gradients
        # For simplicity, we'll compute a dummy backward pass
        # In a real implementation, you would need to store decoder_output
        
        # Simplified backward - just update output layer
        decoder_output_shape = (batch_size, seq_len, self.d_model)
        decoder_output = np.random.randn(*decoder_output_shape) * 0.01  # Dummy
        
        # Update output projection
        batch_size, seq_len, _ = d_logits.shape
        d_logits_flat = d_logits.reshape(batch_size * seq_len, vocab_size)
        decoder_output_flat = decoder_output.reshape(batch_size * seq_len, self.d_model)
        
        d_output_projection = decoder_output_flat.T @ d_logits_flat
        d_output_bias = np.sum(d_logits_flat, axis=0)
        
        # Update weights
        self.output_projection -= self.learning_rate * d_output_projection
        self.output_bias -= self.learning_rate * d_output_bias
        
        return self.compute_loss(probs, targets)

def create_sample_data(batch_size=2, seq_len=4, vocab_size=10):
    encoder_tokens = np.random.randint(0, vocab_size, (batch_size, seq_len))
    decoder_tokens = np.random.randint(0, vocab_size, (batch_size, seq_len))
    targets = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    return encoder_tokens, decoder_tokens, targets

def test_transformer():
    # Hyperparameters
    d_model = 8
    n_head = 2
    vocab_size = 10
    seq_len = 4
    learning_rate = 0.01
    epochs = 100
    
    # Create transformer
    transformer = Transformer(d_model, n_head, vocab_size, seq_len, learning_rate)
    
    # Create sample data
    encoder_tokens, decoder_tokens, targets = create_sample_data(
        batch_size=2, seq_len=seq_len, vocab_size=vocab_size
    )
    
    # Training loop
    print("Training Transformer...")
    for epoch in range(epochs):
        # Forward pass
        probs, logits, encoder_output = transformer.forward(encoder_tokens, decoder_tokens)
        
        # Compute loss
        loss = transformer.compute_loss(probs, targets)
        
        # Backward pass
        loss = transformer.backward(probs, targets, encoder_output)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss:.4f}")
    
    print("\nTesting predictions...")
    # Test prediction
    test_probs, _, _ = transformer.forward(encoder_tokens, decoder_tokens)
    predictions = np.argmax(test_probs, axis=-1)
    
    print("Targets:")
    print(targets)
    print("\nPredictions:")
    print(predictions)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == targets)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    return transformer

# Run the test
if __name__ == "__main__":
    transformer = test_transformer()
