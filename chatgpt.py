import numpy as np

# --- Helper Functions ---
def attention_matrix_generate(d_model, i):
    np.random.seed(i)
    # Ensure output dimensions match expected usage in multi_head_attention
    # Weights should be (d_model, d_model)
    V = np.random.rand(d_model, d_model) * 0.1 # Small random values
    Q = np.random.rand(d_model, d_model) * 0.1
    K = np.random.rand(d_model, d_model) * 0.1
    O = np.random.rand(d_model, d_model) * 0.1
    return (K, Q, V, O) # Returning in K, Q, V, O order as used in MHA init

def neural_network_matrix_generate(d_model, i):
    np.random.seed(i)
    # d_ff is often larger than d_model, a common choice is 4 * d_model
    d_ff = d_model * 4
    W_1 = np.random.rand(d_model, d_ff) * 0.1
    b_1 = np.random.rand(d_ff) * 0.1
    W_2 = np.random.rand(d_ff, d_model) * 0.1
    b_2 = np.random.rand(d_model) * 0.1
    return (W_1, b_1, W_2, b_2)

def positional_encoding(X):
    batch, seq_len, d_model = X.shape
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    # Add positional encoding to input X
    # Ensure broadcasting works correctly: pe shape (seq_len, d_model)
    # X shape (batch, seq_len, d_model) -> needs X + pe[np.newaxis, :, :]
    X = X + pe[np.newaxis, :, :]
    return X

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def softmax_backward(A, dA):
    # A: softmax output, dA: gradient of loss w.r.t. softmax output
    # This function computes the gradient of the loss w.r.t. the input of the softmax function (scores)
    # A, dA: (batch_size, n_head, seq_len, seq_len) - assuming this shape for attention scores
    # Adjusting shapes for clarity if needed based on actual usage
    # For attention, dA is d_Scores from the previous layer (d_out_heads @ V.T)

    # The derivative of softmax(x) w.r.t. x_i is:
    # d(softmax(x)_i) / d(x_j) = softmax(x)_i * (delta_ij - softmax(x)_j)
    # Applying chain rule: d(Loss)/d(x_j) = sum_i (d(Loss)/d(softmax(x)_i)) * (d(softmax(x)_i)/d(x_j))

    # A is the softmax output (attn weights)
    # dA is the gradient arriving from the next layer (d_Scores from MHA backward)

    # For each output element, compute the gradient contribution to all input elements.
    # sum_term = np.sum(A * dA, axis=-1, keepdims=True) # This is sum(softmax(x)_i * grad_i) over i
    # d_scores = A * (dA - sum_term) # This formula is correct

    # Let's re-verify the shapes and calculation:
    # A is (B, H, N, N), dA is (B, H, N, N)
    # We want dScores of shape (B, H, N, N)

    # Expand dA and A for element-wise multiplication and sum
    # dA shape (B, H, N, N)
    # A shape (B, H, N, N)

    # For each output position (k) in the attention matrix, compute its gradient w.r.t. all input positions (j)
    # dScores[b, h, i, j] = dA[b, h, i, k] * (A[b, h, i, j] * (1 - A[b, h, i, k]))  if j == k
    # dScores[b, h, i, j] = dA[b, h, i, k] * (-A[b, h, i, j] * A[b, h, i, k])       if j != k

    # This can be simplified. The gradient dLoss/dScores[j] is a sum over i of (dLoss/dAttn[i]) * (dAttn[i]/dScores[j])
    # dAttn[i]/dScores[j] = attn[i] * (delta_ij - attn[j])

    # A is attn, dA is d_attn (gradient w.r.t. attn)
    # d_scores = np.zeros_like(dA)
    # for i in range(dA.shape[-2]): # Iterate over output sequence length (N)
    #     for j in range(dA.shape[-1]): # Iterate over input sequence length (N)
    #         if i == j:
    #             d_scores[:, :, i, j] = dA[:, :, i, j] * A[:, :, i, j] * (1 - A[:, :, i, j])
    #         else:
    #             d_scores[:, :, i, j] = dA[:, :, i, j] * (-A[:, :, i, j] * A[:, :, i, i]) # This seems wrong

    # Correct formula for gradient of softmax with respect to its input:
    # d(Loss)/d(Scores) = d(Loss)/d(Attn) * d(Attn)/d(Scores)
    # where d(Attn)/d(Scores) is a Jacobian matrix.
    # For element Scores_k, the gradient of Loss wrt Scores_k is:
    # sum_i (dLoss/dAttn_i * dAttn_i/dScores_k)
    # dAttn_i/dScores_k = Attn_i * (delta_ik - Attn_k)
    # So, dLoss/dScores_k = sum_i (dLoss/dAttn_i * Attn_i * (delta_ik - Attn_k))
    # dLoss/dScores_k = sum_i (dLoss/dAttn_i * Attn_i) - sum_i (dLoss/dAttn_i * Attn_i * Attn_k)
    # dLoss/dScores_k = sum_i (dA_i * A_i) - Attn_k * sum_i (dA_i * A_i)
    # dLoss/dScores_k = (dA - sum(dA * A, axis=-1, keepdims=True)) * A

    # The original implementation appears to be correct based on common sources:
    temp = np.sum(A * dA, axis=-1, keepdims=True)
    dS = A * (dA - temp)
    return dS


# --- Layer Classes ---

class add_and_norm:
    def __init__(self, learning_rate=0.1, eps=1e-5):
        """
        Initializes the Add and Norm layer.

        Args:
            learning_rate (float): The learning rate for weight updates.
            eps (float): A small value to prevent division by zero in LayerNorm.
        """
        self.X = None  # Stores the input for the residual connection
        self.sublayer_out = None # Stores the output of the sublayer
        self.learning_rate = learning_rate
        self.eps = eps

        # Parameters (initialized to identity initially, as in the original code)
        self.gamma = None
        self.beta = None

        # Cache for backward pass
        self.Z = None  # Result of X + sublayer_out
        self.mean = None
        self.var = None
        self.X_hat = None # Normalized output before scaling and shifting
        self.d_model = None # Will be inferred from X

    def get_input(self, X):
        """Stores the input X for the residual connection."""
        self.X = X
        if self.d_model is None: # Infer d_model on first input
            self.d_model = X.shape[-1]
            self.gamma = np.ones((self.d_model,)) # (D,)
            self.beta = np.zeros((self.d_model,)) # (D,)

    def forward(self, sublayer_out):
        """
        Performs the forward pass of the Add and Norm layer.

        Args:
            sublayer_out: The output from the preceding sublayer (e.g., Multi-Head Attention or Feed-Forward Network).
                          Shape: (batch_size, seq_len, d_model)

        Returns:
            The output after residual connection and layer normalization.
            Shape: (batch_size, seq_len, d_model)
        """
        if self.X is None:
            raise ValueError("Input X not set. Call get_input(X) first.")

        self.sublayer_out = sublayer_out # Store for backward pass
        # Residual connection
        self.Z = self.X + sublayer_out # (B, N, D)

        # Layer Normalization (applied across the d_model dimension for each token)
        self.mean = np.mean(self.Z, axis=-1, keepdims=True) # (B, N, 1)
        self.var = np.var(self.Z, axis=-1, keepdims=True)   # (B, N, 1)

        # Normalize
        self.X_hat = (self.Z - self.mean) / np.sqrt(self.var + self.eps) # (B, N, D)

        # Scale and Shift
        out = self.gamma * self.X_hat + self.beta # (B, N, D)

        return out

    def backward(self, d_out):
        """
        Performs the backward pass of the Add and Norm layer.

        Args:
            d_out: The gradient of the loss with respect to the output of this layer.
                   Shape: (batch_size, seq_len, d_model)

        Returns:
            A tuple containing:
                - d_X: Gradient with respect to the input X (for the residual connection).
                       Shape: (batch_size, seq_len, d_model)
                - d_sublayer: Gradient with respect to the sublayer output.
                              Shape: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = self.Z.shape

        # ---------- Gradients for gamma and beta ----------
        d_gamma = np.sum(d_out * self.X_hat, axis=(0, 1)) # (D,)
        d_beta = np.sum(d_out, axis=(0, 1)) # (D,)

        # ---------- Gradients for LayerNorm ----------
        d_X_hat = d_out * self.gamma # (B, N, D)

        std_inv = 1.0 / np.sqrt(self.var + self.eps) # (B, N, 1)
        d_var = np.sum(d_X_hat * (self.Z - self.mean) * -0.5 * (std_inv**3), axis=-1, keepdims=True) # (B, N, 1)

        d_mean = np.sum(d_X_hat * -std_inv, axis=-1, keepdims=True) + d_var * np.mean(-2.0 * (self.Z - self.mean), axis=-1, keepdims=True) # (B, N, 1)

        d_Z = d_X_hat * std_inv + d_var * 2.0 * (self.Z - self.mean) / d_model + d_mean / d_model # (B, N, D)

        # ---------- Split gradient for residual connection ----------
        d_X = d_Z # Gradient for the residual input X
        d_sublayer = d_Z # Gradient for the sublayer output

        # ---------- Update parameters (gamma and beta) ----------
        self.gamma = self.gamma - self.learning_rate * d_gamma
        self.beta = self.beta - self.learning_rate * d_beta

        return d_X, d_sublayer


class multi_head_attention:
    def __init__(self, W_Q, W_K, W_V, W_O, n_head, learning_rate=0.1, mask=None):
        """
        Initializes the Multi-Head Attention layer.

        Args:
            W_Q, W_K, W_V, W_O: Weight matrices for Query, Key, Value, and Output projections.
                                Shapes: (d_model, d_model)
            n_head (int): The number of attention heads.
            learning_rate (float): The learning rate for weight updates.
            mask: Optional mask to prevent attention to certain positions (e.g., for decoder).
                  Shape: (batch_size, 1, seq_len, seq_len) or similar broadcastable shape.
        """
        self.X = None # Input tensor: (batch_size, seq_len, d_model)

        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V
        self.W_O = W_O

        self.learning_rate = learning_rate
        self.mask = mask

        self.d_model = W_Q.shape[0] # Infer d_model from weight matrices
        self.n_head = n_head
        self.d_head = self.d_model // self.n_head # Dimension of each head

        # Cache for backward pass
        self.Q = None # (batch_size, n_head, seq_len, d_head)
        self.K = None # (batch_size, n_head, seq_len, d_head)
        self.V = None # (batch_size, n_head, seq_len, d_head)
        self.scores = None # Raw attention scores before softmax
        self.attn = None # Attention weights: (batch_size, n_head, seq_len, seq_len)
        self.out_heads = None # Output from individual heads before concatenation: (batch_size, n_head, seq_len, d_head)
        self.out_cat = None # Concatenated output from heads: (batch_size, seq_len, d_model)

    def get_input(self, X):
        """Stores the input X for the forward and backward passes."""
        self.X = X
        # Re-infer shapes if X has different dimensions or if d_model wasn't set
        if self.d_model is None or self.d_model != X.shape[-1]:
             self.d_model = X.shape[-1]
             self.d_head = self.d_model // self.n_head
             # Note: If d_model changes, weight matrices would need to be re-initialized or handled

    def forward(self):
        """
        Performs the forward pass for Multi-Head Attention.

        Returns:
            The output of the Multi-Head Attention layer.
            Shape: (batch_size, seq_len, d_model)
        """
        if self.X is None:
            raise ValueError("Input X not set. Call get_input(X) first.")

        batch_size, seq_len, d_model = self.X.shape
        n_head, d_head = self.n_head, self.d_head

        # 1. Linear Projections for Q, K, V
        Q = self.X @ self.W_Q
        K = self.X @ self.W_K
        V = self.X @ self.W_V

        # 2. Reshape and Transpose for Multi-Head
        Q = Q.reshape(batch_size, seq_len, n_head, d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, n_head, d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, n_head, d_head).transpose(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention Scores
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_head) # (B, H, N, N)

        # Apply mask if provided
        if self.mask is not None:
            # Mask is typically 0 where attention should be masked, 1 otherwise.
            # We want to set scores to a large negative number where mask is 0.
            # Ensure mask is broadcastable to scores shape.
            scores = np.where(self.mask == 0, -1e9, scores)

        # 4. Attention Weights (Softmax)
        attn = softmax(scores, axis=-1) # (B, H, N, N)

        # 5. Weighted Sum of Values
        out_heads = attn @ V # (B, H, N, d)

        # 6. Concatenate Heads and Final Linear Projection
        out_cat = out_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        out = out_cat @ self.W_O # (B, N, D)

        # Save cache for backward pass
        self.Q = Q
        self.K = K
        self.V = V
        self.scores = scores
        self.attn = attn
        self.out_heads = out_heads
        self.out_cat = out_cat

        return out

    def backward(self, d_out):
        """
        Performs the backward pass for Multi-Head Attention.

        Args:
            d_out: Gradient of the loss with respect to the output of this layer.
                   Shape: (batch_size, seq_len, d_model)

        Returns:
            d_X: Gradient with respect to the input X.
                 Shape: (batch_size, seq_len, d_model)
        """
        if self.X is None or self.attn is None:
             raise ValueError("Forward pass must be completed before backward pass.")

        batch_size, seq_len, d_model = self.X.shape
        n_head, d_head = self.n_head, self.d_head
        scale = np.sqrt(d_head)

        # 1. Gradient for W_O
        d_out_cat = d_out @ self.W_O.T
        dW_O = self.out_cat.reshape(batch_size * seq_len, d_model).T @ d_out.reshape(batch_size * seq_len, d_model)
        d_out_heads = d_out_cat.reshape(batch_size, seq_len, n_head, d_head).transpose(0, 2, 1, 3)

        # 2. Gradients for Attention Weights (attn) and Values (V)
        d_V = self.attn.transpose(0, 1, 3, 2) @ d_out_heads
        d_Scores_from_attn = softmax_backward(self.attn, d_out_heads @ self.V.transpose(0, 1, 3, 2))

        # 3. Gradients for Q and K
        d_Q = (d_Scores_from_attn @ self.K.transpose(0, 1, 3, 2)) / scale
        d_K = (d_Scores_from_attn.transpose(0, 1, 3, 2) @ self.Q) / scale

        # 4. Reshape Gradients for Q, K, V back to (B*N, d_head)
        d_Q_reshaped = d_Q.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, d_head)
        d_K_reshaped = d_K.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, d_head)
        d_V_reshaped = d_V.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, d_head)

        # 5. Gradients for W_Q, W_K, W_V
        X_flat = self.X.reshape(batch_size * seq_len, d_model)
        dW_Q = X_flat.T @ d_Q_reshaped
        dW_K = X_flat.T @ d_K_reshaped
        dW_V = X_flat.T @ d_V_reshaped

        # 6. Gradient w.r.t. Input X
        d_X = (
            d_Q_reshaped @ self.W_Q.T +
            d_K_reshaped @ self.W_K.T +
            d_V_reshaped @ self.W_V.T
        )
        d_X = d_X.reshape(batch_size, seq_len, d_model)

        # 7. Update Weight Matrices
        self.W_Q = self.W_Q - self.learning_rate * dW_Q
        self.W_K = self.W_K - self.learning_rate * dW_K
        self.W_V = self.W_V - self.learning_rate * dW_V
        self.W_O = self.W_O - self.learning_rate * dW_O

        return d_X


class feed_forward:
    def __init__(self, W_1, b_1, W_2, b_2, learning_rate=0.1):
        """
        Initializes the Feed-Forward Network.

        Args:
            W_1, b_1: Weights and bias for the first linear layer.
            W_2, b_2: Weights and bias for the second linear layer.
            learning_rate (float): Learning rate for weight updates.
        """
        self.X = None
        self.W_1 = W_1
        self.b_1 = b_1
        self.W_2 = W_2
        self.b_2 = b_2

