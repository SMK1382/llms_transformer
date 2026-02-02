
def softmax_backward(A, dA):
    """
    A  : (N, N) softmax output (row-wise)
    dA : (N, N) upstream gradient
    """
    temp = np.sum(A * dA, axis=-1, keepdims=True)  # (N,1)
    dS = A * (dA - temp)
    return dS


def multihead_attention_backward(
    X,                    # (N, d_model)
    Wq, Wk, Wv, Wo,        # Wq,Wk,Wv: (d_model, d_model), Wo: (d_model,d_model)
    cache,                # forward cache
    dOut,                 # (N, d_model)
    H                     # number of heads
):
    """
    cache contains:
      Q, K, V  : (N, H, d_h)
      A        : (N, H, N)
      O        : (N, H, d_h)
      O_cat    : (N, d_model)
    """

    Q, K, V, A, O, O_cat = cache
    N, d_model = X.shape
    d_h = d_model // H
    scale = np.sqrt(d_h)

    # --------------------------------------------------
    # Grad through final linear projection
    # --------------------------------------------------
    dO_cat = dOut @ Wo.T                 # (N, d_model)
    dWo = O_cat.T @ dOut                 # (d_model, d_model)

    # reshape to heads
    dO = dO_cat.reshape(N, H, d_h)       # (N, H, d_h)

    # --------------------------------------------------
    # Grad through V and A
    # --------------------------------------------------
    dV = np.zeros_like(V)                # (N, H, d_h)
    dA = np.zeros_like(A)                # (N, H, N)

    for h in range(H):
        dV[:, h, :] = A[:, h, :].T @ dO[:, h, :]
        dA[:, h, :] = dO[:, h, :] @ V[:, h, :].T

    # --------------------------------------------------
    # Softmax backward (scores)
    # --------------------------------------------------
    dS = np.zeros_like(A)                # (N, H, N)
    for h in range(H):
        dS[:, h, :] = softmax_backward(A[:, h, :], dA[:, h, :])

    # --------------------------------------------------
    # Scores -> Q, K
    # --------------------------------------------------
    dQ = np.zeros_like(Q)                # (N, H, d_h)
    dK = np.zeros_like(K)                # (N, H, d_h)

    for h in range(H):
        dQ[:, h, :] = (dS[:, h, :] @ K[:, h, :]) / scale
        dK[:, h, :] = (dS[:, h, :].T @ Q[:, h, :]) / scale

    # --------------------------------------------------
    # Projection matrix gradients
    # --------------------------------------------------
    dWq = X.T @ dQ.reshape(N, d_model)
    dWk = X.T @ dK.reshape(N, d_model)
    dWv = X.T @ dV.reshape(N, d_model)

    # --------------------------------------------------
    # Gradient to input X
    # --------------------------------------------------
    dX = (
        dQ.reshape(N, d_model) @ Wq.T +
        dK.reshape(N, d_model) @ Wk.T +
        dV.reshape(N, d_model) @ Wv.T
    )

    return dX, dWq, dWk, dWv, dWo

