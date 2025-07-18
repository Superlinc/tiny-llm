import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """
    query: N x H x L x D
    key: N x H x L x D
    value: N x H x L x D
    mask: L x L
    """
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    scores = mx.matmul(query, key.swapaxes(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        """
        E is hidden_size or embed_dim or dims or model_dim
        H is num_heads
        D is head_dim
        L is seq_len, in PyTorch API it's S (source len)

        w_q/w_k/w_v: E x (H x D)
        output/input: N x L x E
        w_o: (H x D) x E
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads

        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo


    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        query: N x L x E
        key: N x L x E
        value: N x L x E
        mask: L x L
        """
        N, L, _ = query.shape
        q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        x = scaled_dot_product_attention_simple(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )
        x = x.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_size)
        return linear(x, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
