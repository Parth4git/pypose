import math
import torch


def vec2skew(input:torch.Tensor) -> torch.Tensor:
    r"""
    Batched Skew Matrices.

    .. math::
        {\displaystyle \mathbf{y}_i={\begin{bmatrix}\,\,
        0&\!-x_{i,3}&\,\,\,x_{i,2}\\\,\,\,x_{i,3}&0&\!-x_{i,1}
        \\\!-x_{i,2}&\,\,x_{i,1}&\,\,0\end{bmatrix}},}

    Args:
        input (Tensor): the tensor :math:`\mathbf{x}` to convert

    Return:
        Tensor: the skew matrices :math:`\mathbf{y}`

    Shape:
        - Input: :code:`(*, 3)`
        - Output: :code:`(*, 3, 3)`

    Note:
        The last dimension of the input tensor has to be 3.

    Example:
        >>> pp.vec2skew(torch.randn(1,3))
        tensor([[[ 0.0000, -2.2059, -1.2761],
                [ 2.2059,  0.0000,  0.2929],
                [ 1.2761, -0.2929,  0.0000]]])
    """
    assert input.shape[-1] == 3, "Last dim should be 3"
    shape, v = input.shape, input.view(-1,3)
    S = torch.zeros(v.shape[:-1]+(3,3), device=v.device, dtype=v.dtype)
    S[:,0,1], S[:,0,2] = -v[:,2],  v[:,1]
    S[:,1,0], S[:,1,2] =  v[:,2], -v[:,0]
    S[:,2,0], S[:,2,1] = -v[:,1],  v[:,0]
    return S.view(shape[:-1]+(3,3))


def cumops_(input, dim, ops):
    r'''
        Inplace version of :meth:`pypose.cumops`
    '''
    L, v = input.shape[dim], input
    assert dim != -1 or dim != v.shape[-1], "Invalid dim"
    for i in torch.pow(2, torch.arange(math.log2(L)+1, device=v.device, dtype=torch.int64)):
        index = torch.arange(i, L, device=v.device, dtype=torch.int64)
        v.index_copy_(dim, index, ops(v.index_select(dim, index), v.index_select(dim, index-i)))
    return v


def cummul_(input, dim):
    r'''
        Inplace version of :meth:`pypose.cummul`
    '''
    return cumops_(input, dim, lambda a, b : a * b)


def cumprod_(input, dim):
    r'''
        Inplace version of :meth:`pypose.cumprod`
    '''
    return cumops_(input, dim, lambda a, b : a @ b)


def cumops(input, dim, ops):
    r"""Returns the cumulative customized operation of LieTensor elements of input in the dimension dim.

    For example, if input is a vector of size N, the result will also be a vector of size N, with elements.

    .. math::
        y_i = x_1~\mathrm{ops}~x_2 ~\mathrm{ops}~ \cdots ~\mathrm{ops}~ x_i

    Args:
        input (LieTensor): the input LieTensor
        dim (int): the dimension to do the operation over
        ops (func): the function to be customized

    Returns:
        LieTensor: LieTensor

    Note:
        - The users are supposed to provide meaningful customized operation.
        - It doesn't check whether the results are valid for mathematical
          definition of LieTensor, e.g., quaternion.

    Examples:
        >>> input = pp.randn_SE3(2)
        >>> input.cumprod(dim = 0)
        SE3Type LieTensor:
        tensor([[-0.6466,  0.2956,  2.4055, -0.4428,  0.1893,  0.3933,  0.7833],
                [ 1.2711,  1.2020,  0.0651, -0.0685,  0.6732,  0.7331, -0.0685]])
        >>> pp.cumops(input, 0, lambda a, b : a @ b)
        SE3Type LieTensor:
        tensor([[-0.6466,  0.2956,  2.4055, -0.4428,  0.1893,  0.3933,  0.7833],
                [ 1.2711,  1.2020,  0.0651, -0.0685,  0.6732,  0.7331, -0.0685]])
    """
    return cumops_(input.clone(), dim, ops)


def cummul(input, dim):
    r"""Returns the cumulative multiplication (*) of LieTensor elements of input in the dimension dim.

    For example, if input is a vector of size N, the result will also be a vector of size N, with elements.

    .. math::
        y_i = x_1 * x_2 * \cdots @ x_i

    Args:
        input (LieTensor): the input tenso
        dim (int): the dimension to do the operation over

    Returns:
        LieTensor: The LieTensor

    Examples:
        >>> input = pp.randn_SE3(2)
        >>> pp.cumprod(input, dim=0)
        SE3Type LieTensor:
        tensor([[-1.9615, -0.1246,  0.3666,  0.0165,  0.2853,  0.3126,  0.9059],
                [ 0.7139,  1.3988, -0.1909, -0.1780,  0.4405, -0.6571,  0.5852]])
    """
    return cumops(input, dim, lambda a, b : a * b)


def cumprod(input, dim):
    r"""Returns the cumulative product (@) of LieTensor elements of input in the dimension dim.

    For example, if input is a vector of size N, the result will also be a vector of size N, with elements.

    .. math::
        y_i = x_1 @ x_2 @ \cdots @ x_i

    Args:
        input (LieTensor): the input tenso
        dim (int): the dimension to do the operation over

    Returns:
        LieTensor: The LieTensor

    Examples:
        >>> input = pp.randn_SE3(2)
        >>> pp.cumprod(input, dim=0)
        SE3Type LieTensor:
        tensor([[-1.9615, -0.1246,  0.3666,  0.0165,  0.2853,  0.3126,  0.9059],
                [ 0.7139,  1.3988, -0.1909, -0.1780,  0.4405, -0.6571,  0.5852]])
    """
    return cumops(input, dim, lambda a, b : a @ b)
