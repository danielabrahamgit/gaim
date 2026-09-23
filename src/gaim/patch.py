import torch
import torch.nn.functional as F
from operator import index


def _spatial_steps(stride, ndim):
    try:
        steps = (index(stride),) * ndim
    except TypeError:
        steps = tuple(index(step) for step in stride)
    if len(steps) != ndim or any(step <= 0 for step in steps):
        raise ValueError('stride must have one positive integer per spatial axis')
    return steps

def _pad_spatial(img: torch.Tensor,
                 patch_size: tuple,
                 mode: str = 'reflect') -> torch.Tensor:
    pads = []
    for size in reversed(patch_size):
        pads.extend([(size - 1) // 2, size // 2])
    ndim = len(patch_size)
    batch_shape = img.shape[:-ndim]
    # Nonconstant torch padding expects a batch/channel dimension. Flattening
    # leading dimensions also supports unbatched images and arbitrary batches.
    img = img.reshape(-1, 1, *img.shape[-ndim:])
    if torch.is_complex(img) and mode != 'constant':
        padded = torch.complex(
            F.pad(img.real, pads, mode=mode),
            F.pad(img.imag, pads, mode=mode),
        )
    else:
        padded = F.pad(img, pads, mode=mode)
    return padded.reshape(*batch_shape, *padded.shape[-ndim:])

def strided_patchify(img: torch.Tensor,
                     patch_size: tuple,
                     stride=1,
                     mode: str = 'reflect') -> torch.Tensor:
    """Extract padded patches on the grid 0, stride, 2*stride, ... per axis.

    Returns shape (..., *grid_size, *patch_size), with
    grid_size[i] = ceil(im_size[i] / stride[i]). Padding and patch alignment
    are identical for every stride, including even patch sizes: the anchor
    pixel is at index (patch_size[i]-1)//2 inside its patch. Border anchors
    are retained even when image dimensions are not divisible by the stride.

    Use the same strided image coordinates for fitting patch-level quantities;
    use interpolate_patch_grid to display them on the original image grid.
    """
    patch_size = tuple(index(size) for size in patch_size)
    ndim = len(patch_size)
    if not 0 < ndim <= img.ndim or any(size <= 0 for size in patch_size):
        raise ValueError('patch_size must contain positive sizes for trailing image axes')
    steps = _spatial_steps(stride, ndim)
    patches = _pad_spatial(img, patch_size, mode=mode)
    for size, step in zip(patch_size, steps):
        patches = patches.unfold(-ndim, size, step)
    return patches.contiguous()

def interpolate_patch_grid(values: torch.Tensor,
                           im_size: tuple,
                           stride=1) -> torch.Tensor:
    """Multilinearly interpolate patch-grid values at their pixel coordinates.

    Leading dimensions are preserved. Samples lie at 0, stride, 2*stride,
    ...; values beyond the last sample use border extension. Unlike resizing
    to im_size, this preserves sample positions for nondivisible dimensions.
    Interpolate score curves before argmax, rather than integer class labels.
    """
    im_size = tuple(index(size) for size in im_size)
    ndim = len(im_size)
    if not 0 < ndim <= values.ndim or any(size <= 0 for size in im_size):
        raise ValueError('im_size must contain positive sizes for trailing spatial axes')
    steps = _spatial_steps(stride, ndim)
    expected = tuple((size - 1) // step + 1 for size, step in zip(im_size, steps))
    if values.shape[-ndim:] != expected:
        raise ValueError(f'patch grid has shape {values.shape[-ndim:]}, expected {expected}')
    if not (values.is_floating_point() or values.is_complex()):
        raise TypeError('interpolate floating-point scores, not integer labels')
    result = values
    for axis, (size, step) in enumerate(zip(im_size, steps)):
        if step == 1:
            continue
        dim = result.ndim - ndim + axis
        positions = torch.arange(size, device=values.device)
        lower = (positions // step).clamp_max(result.shape[dim] - 1)
        upper = (lower + 1).clamp_max(result.shape[dim] - 1)
        shape = [1] * result.ndim
        shape[dim] = size
        weight = ((positions % step).to(values.real.dtype) / step).reshape(shape)
        result = (1 - weight) * result.index_select(dim, lower) + weight * result.index_select(dim, upper)
    return result
