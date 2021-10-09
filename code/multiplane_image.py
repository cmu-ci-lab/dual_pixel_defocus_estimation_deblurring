""" Functions related to multiplane image (MPI) representation,
e.g., composite all-in-focus images & defocus maps, and render defocus-blurred images.

An MPI discretizes a 3D scene into fronto-parallel planar layers at fixed depths.
Each layer is an intensity-alpha image, or RGBA image if it has multiple color channels.

Code reference: https://github.com/google-research/google-research/blob/master/single_view_mpi/libs/mpi.py
"""

import jax.numpy as np
import numpy as onp
import util


def compute_layer_visibility(mpi_alphas):
  """Compute visibility of each MPI layer from alpha channels.
    The visibility of a pixel at i-th MPI layer is the product of (1-alpha) of all the layers in front of it, i.e:
      (1 - alpha_i+1) * (1 - alpha_i+2) * ... (1 - alpha_n-1)
    Args:
      mpi_alphas: [L, ..., H, W, 1] alpha channels for L layers, back to front.

    Returns:
      [L, ..., H, W, 1] layer visibility.
    """
  mpi_visibility = np.cumprod(1.0 - mpi_alphas[::-1, ...], axis=0)
  mpi_visibility = np.concatenate([np.ones_like(mpi_visibility[[0], ...]), mpi_visibility[:-1, ...]], axis=0)[::-1, ...]

  return mpi_visibility


def compute_layer_transmittance(mpi_alphas):
  """Compute transimittance of each MPI layer from alpha channels.
  The transmittance of a pixel at i-th MPI layer is the product of its own alpha value and (1-alpha) of all the layers in front of it, i.e:
      alpha_i * (1 - alpha_i+1) * (1 - alpha_i+2) * ... (1 - alpha_n-1)

  Args:
    mpi_alphas: [L, ..., H, W, 1] alpha channels for L layers, back to front.

  Returns:
    [L, ..., H, W, 1] The resulting transmittance.
  """

  return mpi_alphas * compute_layer_visibility(mpi_alphas)


def compose_sharp_image_from_mpi(mpi):
  """ Compose all-in-focus images from MPIs.

  Args:
    mpi: [L, ..., H, W, C+1] num_layers, height, width, num_color_channels+1 (alpha channel)

  Returns:
    sharp_im: [..., H, W, C] sharp / all-in-focus images
  """
  mpi_colors = mpi[..., :-1]
  mpi_alphas = mpi[..., -1:]
  sharp_im = np.sum(compute_layer_transmittance(mpi_alphas) * mpi_colors, axis=0)

  return sharp_im


def compute_defocus_map_from_mpi(mpi, defocus_scales):
  """ Compute defocus maps from MPIs.
  This is done in a similar way as composing sharp images from MPIs.

  Args:
    mpi: [L, H, W, C+1] num_layers, height, width, num_color_channels+1 (alpha channel)
    defocus_scales: a numpy array of blur kernel scales

  Returns:
    [H, W] defocus map
  """

  if mpi.shape[0] != defocus_scales.size:
    raise ValueError(f'MPI should have the same number of layers as kernel scales.')

  mpi_alphas = mpi[..., -1:]
  defocus_map = np.sum(compute_layer_transmittance(mpi_alphas) * defocus_scales[..., None, None, None], axis=0)

  return defocus_map[..., 0]


def extract_patches(images, patch_size, num_rows, num_cols, padding=0):
  """ Divide images into image patches according to patch parameters

  Args:
    images: [..., #rows * P, #cols * P, C] height, width, #channels, P: patch size

  Returns:
    image_patches: [#rows * #cols, ..., P, P, C] The resulting image patches.
  """

  xv, yv = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
  yv *= patch_size
  xv *= patch_size

  patch_size_padding = patch_size + 2 * padding
  xv_size, yv_size = np.meshgrid(np.arange(patch_size_padding), np.arange(patch_size_padding))

  yv_all = yv.reshape(-1)[..., None, None] + yv_size[None, ...]
  xv_all = xv.reshape(-1)[..., None, None] + xv_size[None, ...]
  patches = images[..., yv_all, xv_all, :]
  patches = np.moveaxis(patches, -4, 0)

  return patches


def stitch_patches(patches, patch_size, num_rows, num_cols, stitch_axis):
  """ Stitch patches according to the given dimension

  Args:
    patches: [#rows * #cols, ..., P, P, C] / [#rows * #cols, ..., F, F]
    stitch_axis: (-3, -2) / (-2, -1)

  Returns:
    [..., #rows * P, #cols * P, C]  stitched images / [..., #rows * F, #cols * F] stitched kernels
  """

  axis_row, axis_col = stitch_axis
  patches_reshape = np.reshape(patches, (num_rows, num_cols, *patches.shape[1:]))
  patches_reshape = np.moveaxis(patches_reshape, (0, 1), (axis_row - 2, axis_col - 1))
  new_shape = onp.array(patches.shape[1:])
  new_shape[axis_row] *= num_rows
  new_shape[axis_col] *= num_cols
  images = np.reshape(patches_reshape, new_shape)

  return images


def convolve_mpi_filter(mpi, blur_kernels_scaled, patch_params):
  """ Convolve each layer of an MPI with scaled spatially-varying blur kernels.
  The amount of scale depends on the layer's defocus distance.

  Args:
    mpi: [L, H, W, C+1] num_layers, height, width, num_color_channels+1 (alpha channel)
    blur_kernels_scaled: [L, #rows*#cols, #observations, F, F] scaled blur kernels, F: blur kernel size, assumed to be odd
    patch_params: a dictionary of image patch parameters for extracting image patches

  Returns:
    filtered_mpi:  [L, #observations, H, W, C]
  """

  filter_halfwidth = blur_kernels_scaled.shape[-1] // 2
  num_observations = blur_kernels_scaled.shape[-3]

  mpi_patches = extract_patches(mpi, **patch_params, padding=filter_halfwidth)  # [#rows*#cols, L, H, W, C]
  mpi_patches = np.repeat(mpi_patches[..., None, :, :, :], repeats=num_observations, axis=-4)  # [#rows*#cols, L, #observations, H, W, C]

  filters_rescale_ = np.moveaxis(blur_kernels_scaled, 0, 1)
  filtered_patches = util.filter_image_batch(mpi_patches, filters_rescale_, 'valid')
  filtered_mpi = stitch_patches(filtered_patches, **patch_params, stitch_axis=(-3, -2))  # [L, #observations, H, W, C]

  return filtered_mpi


def render_blurred_image_from_mpi(mpi, blur_kernels_scaled, patch_params):
  """ Render defocus-blurred images by first convolving each layer of an MPI with scaled spatially-varying blur kernels,
  and then blending all filtered layers.

  Args:
    mpi: [L, H, W, C+1] num_layers, height, width, num_image_channels+1 (alpha channel)
    blur_kernels_scaled: [L, #rows*#cols, #observations, F, F] resized filters, F: filter size
    patch_params: a dictionary of image patch parameters, useful for extracting image patches

  Returns:
    blurred_ims: [#observations, H, W, C] defocus-blurred image
    filtered_transmittance:  [L, H, W, C] filtered transmittance
  """

  mpi_colors = mpi[..., :-1]
  mpi_alphas = mpi[..., -1:]
  filtered_mpi = convolve_mpi_filter(np.concatenate([mpi_colors * mpi_alphas, mpi_alphas], axis=-1), blur_kernels_scaled, patch_params)

  filtered_colors = filtered_mpi[..., :-1]
  filtered_alphas = filtered_mpi[..., -1:]
  filtered_visibility = compute_layer_visibility(filtered_alphas)
  blurred_ims = np.sum(filtered_visibility * filtered_colors, axis=0)
  filtered_transmittance = filtered_visibility * filtered_alphas

  return blurred_ims, filtered_transmittance