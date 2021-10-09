""" Utility functions
"""

import jax.numpy as np
from jax import vmap, scipy
import numpy as onp
import scipy as oscipy
import PIL.Image
import os
import flax


def load_data_and_calibration(data_dir, dp_file, patch_params):
  """Load Google Pixel 4 DP data and calibration

  Args:
    data_dir: data directory
    dp_file: dual-pixel filename
    patch_params: a dictionary containing image patch parameters

  Returns:
    left and right dp images: [2, H, W, C]
    calibrated blur kernels: [#rows * #cols, F, F], F: blur kernel size
  """

  def _load_and_preprocess_pixel_data(path_to_file):
    # first deduct black level (1024 for 14-bit Google Pixel 4 DP data), then normalize to [0, 1]
    with PIL.Image.open(path_to_file) as f:
      image = onp.array(f) - 1024
      image[image < 0] = 0
      image = np.stack([np.float32(image)] * 1, axis=2) / (2 ** 14 - 1)
    return image
  left_image = _load_and_preprocess_pixel_data(os.path.join(data_dir, f'{dp_file}_left.png'))
  right_image = _load_and_preprocess_pixel_data(os.path.join(data_dir, f'{dp_file}_right.png'))


  calibration_dir = 'calibration'
  # Vignetting correction
  left_white_calib = _load_and_preprocess_pixel_data(os.path.join(data_dir, calibration_dir, f'white_sheet_left.png'))
  right_white_calib = _load_and_preprocess_pixel_data(os.path.join(data_dir, calibration_dir, f'white_sheet_right.png'))
  per_pixel_scale = left_white_calib / right_white_calib
  window_size = 101
  per_pixel_scale_avg = flax.nn.avg_pool(per_pixel_scale[None], (window_size, window_size), strides=(1, 1), padding='SAME')[0]
  right_image = per_pixel_scale_avg * right_image

  # Keep only central field of view (1008 * 1344)
  def _crop_image_central_fov(images, patch_size, num_rows, num_cols):
    """ Crop images

    Args:
      images: [..., H, W, C] #images, height, width, #channels.

    Returns:
      [..., #rows * P, #cols * P, C] cropped images
    """

    crop_y = patch_size * num_rows
    crop_x = patch_size * num_cols
    offset_y = (images.shape[-3] - crop_y) // 2
    offset_x = (images.shape[-2] - crop_x) // 2

    return images[..., offset_y:offset_y + crop_y, offset_x:offset_x + crop_x, :]
  observations = np.stack([left_image, right_image], axis=0)
  observations = _crop_image_central_fov(observations, **patch_params)

  # Calibrated blur kernels
  blur_kernels_left = np.load(os.path.join(data_dir, calibration_dir, f'blur_kernels_left.npy'))
  blur_kernels_right = np.load(os.path.join(data_dir, calibration_dir, f'blur_kernels_right.npy'))

  return observations, np.stack([blur_kernels_left, blur_kernels_right], axis=1)


def compute_bias_correction(observations, blur_kernels_scaled):
  """ Computer bias correction term

  Args:
    observations: [#observations, H, W, C] height, width, num_color_channels
    blur_kernels_scaled: [L, #rows*#cols, #observations, F, F] scaled blur kernels, F: blur kernel size, assumed to be odd

  Returns:
    bias correction term [L, ]
  """

  def _my_ft2(x, axes=(-2, -1)):
    """2D Fourier Transform with fftshift

    Args:
      x: [H, W] spatial domain

    Returns:
      Resulting fourier transform
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)

  blur_kernels_FT = _my_ft2(blur_kernels_scaled, axes=(-2, -1))
  C = 5e-3
  gaussian_noise_stv = 5e-3 ** 2 * 2
  K = np.sum(np.abs(blur_kernels_FT) ** 2, axis=-3)
  bias_correction = np.mean(observations) / 0.5 * gaussian_noise_stv * np.mean(C / (K + C), axis=(-3, -2, -1))

  return bias_correction


def collision_entropy(input, axis=0):
  """ Compute collision entropy of a tensor along given axis

  Args:
    input: a tensor
    axis

  Returns:
    resulting collision entropy
  """
  input /= np.sum(input, axis=axis)
  entropy = -np.log(np.sum(input ** 2, axis=axis))
  # numerical stability
  entropy = np.nan_to_num(entropy, nan=-onp.log(1 / input.shape[axis]))

  return entropy


def filter_image_2d(im, f, mode):
  """ 2D image convolution

  Args:
    im: [H, W, C]
    f: [F, F]
    mode: convolve2d mode

  Returns:
    [H_new, W_new, C]
  """
  return vmap(scipy.signal.convolve2d, in_axes=(-1, None, None), out_axes=-1)(im, f, mode)


def filter_image_batch(ims, fs, mode):
  """ 2D image convolution, batch processing

  Args:
    im: [..., H, W, C]
    f: [..., F, F]
    mode: convolve2d mode

  Returns:
    [..., H_new, W_new, C]
  """
  ims_reshape = np.reshape(ims, (-1, *ims.shape[-3:]))
  fs_reshape = np.reshape(fs, (-1, *fs.shape[-2:]))
  filtered_ims = vmap(filter_image_2d, in_axes=(0, 0, None), out_axes=0)(ims_reshape, fs_reshape, mode)

  return np.reshape(filtered_ims, (*ims.shape[:-3], *filtered_ims.shape[-3:]))


def filter_bilateral(img_in, sigma_s, sigma_v, reg_constant=1e-8):
  """Simple bilateral filtering of an input image
  Code reference:
  http://jamesgregson.ca/bilateral-filtering-in-python.html

  Performs standard bilateral filtering of an input image. If padding is desired,
  img_in should be padded prior to calling

  Args:
    img_in       (ndarray) monochrome input image
    sigma_s      (float)   spatial gaussian std. dev.
    sigma_v      (float)   value gaussian std. dev.
    reg_constant (float)   optional regularization constant for pathalogical cases

  Returns:
    result       (ndarray) output bilateral-filtered image

  Raises:
    ValueError whenever img_in is not a 2D float32 valued numpy.ndarray
  """

  # check the input
  if not isinstance(img_in, np.ndarray) or img_in.dtype != 'float32' or img_in.ndim != 2:
    raise ValueError('Expected a 2D numpy.ndarray with float32 elements')

  # make a simple Gaussian function taking the squared radius
  gaussian = lambda r2, sigma: (np.exp(-0.5 * r2 / sigma ** 2) * 3).astype(np.int32) * 1.0 / 3.0

  # define the window width to be the 3 time the spatial std. dev. to
  # be sure that most of the spatial kernel is actually captured
  win_width = np.int32(3 * sigma_s + 1)

  # initialize the results and sum of weights to very small values for
  # numerical stability. not strictly necessary but helpful to avoid
  # wild values with pathological choices of parameters
  wgt_sum = np.ones(img_in.shape) * reg_constant
  result = img_in * reg_constant

  # accumulate the result by circularly shifting the image across the
  # window in the horizontal and vertical directions. within the inner
  # loop, calculate the two weights and accumulate the weight sum and
  # the unnormalized result image
  for shft_x in range(-win_width, win_width + 1):
    for shft_y in range(-win_width, win_width + 1):
      # compute the spatial weight
      w = gaussian(shft_x ** 2 + shft_y ** 2, sigma_s)

      # shift by the offsets
      off = np.roll(img_in, [shft_y, shft_x], axis=[0, 1])

      # compute the value weight
      tw = w * gaussian((off - img_in) ** 2, sigma_v)

      # accumulate the results
      result += off * tw
      wgt_sum += tw

  # normalize the result and return
  return result / wgt_sum


def charbonnier_loss_from_L2_loss(x_square, gamma=1/10):
  """ isotropic total variatio on 2D data (e.g. single-channel images, or defocus maps)

  Args:
    x_square: L2 loss

  Returns:
    charbonnier loss
  """
  return np.sqrt(x_square / (gamma ** 2) + 1) - 1


def isotropic_total_variation(I, gamma):
  """ isotropic total variatio on 2D data (e.g. single-channel images, or defocus maps)

  Args:
    I: [H, W]

  Returns:
    [H, W] per-pixel total variation
  """

  gauss_1d = np.array([1, 2, 1])
  f_tv = gauss_1d[:, None] @ gauss_1d[None, :]
  f_tv = f_tv / np.sum(f_tv)

  I_blur = scipy.signal.convolve2d(I, f_tv, 'same')
  I_sq_blur = scipy.signal.convolve2d(I ** 2, f_tv, 'same')
  isotropic_tv = np.abs(I_sq_blur - I_blur ** 2)

  return charbonnier_loss_from_L2_loss(isotropic_tv, gamma)


def isotropic_total_variation_batch(Is, gamma):
  """ isotropic total variatio on 2D data (e.g. single-layer images, or depth maps)

  Args:
    I: [..., H, W]

  Returns:
    [..., H, W] per-pixel total variation
  """
  Is_reshape = np.reshape(Is, (-1, *Is.shape[-2:]))
  isotropic_tvs = vmap(isotropic_total_variation, in_axes=(0, None), out_axes=0)(Is_reshape, gamma)

  return np.reshape(isotropic_tvs, Is.shape)


def edge_mask_from_image_tv(im_tv, gamma, beta):
  """ edge mask from image total variation

  Args:
    im_tv: charbonnier image total variation
    gamma: gamma value used for converting L2 total variation to charbonnier total variation
    beta: parameter for edge mask

  Returns:
    edge mask
  """

  return 1 - np.exp(-((im_tv + 1) ** 2 - 1) * (gamma ** 2) / (2 * beta ** 2))


def rescale_blur_kernels_one_scale(blur_kernels, blur_kernel_outsize, scale):
  """Given filters of size (F, F), generates output filters of size (output_size, output_size),
  and scaled down by a factor of scale.
  Note that scale = 1 corresponds to the case when the filters are resized to output_size.

  Args:
    blur_kernels: [..., F, F]  multiple input filters of size (F, F), F should be odd
    blur_kernel_outsize: scalar, should be odd
    scale: scalar, when <0, the input is flipped about the center.

  Returns:
    [..., output_size, output_size] resampled filters of size (output_size, output_size)
  """

  if blur_kernel_outsize % 2 != 1:
    raise ValueError(f'output_size={blur_kernel_outsize} should be odd')
  F = blur_kernels.shape[-1]
  if F % 2 != 1:
    raise ValueError(f'Input dimensions should be odd but is {F}')

  flip = True if scale < 0 else False
  scale = abs(scale)

  # Translate these coordinates to the input coordinate space based on scale.
  base_scale = F / blur_kernel_outsize
  scale *= base_scale

  filter_halfsize = blur_kernel_outsize // 2
  output_pixel_centers = np.linspace(-filter_halfsize * scale, filter_halfsize * scale, blur_kernel_outsize)

  input_pixel_centers = np.linspace(-(F // 2), F // 2, F)

  if scale < 2.0:
    input_reshape = np.reshape(blur_kernels, (-1, F, F))

    def _unstack(x, axis=-1):
      """ unstack a numpy array along the input axis

      Args:
        x: a numpy array
        axis

      Returns:
        a list
      """
      return tuple(np.moveaxis(x, axis, 0))

    fs = [oscipy.interpolate.interp2d(input_pixel_centers, input_pixel_centers, inp, kind='linear', bounds_error=False, fill_value=0)
          for inp in _unstack(input_reshape, 0)]
    output = np.stack([f(output_pixel_centers, output_pixel_centers) for f in fs], axis=0)

    # Scale, so that resampled image sums to the same value.
    output = output * np.sum(input_reshape, (-1, -2), keepdims=True) / np.sum(output, (-1, -2), keepdims=True)
    output = np.reshape(output, blur_kernels.shape[:-2] + (blur_kernel_outsize, blur_kernel_outsize))

  else:
    x_out, y_out = np.meshgrid(output_pixel_centers, output_pixel_centers)
    x_out = np.reshape(x_out, (-1))
    y_out = np.reshape(y_out, (-1))

    # Sigma for Gaussian kernel used to filter the input.
    sigma = (scale - 1.0) / 2.0

    def _get_weights_1d(in_coords, out_coords):
      in_coords = np.tile(in_coords[..., None], (1, 1, blur_kernel_outsize * blur_kernel_outsize))
      in_coords = in_coords - out_coords[None, None, :]
      return np.exp(-(in_coords ** 2) / (2 * sigma * sigma))

    x_in, y_in = np.meshgrid(input_pixel_centers, input_pixel_centers)
    gx = _get_weights_1d(x_in, x_out)
    gy = _get_weights_1d(y_in, y_out)
    weights = gx * gy / (2.0 * np.pi * sigma * sigma)
    weights = np.moveaxis(weights, -1, 0)
    weights = np.reshape(weights, (blur_kernel_outsize * blur_kernel_outsize, F * F))

    input_reshape = np.moveaxis(blur_kernels, [-2, -1], [0, 1])  # [F, F, ...]
    input_vec = np.reshape(input_reshape, (F * F, -1))

    output_vec = np.matmul(weights, input_vec)

    # Scale, so that resampled image sums to the same value.
    output_vec = output_vec * np.sum(input_vec, 0) / np.sum(output_vec, 0)

    output = np.reshape(output_vec, (blur_kernel_outsize, blur_kernel_outsize) + (blur_kernels.shape[:-2]))
    output = np.moveaxis(output, [0, 1], [-2, -1])

  if flip:
    output = np.flip(np.flip(output, -1), -2)

  return output


def rescale_blur_kernels(blur_kernels, blur_kernel_outsize, scales):
  """ Rescale blur kernels

  Args:
    blur_kernels: [..., F, F] calibrated filters

  Returns:
    blur_kernels_scaled: [#scales, ..., F, F] calibrated filters
  """
  blur_kernels_scaled = []
  for s in scales:
    filters_one_scale = rescale_blur_kernels_one_scale(blur_kernels, blur_kernel_outsize, 1 / s)
    blur_kernels_scaled.append(filters_one_scale)

  return np.stack(blur_kernels_scaled, axis=0)


def save_16_bit_figure(im):
  return (onp.asarray(im) * (2 ** 16 - 1)).astype(onp.uint16)


def save_8_bit_figure(im):
  return (onp.asarray(im) * (2 ** 8 - 1)).astype(onp.uint8)


def normalize_0_to_1(x):
  if (np.max(x) - np.min(x)) < 1e-10:
    y = 0.5 * np.ones_like(x)
  else:
    y = (x - np.min(x)) / (np.max(x) - np.min(x))
  return y