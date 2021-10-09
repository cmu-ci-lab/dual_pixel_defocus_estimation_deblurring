"""MPI Optimization functions
"""

import jax.numpy as np
import flax
from jax import jit, lax
import jax.experimental.optimizers
import functools
import multiplane_image
from util import isotropic_total_variation as tv
from util import isotropic_total_variation_batch as tv_batch
import util


def lossfun_helper(params, precomputed_vars, optim_params, patch_params):
  """Compute total loss of the current MPI

  Args:
    params: a dictionary of MPI
    precomputed_vars: a dictionary of precomputed variables
    optim_params: a dictionary of optimization parameters
    patch_params: a dictionary of image patch parameters

  Returns:
    loss: a scalar
    a dictionary of outputs (all-in-focus image & defocus map)
    a dictionary of each loss term
  """

  observations = precomputed_vars['observations']
  observations_volume = precomputed_vars['observations_volume']
  filter_halfwidth = precomputed_vars['filter_halfwidth']
  blur_kernels_scaled = precomputed_vars['blur_kernels_scaled']
  bias_correction = precomputed_vars['bias_correction']

  scales = optim_params['scales']
  weight_loss_data = optim_params['weight_loss_data']
  weight_loss_aux_data = optim_params['weight_loss_aux_data']
  weight_prior_sharp_im_tv = optim_params['weight_prior_sharp_im_tv']
  weight_prior_alpha_tv = optim_params['weight_prior_alpha_tv']
  weight_prior_entropy = optim_params['weight_prior_entropy']

  intensity_scale_factor = 0.5 / np.mean(observations)

  # ========== Compose outputs from MPI ==========
  # sigmoid function makes sure both MPI colors and alphas are within the range [0, 1]
  mpi = flax.nn.sigmoid(params[0])
  mpi = np.pad(mpi, pad_width=((0,) * 2, (filter_halfwidth,) * 2, (filter_halfwidth,) * 2, (0,) * 2), mode='edge')
  mpi_colors = mpi[..., :-1]
  mpi_alphas = mpi[..., -1:]
  sharp_im = multiplane_image.compose_sharp_image_from_mpi(mpi)
  defocus_map = multiplane_image.compute_defocus_map_from_mpi(mpi, scales)
  mpi_transmittance = multiplane_image.compute_layer_transmittance(mpi_alphas)


  # ========== Bias-corrected data loss ==========
  renderings_blurred, filtered_transmittance = multiplane_image.render_blurred_image_from_mpi(mpi, blur_kernels_scaled, patch_params)
  cost_data_L2 = (renderings_blurred - observations) ** 2 - np.sum(filtered_transmittance * bias_correction[..., None, None, None, None], axis=0)
  gamma = 1 / 10
  loss_data = weight_loss_data * intensity_scale_factor * np.mean(np.mean(util.charbonnier_loss_from_L2_loss(cost_data_L2, gamma), axis=(0, -1)))


  # ========== Auxiliary data loss ==========
  renderings_per_layer = multiplane_image.convolve_mpi_filter(mpi_colors, blur_kernels_scaled, patch_params)
  cost_aux_data_L2 = (renderings_per_layer - observations_volume) ** 2 - bias_correction[..., None, None, None, None]
  cost_aux_data_Charbonnier = lax.stop_gradient(filtered_transmittance) * util.charbonnier_loss_from_L2_loss(cost_aux_data_L2, gamma)
  cost_aux_data_Charbonnier = scales.size * np.mean(cost_aux_data_Charbonnier, axis=(0, 1, -1))
  loss_aux_data = weight_loss_aux_data * intensity_scale_factor * np.mean(cost_aux_data_Charbonnier)


  # ========== Intensity smoothness prior ==========
  beta = 1 / 32
  sharp_im_tv = tv(np.mean(intensity_scale_factor * sharp_im, axis=-1), gamma)[filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth]
  edge_mask = util.edge_mask_from_image_tv(lax.stop_gradient(sharp_im_tv), gamma, beta)
  sharp_im_tv_bilateral = sharp_im_tv * (1 - edge_mask)
  sharp_im_tv_per_layer = np.mean( (scales.size * lax.stop_gradient(mpi_transmittance)[..., 0] * tv_batch(np.mean(intensity_scale_factor * mpi_colors, axis=-1), gamma))
                          [..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth], axis=0)
  edge_mask_per_layer = util.edge_mask_from_image_tv(lax.stop_gradient(sharp_im_tv_per_layer), gamma, beta)
  sharp_im_tv_per_layer_bilateral = sharp_im_tv_per_layer * (1 - edge_mask_per_layer)
  prior_sharp_im_tv = weight_prior_sharp_im_tv * \
                      (np.mean(sharp_im_tv_bilateral + sharp_im_tv_per_layer_bilateral) + np.mean(sharp_im_tv + sharp_im_tv_per_layer))


  # ========== Alpha and Transmittance smoothness prior ==========
  alpha_tv = tv_batch(np.mean(np.sqrt(mpi_alphas), axis=-1), gamma)[..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth] + \
             tv_batch(np.mean(np.sqrt(mpi_transmittance), axis=-1), gamma)[..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth]
  edge_mask_volume = np.repeat(edge_mask[None, ...], repeats=alpha_tv.shape[0], axis=0)
  alpha_tv_bilateral = alpha_tv * (1 - edge_mask_volume)
  prior_alpha_tv = weight_prior_alpha_tv * (np.mean(alpha_tv_bilateral) + np.mean(alpha_tv))


  # ========== Entropy prior ==========
  alpha_entropy = util.collision_entropy(np.sqrt(mpi_alphas[1:, ...]), axis=0)[filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth] + \
                  util.collision_entropy(np.sqrt(mpi_transmittance[0:, ...]), axis=0)[filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth]
  prior_entropy = weight_prior_entropy * np.mean(alpha_entropy)

  # total loss
  loss = loss_data + loss_aux_data + prior_sharp_im_tv + prior_alpha_tv + prior_entropy

  return loss, \
         {'sharp_im': sharp_im[filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth, :],
          'defocus_map': defocus_map[filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth],
          'mpi': mpi}, \
         {'loss_data': loss_data,
          'loss_aux_data': loss_aux_data,
          'prior_sharp_im_tv': prior_sharp_im_tv,
          'prior_alpha_tv': prior_alpha_tv,
          'prior_entropy': prior_entropy}


def optimization(init, precomputed_vars, optim_params, patch_params):
  """MPI optimization

  Args:
    init: initialized MPI
    precomputed_vars: a dictionary of precomputed variables
    optim_params: a dictionary of optimization parameters
    patch_params: a dictionary of image patch parameters

  Returns:
    outputs: a dictionary containing the all-in-focus image and defocus map rendered from optimized MPI,
             as well as the MPI itself
  """

  def _log_decay(i, init_lr, final_lr, num_iters):
    """Log decay for optimization learning rate

    Args:
      i: current iteration
      init_lr: initial learning rate
      final_lr: final learning rate
      num_iters: total number of iterations

    Returns:
      learning rate for current iteration
    """

    t = np.clip(i / num_iters, 0, 1)
    lr = np.exp(np.log(init_lr) * (1 - t) + np.log(final_lr) * t)
    lr = np.clip(lr, np.minimum(init_lr, final_lr), np.maximum(init_lr, final_lr))

    return lr
  lr_fun = functools.partial(_log_decay, init_lr=optim_params['init_lr'], final_lr=optim_params['final_lr'], num_iters=optim_params['num_iters'])
  opt_init, opt_update, opt_get_params = jax.experimental.optimizers.adam(lr_fun)
  opt_state = opt_init(init)

  # One optimization step
  def lossfun(params, precomputed_vars, optim_params, patch_params):
    return lossfun_helper(params, precomputed_vars, optim_params, patch_params)[0]
  @jit
  def _step(i, opt_state):
    loss, grad = jax.value_and_grad(lossfun, argnums=0)(opt_get_params(opt_state), precomputed_vars, optim_params, patch_params)
    grad_no_nan = [np.nan_to_num(grad_this, nan=0.) for grad_this in grad]
    return loss, opt_update(i, grad_no_nan, opt_state)

  # Run for num_iters
  num_iters = optim_params['num_iters']
  for i in range(num_iters + 1):
    loss, opt_state = _step(i, opt_state)
    if i % (np.minimum(num_iters // 5, 500)) == 0:
      print(f'{i:5d} | lr: {lr_fun(i):0.5e} | total loss: {loss:0.5e}')

  # Generate outputs from optimized MPI
  loss_this, outputs, loss_terms_all = lossfun_helper(opt_get_params(opt_state), precomputed_vars, optim_params, patch_params)
  loss_data = loss_terms_all['loss_data']
  loss_aux_data = loss_terms_all['loss_aux_data']
  prior_sharp_im_tv = loss_terms_all['prior_sharp_im_tv']
  prior_alpha_tv = loss_terms_all['prior_alpha_tv']
  prior_entropy = loss_terms_all['prior_entropy']
  print(f'total loss: {loss_this:0.5e} | loss_data: {loss_data:0.5e} | loss_aux_data: {loss_aux_data:0.5e}')
  print(f'prior_sharp_im_tv: {prior_sharp_im_tv:0.5e} | prior_alpha_tv: {prior_alpha_tv:0.5e} | prior_entropy: {prior_entropy:0.5e}')

  return outputs
