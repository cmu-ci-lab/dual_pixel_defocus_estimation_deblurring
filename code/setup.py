import jax.numpy as np


def set_up_parameters(i_file=0):
  """The following parameters are for reproducing results in the paper:

  Shumian Xin, Neal Wadhwa, Tianfan Xue, Jonathan T. Barron, Pratul P. Srinivasan, Jianwen Chen, Ioannis Gkioulekas, and Rahul Garg.
  "Defocus Map Estimation and Deblurring from a Single Dual-Pixel Image", ICCV 2021.

  Args:
    i_file: DP file index

  Returns:
    num_of_files: total number of files in the data directory
    input_params: a dictionary of input DP info (directory & filename)
    optim_params: a dictionary of optimization parameters
  """

  data_dir = '../DP_data_pixel_4'
  num_of_files = 17

  # Input parameters
  input_params = \
    dict(data_dir = data_dir,
         dp_file=f'{i_file + 1:03d}')

  # Optimization parameters
  num_mpi_layers = 12
  mpi_start_scales = [0.5, 0.5, 0.7, 0.7, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.1, 0.3, 0.2]
  mpi_end_scales = [1.6, 1.6, 1.6, 1.6, 1.3, 1.4, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.6, 1.5, 1.6, 1.5, 1.6]
  weights_prior_sharp_im_tv = \
    [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
  weights_prior_alpha_tv = \
    [1.5e4, 1.5e4, 1.5e4, 1.5e4, 7.5e4, 7.5e4, 7.5e4, 7.5e4, 7.5e4, 7.5e4, 7.5e4, 7.5e4, 7.5e4, 7.5e4, 7.5e4, 7.5e4, 7.5e4]
  weights_prior_entropy = \
    [20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 20.0, 20.0, 20.0, 20.0, 20.0]

  optim_params = \
    dict(init_lr = 3e-1,
         final_lr = 1e-1,
         num_iters = 10000,
         weight_loss_data = 2.5e4,
         weight_loss_aux_data = 2.5e4,
         weight_prior_sharp_im_tv = weights_prior_sharp_im_tv[i_file],
         weight_prior_alpha_tv = weights_prior_alpha_tv[i_file] * ((num_mpi_layers / 12) ** 2),
         weight_prior_entropy = weights_prior_entropy[i_file] / (-np.log(np.sum((np.ones((num_mpi_layers,)) / num_mpi_layers) ** 2))),
         num_mpi_layers = num_mpi_layers,
         mpi_start_scale = mpi_start_scales[i_file],
         mpi_end_scale = mpi_end_scales[i_file])

  return num_of_files, input_params, optim_params