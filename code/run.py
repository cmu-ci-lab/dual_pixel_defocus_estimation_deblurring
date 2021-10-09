import jax.numpy as np
import numpy as onp
import setup
import optim
import util
from scipy.special import logit
import os
import cv2


def main(input_params, optim_params):
  """main function for jointly estimating an all-in-focus image and a defocus map from input dual-pixel (DP) image

  Args:
    input_params: a dictionary of input DP info (directory & filename)
    optim_params: a dictionary of optimization parameters

  Returns:
    None. (Save the two outputs to a folder)
  """

  num_mpi_layers = optim_params['num_mpi_layers']
  mpi_start_scale = optim_params['mpi_start_scale']
  mpi_end_scale = optim_params['mpi_end_scale']

  # Load input DP data & calibrated blur kernels
  patch_params = dict(patch_size=168, num_rows=6, num_cols=8)
  observations, blur_kernels = util.load_data_and_calibration(input_params['data_dir'], input_params['dp_file'], patch_params)
  # Normalize input DP data
  observations_max = np.max(observations)
  observations /= observations_max

  # Compute scaled blur kernels
  scales = np.linspace(mpi_end_scale, mpi_start_scale, num=num_mpi_layers, endpoint=True)
  optim_params['scales'] = scales
  filter_halfwidth = blur_kernels.shape[-1] // 2
  blur_kernels_scaled = util.rescale_blur_kernels(blur_kernels, filter_halfwidth*2+1, scales)

  precomputed_vars = \
    dict(observations = observations,
         observations_volume = np.repeat(observations[None, ...], repeats=num_mpi_layers, axis=0),
         filter_halfwidth = filter_halfwidth,
         blur_kernels_scaled = blur_kernels_scaled,
         bias_correction = util.compute_bias_correction(observations, blur_kernels_scaled) )


  # Initialize color channels of all MPI layers to be the mean of the input DP images,
  #            alpha channels to be [1, 1/2, 1/3, ..., 1/num_of_mpi_layers] s.t. all layers have the same transmittance
  mpi_colors = onp.ones((num_mpi_layers, *observations.shape[1:])) * np.mean(observations)
  mpi_alphas = onp.ones((*mpi_colors.shape[:-1], 1)) / onp.arange(1., num_mpi_layers + 1., 1.)[..., None, None, None]
  mpi_init = onp.concatenate([mpi_colors, mpi_alphas], axis=-1)

  print(' ---> Start optimization ...')
  outputs = optim.optimization([logit(mpi_init)], precomputed_vars, optim_params, patch_params)


  print(' ---> Save results ...')
  out_dir_parent = '../results'
  os.makedirs(os.path.join(out_dir_parent, 'all_in_focus_im'), exist_ok=True)
  os.makedirs(os.path.join(out_dir_parent, 'defocus_map'), exist_ok=True)

  intensity_scale_factor = 0.5 / np.mean(observations)
  sharp_im = observations_max / intensity_scale_factor * \
             util.filter_bilateral(intensity_scale_factor * outputs['sharp_im'][..., 0], sigma_s=3.0, sigma_v=0.05)
  defocus_map = outputs['defocus_map']
  cv2.imwrite(os.path.join(out_dir_parent, 'all_in_focus_im', input_params['dp_file'] + f'_sharp_im.png'),
              util.save_16_bit_figure(sharp_im))
  cv2.imwrite(os.path.join(out_dir_parent, 'defocus_map', input_params['dp_file'] + f'_depth.png'),
              util.save_8_bit_figure(util.normalize_0_to_1(defocus_map)))

  return


if __name__ == '__main__':

  num_of_files, _, _ = setup.set_up_parameters()

  for i in range(num_of_files):
    print(f' ===> Data {i+1:03d} ...')

    _, input_params, optim_params = setup.set_up_parameters(i_file=i)
    main(input_params, optim_params)