#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
at::Tensor temporal_deform_attn_forward_cuda(
    const at::Tensor &value,
    const at::Tensor &data_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step
);

at::Tensor temporal_deform_attn_backward_cuda(
    const at::Tensor &value,
    const at::Tensor &data_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step
);

// C++ interface
at::Tensor temporal_deform_attn_forward(
    const at::Tensor &value,
    const at::Tensor &data_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step
)
{
    return temporal_deform_attn_forward_cuda(value, data_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
}

at::Tensor temporal_deform_attn_backward(
    const at::Tensor &value,
    const at::Tensor &data_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step
)
{
    return temporal_deform_attn_backward_cuda(value, data_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &temporal_deform_attn_forward, "TDA forward with CUDA");
  m.def("backward", &temporal_deform_attn_backward, "TDA backward with CUDA");
}