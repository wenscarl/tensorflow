/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/linalg/einsum_op_impl.h"

namespace tensorflow {


#define REGISTER_EINSUM_KERNELS_CPU(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Einsum").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      EinsumCpuOp<TYPE>);
TF_CALL_complex128(REGISTER_EINSUM_KERNELS_CPU);
#undef REGISTER_EINSUM_KERNELS_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_EINSUM_KERNELS_GPU(TYPE)                             \
 REGISTER_KERNEL_BUILDER(                                         \
     Name("Einsum").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
     EinsumGpuOp<TYPE>);

TF_CALL_complex128(REGISTER_EINSUM_KERNELS_GPU);
#undef REGISTER_EINSUM_KERNELS_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


}  // namespace tensorflow
