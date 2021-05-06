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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/linalg/einsum_op.h"
#include "third_party/gpus/cuda/include/cutensor.h"
#include "tensorflow/core/framework/op_kernel.h"
namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

inline cutensorHandle_t CreateCuTensorHandle() {
  cutensorHandle_t handle;
  cutensorInit(&handle);
  if (getenv("CUTENSOR_CACHE") && atoi(getenv("CUTENSOR_CACHE")) == 1) {
    cutensorPlanCacheline_t* cachelines = new cutensorPlanCacheline_t[32];
    cutensorHandleAttachPlanCachelines(&handle, cachelines, 32);
  }
  return handle;
}

inline cutensorHandle_t* GetCuTensorHandle() {
  static cutensorHandle_t handle = CreateCuTensorHandle();
  return &handle;
}


template <typename T>
struct EinsumCutensorFunctor<GPUDevice, T> {
  static EIGEN_ALWAYS_INLINE Status
   Compute(OpKernelContext* context, const T* input0, const T* input1, T* out, float* works, string equation_,std::vector<int64> input_0_shape, std::vector<int64> input_1_shape) {
    // Grab the input tensor


    constexpr int kMaxNumModes_ = 12; // maximal number of modes supported by cuTENSOR
 //   Einsum<T, int64, kMaxNumModes_> myEinsum(equation_, input_0_shape, input_1_shape);
    //OP_REQUIRES(context, myEinsum.isInitialized(), errors::Internal("cutensor_python: Initialization failed."));
  //  myEinsum.isInitialized();

 //   auto output_dims = myEinsum.getOutputShape();
    // Create an output tensor
 //   Tensor* output_tensor = NULL;
 //   TensorShape output_shape = TensorShape(output_dims);
 //   //OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
 //   context->allocate_output(0, output_shape, &output_tensor);

 //   size_t worksize = myEinsum.getWorksize();
 //   Tensor work_tensor;
 //   int64 work_tensor_size = worksize / sizeof(float);
 //   TensorShape work_shape = { work_tensor_size };
 //   //OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, work_shape, &work_tensor));
 //   context->allocate_temp(DT_FLOAT, work_shape, &work_tensor);

    const GPUDevice& device = context->eigen_device<GPUDevice>();
 //   
//    auto ret = myEinsum.execute(GetCuTensorHandle(),
//                                  input0,
//                                  input1,
//                                  out,
//                                  works,
//                                  device.stream());
 //     OP_REQUIRES(context, ret, errors::Internal("cutensor_python: Launch failed."));


    }
};

}


#define DECLARE_GPU_SPECS_NDIM(T, NDIM)                              \
  template struct functor::StrideFunctor<GPUDevice, T, NDIM>; \
  template struct functor::InflateFunctor<GPUDevice, T, NDIM>; \

#define DECLARE_GPU_SPECS_NODIM(T)                                   \
  template struct functor::EinsumCutensorFunctor<GPUDevice, T>;

#define DECLARE_GPU_SPECS(T)    \
  DECLARE_GPU_SPECS_NDIM(T, 1); \
  DECLARE_GPU_SPECS_NDIM(T, 2); \
  DECLARE_GPU_SPECS_NDIM(T, 3); \
  DECLARE_GPU_SPECS_NDIM(T, 4); \
  DECLARE_GPU_SPECS_NDIM(T, 5); \
  DECLARE_GPU_SPECS_NDIM(T, 6); \
  DECLARE_GPU_SPECS_NODIM(T);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_COMPLEX_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS_NODIM
#undef DECLARE_GPU_SPECS_NDIM
#undef DECLARE_GPU_SPECS



}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
