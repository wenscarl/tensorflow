/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/cuda/cuda_tsr.h"

#include <functional>
#include <memory>
#include <utility>

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"


namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuDnnPlugin);

namespace {

static_assert(CUTENSOR_VERSION >= 1201, "cuTensor needs to be version 1201 or higher");

// Exits the program if 'expr' doesn't return CUDNN_STATUS_SUCCESS.
#define CHECK_CUTENSOR_OK(expr) CHECK_EQ(expr, CUTENSOR_STATUS_SUCCESS)

// If 'expr' doesn't return CUDNN_STATUS_SUCCESS, returns from the current
// function with a non-successful port::Status.
#define RETURN_IF_CUTENSOR_ERROR(expr)                                      \
  do {                                                                   \
    cutensorStatus_t _status = expr;                                        \
    if (!SE_PREDICT_TRUE(_status == CUTENSOR_STATUS_SUCCESS)) {             \
      std::ostringstream oss;                                            \
      oss << ToString(_status) << "\nin " << __FILE__ << "(" << __LINE__ \
          << "): '" << #expr << "'";                                     \
      return port::Status(port::error::UNKNOWN, oss.str().c_str());      \
    }                                                                    \
  } while (false)

std::string ToString(cutensorStatus_t status) {
  switch (status) {
    case CUTENSOR_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUTENSOR_STATUS_INVALID_VALUE:
      return "CUTENSOR_STATUS_INVALID_VALUE";
    case CUTENSOR_STATUS_IO_ERROR:
      return "CUTENSOR_STATUS_IO_ERROR";
    case CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE:
      return "CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
    case CUTENSOR_STATUS_NOT_SUPPORTED:
      return "CUTENSOR_STATUS_NOT_SUPPORTED";
    default:
      return absl::StrCat("<unknown cutensor status: ", static_cast<int>(status),
                          ">");
  }
}

} // namespace

port::Status CUDATsrHandle::Initialize(
    GpuExecutor *parent, const std::string equation,
    const std::vector<int64> &A_shape,
    const std::vector<int64> &B_shape) {
  if (isInitialized()) {
    LOG(FATAL) << "Try to repeatedly initialized.";
  }
  is_initialized_ = true;
  cuda::ScopedActivateExecutorContext sac(parent);

  auto ret = cutensorInit(&cutensor_handle_);
  if (ret != CUTENSOR_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create cuTensor handle:" << ret;
    return port::Status(port::error::INTERNAL,
                        "Failed to create cuTensor handle.");
  }
  parent_ = parent;
  cutensor_einsum_ = new Einsum<double, int64, 12>(
      equation, A_shape, B_shape);

  return port::Status::OK();
} // end Initialized

CUDATsrHandle::~CUDATsrHandle() {
  cuda::ScopedActivateExecutorContext sac(parent_);
}

std::unique_ptr<tsr::Handle> CUDATsr::CreateHandle(Stream *stream,
    const std::string equation, const std::vector<int64> &A_shape,
    const std::vector<int64> &B_shape) {
  std::unique_ptr<CUDATsrHandle> tsr_handle_ptr{new CUDATsrHandle()};
  port::Status status = tsr_handle_ptr->Initialize(parent_, equation, A_shape,
                                                   B_shape);
  // TODO(yangzihao): In the future, send error msg back to TensorFlow
  // so it can fail gracefully,
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize cutensor handle: "
               << status.error_message();
  }
  return std::move(tsr_handle_ptr);
}

// template <typename dummy>
// cutensorHandle_t CUDATsr::DoTsrInternalGetHandle(Stream *stream, tsr::Handle *handle, dummy aa) {
//   aa;
//   CUDATsrHandle *cuda_tsr_handle = dynamic_cast<CUDATsrHandle *>(handle);
//   return cuda_tsr_handle->GetHandle();
// }


bool CUDATsr::DoCuTensorContractionInternal(Stream *stream, tsr::Handle *handle,
                                            const void* A_raw,
                                            const void* B_raw, void* C_raw,
                                            void *work_raw) {
  CUDATsrHandle *cuda_tsr_handle = dynamic_cast<CUDATsrHandle *>(handle);
  cutensorHandle_t cutensor_handle = cuda_tsr_handle->GetHandle();
  if (cuda_tsr_handle == nullptr) {
    LOG(ERROR) << "the passed-in plan is not a CUDATsrHandle object.";
    return false;
  }

  cuda::ScopedActivateExecutorContext sac(parent_);
  auto ret = cuda_tsr_handle->GetCutensorEinsum()->execute(
      &cutensor_handle, A_raw, B_raw, C_raw, work_raw, 0);

  if (ret != CUTENSOR_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to run cuTensor routine: " << ret;
    return false;
  }

  return true;
}

#define STREAM_EXECUTOR_CUDA_DEFINE_TSR(__type)                                \
bool CUDATsr::DoTsrContraction(Stream *stream, tsr::Handle *handle,     \
                                   const void* A_raw,                          \
                                   const void* B_raw, void* C_raw,             \
                                   void *work_raw) {                           \
  return DoCuTensorContractionInternal(                                \
      stream, handle, A_raw, B_raw, C_raw, work_raw);                          \
}

STREAM_EXECUTOR_CUDA_DEFINE_TSR(double)
// STREAM_EXECUTOR_CUDA_DEFINE_TSR(float)
// STREAM_EXECUTOR_CUDA_DEFINE_TSR(Eigen::half)
// STREAM_EXECUTOR_CUDA_DEFINE_TSR(complex64)
// STREAM_EXECUTOR_CUDA_DEFINE_TSR(complex128)
#undef STREAM_EXECUTOR_CUDA_DEFINE_TSR

} // namespace gpu

void initialize_cutsr() {
  port::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::TsrFactory>(
          cuda::kCudaPlatformId, gpu::kCuTsrPlugin, "cuTSR",
          [](internal::StreamExecutorInterface *parent) -> tsr::TsrSupport * {
            gpu::GpuExecutor *cuda_executor =
                dynamic_cast<gpu::GpuExecutor *>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR) << "Attempting to initialize an instance of the cuTsr "
                         << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            return new gpu::CUDATsr(cuda_executor);
          });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuTSR factory: "
               << status.error_message();
  }

  PluginRegistry::Instance()->SetDefaultFactory(
      cuda::kCudaPlatformId, PluginKind::kTsr, gpu::kCuTsrPlugin);
}

} // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_cutsr,
                            { stream_executor::initialize_cutsr(); });
