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
// #include "tensorflow/core/framework/types.h"


namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuTsrPlugin);

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
    GpuExecutor *parent) {
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
  // cutensor_einsum_ = new Einsum<double, int64, kMaxNumModes_>(
  //     equation, A_shape, B_shape);

  return port::Status::OK();
} // end Initialized

CUDATsrHandle::~CUDATsrHandle() {
  cuda::ScopedActivateExecutorContext sac(parent_);
}

std::unique_ptr<tsr::Handle> CUDATsr::CreateHandle(
    Stream *stream, const std::string equation,
    const std::vector<int> &A_shape,
    const std::vector<int> &B_shape = emptyVec) {
  std::unique_ptr<CUDATsrHandle> tsr_handle_ptr{new CUDATsrHandle()};
  port::Status status = tsr_handle_ptr->Initialize(parent_);
  SetTensorContents(equation, A_shape, B_shape);
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

template <typename ComputeType>
bool CUDATsr::DoCuTensorContractionInternal(Stream *stream, tsr::Handle *handle,
                                            ComputeType &alpha, ComputeType &beta,
                                            const void* A_raw,
                                            const void* B_raw, void* C_raw,
                                            void *work_raw) {
  CUDATsrHandle *cuda_tsr_handle = dynamic_cast<CUDATsrHandle *>(handle);
  cutensorHandle_t cutensor_handle = cuda_tsr_handle->GetHandle();
//   cutensorHandle_t cutensor_handle ;
//   cutensorInit(&cutensor_handle);
//  cutensorHandle_t* cutensor_handle_ptr = &cut_h;


   printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
   printf("numModesA_: %d, numModeB_: %d, numModesC_:%d\n", numModesA_, numModesB_, numModesC_);
   std::cout<< numModesA_ <<"  "<< numModesB_<<std::endl;
   if (cuda_tsr_handle == nullptr) {
     LOG(ERROR) << "the passed-in plan is not a CUDATsrHandle object.";
     return false;
   }
//   cuda::ScopedActivateExecutorContext sac(parent_);
//  // auto ret = cuda_tsr_handle->GetCutensorEinsum()->execute(
//  //////     &cutensor_handle, A_raw, B_raw, C_raw, work_raw, 0);
////////////////////////////////////////////////////////////////////////////////////////////////////////
//   if (!isInitialized_) return false;

        cudaDataType_t cudaType = CuTensorTypeTraits<ComputeType>::cudaType;
        cutensorComputeType_t computeType = CuTensorTypeTraits<ComputeType>::cutensorType;

   printf("ffffffffffffffffffffffffffffff###########################$$\n");
        cutensorTensorDescriptor_t descA;
        HANDLE_ERROR(cutensorInitTensorDescriptor(&cutensor_handle,
                    &descA,
                    numModesA_,
                    extentA_.data(),
                    NULL /* = stride */,
                    cudaType, CUTENSOR_OP_IDENTITY));

        cutensorTensorDescriptor_t descC;
        HANDLE_ERROR(cutensorInitTensorDescriptor(&cutensor_handle,
                    &descC,
                    numModesC_,
                    extentC_.data(),
                    NULL /* = stride*/,
                    cudaType, CUTENSOR_OP_IDENTITY));
   printf("#################################################$$\n");
        uint32_t alignmentRequirementA;
        HANDLE_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle,
                    A_raw, &descA, &alignmentRequirementA));

        uint32_t alignmentRequirementC;
        HANDLE_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle,
                    C_raw, &descC, &alignmentRequirementC));


   printf("22222222222222222222222$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
        cutensorTensorDescriptor_t descB;
        uint32_t alignmentRequirementB;
        if (numModesB_ > 0)
        {
            // dispatch to contraction
            HANDLE_ERROR(cutensorInitTensorDescriptor(&cutensor_handle,
                        &descB,
                        numModesB_,
                        extentB_.data(),
                        NULL /* = stride*/,
                        cudaType, CUTENSOR_OP_IDENTITY));

            HANDLE_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle,
                        B_raw, &descB, &alignmentRequirementB));

            cutensorContractionDescriptor_t desc;
            HANDLE_ERROR(cutensorInitContractionDescriptor(&cutensor_handle, &desc,
                        &descA, modesA_.data(), alignmentRequirementA,
                        &descB, modesB_.data(), alignmentRequirementB,
                        &descC, modesC_.data(), alignmentRequirementC,
                        &descC, modesC_.data(), alignmentRequirementC,
                        computeType));

            cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
            cutensorContractionFind_t find;
            HANDLE_ERROR(cutensorInitContractionFind( 
                        &cutensor_handle, &find, 
                        algo));

            cutensorContractionPlan_t plan;
            HANDLE_ERROR(cutensorInitContractionPlan(&cutensor_handle,
                        &plan, &desc, &find, kWorksize_));

    //         typename CuTensorTypeTraits<ComputeType>::ScalarType alpha1 = 1   ;
    //         typename CuTensorTypeTraits<ComputeType>::ScalarType beta1 = 0;

            HANDLE_ERROR(cutensorContraction(&cutensor_handle, &plan,
                        (const void*) &alpha, A_raw, B_raw,
                        (const void*) &beta,  C_raw, C_raw,
                        work_raw, kWorksize_, AsGpuStreamValue(stream)));
        }
        else
        {
            // dispatch to reduction
      //       typename CuTensorTypeTraits<ComputeType>::ScalarType alpha1 = alpha;
      //       typename CuTensorTypeTraits<ComputeType>::ScalarType beta1 = beta;
            HANDLE_ERROR(cutensorReduction(&cutensor_handle,
                        (const void*)&alpha, A_raw, &descA, modesA_.data(),
                        (const void*)&beta,  A_raw, &descC, modesC_.data(), // beta == 0 => will not be used
                        C_raw, &descC, modesC_.data(),
                        CUTENSOR_OP_ADD, computeType, work_raw, kWorksize_, AsGpuStreamValue(stream)));
        }

////////////////////////////////////////////////////////////////////////////////////////////////////

  return true;
}

void CUDATsr::SetTensorContents(
    const std::string &equation,
    const std::vector<int> &A_shape,
    const std::vector<int> &B_shape = emptyVec ) {

    numModesA_ = A_shape.size();
        numModesB_ = B_shape.size();
        numModesC_ = 0;

 std::cout<<equation<<std::endl;
  for (auto i: A_shape)
  std::cout<<i<<std::endl;
  const auto arrow_pos = equation.find("->");
  const auto comma_pos = equation.find(",");
  const auto dots = equation.find("...");
  const bool isBroadcast = (dots != std::string::npos);
  const bool isImplicit = (arrow_pos == std::string::npos);
  std::cout<<"setTensorContents here_111111111111!\n";
  if (isBroadcast) // TODO
  {
      return;
  }
  const bool usesB = (comma_pos != std::string::npos);
  if (! usesB)
  {
      numModesB_ = 0;
  }

  std::cout<<"setTensorContents here_222222222222222222222\n";
  size_t a_start = 0;
  size_t a_end = isImplicit ? ((comma_pos == std::string::npos) ? equation.size() : comma_pos) :
                              ((comma_pos == std::string::npos) ? arrow_pos : comma_pos);
  size_t b_start = usesB ? comma_pos + 1 : 0;
  size_t b_end   = usesB ? (isImplicit ? equation.size() : arrow_pos) : 0;
  size_t c_start = isImplicit ? equation.size() : arrow_pos + 2;
  size_t c_end = equation.size();

  std::cout<<"setTensorContents here_33333333333333\n";

  char modeA[kMaxNumModes_ + 2];
  uint32_t numModesA = 0;
  for (int i = a_start; i < a_end && numModesA < kMaxNumModes_ + 2; ++i){
      if (equation.at(i) != ' ') // skip spaces
      {
          modeA[numModesA++] = equation.at(i);
      }
  }

  char modeB[kMaxNumModes_ + 2];
  uint32_t numModesB = 0;
  for (int i = b_start; i < b_end && numModesB < kMaxNumModes_ + 2; ++i){
      if (equation.at(i) != ' ') // skip spaces
      {
          modeB[numModesB++] = equation.at(i);
      }
  }

  char modeC[kMaxNumModes_ + 2];
  uint32_t numModesC = 0;
  for (int i = c_start; i < c_end && numModesC < kMaxNumModes_ + 2; ++i){
      if (equation.at(i) != ' ') // skip spaces
      {
          modeC[numModesC++] = equation.at(i);
      }
  }

  std::cout<<"setTensorContents here_44444444444444\n";

  if ((numModesA != numModesA_) || (numModesB != numModesB_))
  {
      // substring size and shape don't match
      return;
  }
  if (numModesA_ > kMaxNumModes_ || numModesB_ > kMaxNumModes_)
  {
      // too many modes
      return;
  }

  std::cout<<"setTensorContents here!\n";
  /**
   * Copy all modes from modeA to modeC if they don't appear in modeB
   */
  auto copyModesIf = [](const char* modeA, uint32_t numModesA,
          const char* modeB, uint32_t numModesB,
          char* modeC, uint32_t &numModesC)
  {
      for (uint32_t i = 0; i < numModesA; i++)
      {
          auto mode = modeA[i];
          bool found = false;
          for(uint32_t j=0; j < numModesB; ++j){
              if(mode == modeB[j])
              {
                  found = true;
                  break;
              }
          }

          if (!found) // is non-contracted mode
          {
              modeC[numModesC++] = mode;
              if (numModesC > kMaxNumModes_)
              {
                  // too many modes
                  return false;
              }
          }
      }
      return true;
  };


  std::array<char, kMaxNumModes_+1> implicitModeC;
  char* redirectModeC;
  if (isImplicit)
  {
      // we have to copy all non-contracted modes from A over to C
      if (copyModesIf(modeA, numModesA_, modeB, numModesB_, implicitModeC.data(), numModesC_) == false)
      {
          return;
      }
      // we have to copy all non-contracted modes from B over to C
      if (copyModesIf(modeB, numModesB_, modeA, numModesA_, implicitModeC.data(), numModesC_) == false)
      {
          return;
      }
      std::sort(implicitModeC.begin(), std::next(implicitModeC.begin(), numModesC_)); // modes are sorted w.r.t. lexical order
      implicitModeC[numModesC_] = '\0';
      redirectModeC = implicitModeC.data();
  }
  else
  {
      redirectModeC = modeC;
      numModesC_ = numModesC;
  }

  for (uint32_t i = 0; i < numModesA_; i++)
  {
      modesA_[i] = modeA[numModesA_ - i - 1];
      extentA_[i] = A_shape[numModesA_ - i - 1];
  }

  for (uint32_t i = 0; i < numModesB_; i++)
  {
      modesB_[i] = modeB[numModesB_ - i - 1];
      extentB_[i] = B_shape[numModesB_ - i - 1];
  }

  for (uint32_t i = 0; i < numModesC_; i++)
  {
      const auto mode = redirectModeC[numModesC_ - i - 1];
      modesC_[i] = mode;
      bool found = false;
      for (uint32_t j=0; j < numModesA_; ++j)
      {
          if (modesA_[j] == mode)
          {
              extentC_[i] = extentA_[j];
              found = true;
              break;
          }
      }
      for (uint32_t j=0; !found && j < numModesB_; ++j)
      {
          if (modesB_[j] == mode)
          {
              extentC_[i] = extentB_[j];
              break;
          }
      }
  }

  std::cout<<"setTensor finished!\n";
  isInitialized_ = true;
}

#define STREAM_EXECUTOR_CUDA_DEFINE_TSR(__type)                                \
bool CUDATsr::DoTsrContraction(Stream *stream, tsr::Handle *handle,     \
                               __type &alpha, __type &beta, \
                                   const void* A_raw,                          \
                                   const void* B_raw, void* C_raw,             \
                                   void *work_raw) {                           \
  return DoCuTensorContractionInternal(                                \
      stream, handle,alpha, beta, A_raw, B_raw, C_raw, work_raw);                          \
}

STREAM_EXECUTOR_CUDA_DEFINE_TSR(double)
STREAM_EXECUTOR_CUDA_DEFINE_TSR(float)
STREAM_EXECUTOR_CUDA_DEFINE_TSR(Eigen::half)
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
