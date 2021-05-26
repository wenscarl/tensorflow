#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_TSR_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_TSR_H_

#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/tsr.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/temporary_device_memory.h"
#include "third_party/gpus/cuda/include/cutensor.h"


#define HANDLE_ERROR(x) { const auto err = x;\
    if (err == CUTENSOR_STATUS_NOT_SUPPORTED) { return false; }\
    if (err != CUTENSOR_STATUS_SUCCESS) {printf("cutensor_python: Error %s in line %d\n", cutensorGetErrorString(err), __LINE__); return false; } }


namespace stream_executor {
namespace gpu {

// Opaque and unique identifier for the cuDNN plugin.

extern const PluginId kCuTsrPlugin;

template<typename U>
struct CuTensorTypeTraits;

template<>
struct CuTensorTypeTraits<double> {
  static const cudaDataType_t cudaType = CUDA_R_64F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_R_MIN_64F;
  typedef double ScalarType;
};
template<>
struct CuTensorTypeTraits<float> {
  static const cudaDataType_t cudaType = CUDA_R_32F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_R_MIN_32F;
  typedef float ScalarType;
};
template<>
struct CuTensorTypeTraits<Eigen::half> {
  static const cudaDataType_t cudaType = CUDA_R_16F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_R_MIN_16F;
  typedef float ScalarType;
};
template<>
struct CuTensorTypeTraits<std::complex<float>> {
  static const cudaDataType_t cudaType = CUDA_C_32F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_C_MIN_32F;
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<std::complex<double>> {
  static const cudaDataType_t cudaType = CUDA_C_64F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_C_MIN_64F;
  typedef double ScalarType;
};

constexpr int kMaxNumModes_ = 12;
static const std::vector<int> emptyVec;

class CUDATsrHandle : public tsr::Handle {
 public:
  CUDATsrHandle()
      : parent_(nullptr), is_initialized_(false) {}
 //       cutensor_handle_(-1) {}
  ~CUDATsrHandle() override;

  cutensorHandle_t GetHandle() const {
    if (isInitialized()) {
      return cutensor_handle_;
    } else {
      LOG(FATAL) << "Try to get cutensorHandle value before initialization.";
    }
  }

  // Einsum<double, int64, 12>* GetCutensorEinsum() const {
  //   return cutensor_einsum_;
  // }

 port::Status Initialize(GpuExecutor* parent);
 protected:
  bool isInitialized() const { return is_initialized_; }

 private:
  GpuExecutor* parent_;
  cutensorHandle_t cutensor_handle_;
  // Einsum<double, int64, 12>* cutensor_einsum_;
  bool is_initialized_;
};


// cudnn-library based Tensor support. For details on overridden interface
// functions, see tsr.h.
class CUDATsr : public tsr::TsrSupport {
 public:
  explicit CUDATsr(GpuExecutor* parent) : parent_(parent) {}
  ~CUDATsr() override {}

  TENSORFLOW_STREAM_EXECUTOR_GPU_TSR_SUPPORT_OVERRIDES
 private:
  GpuExecutor* parent_;
  static const size_t kWorksize_ = 1024ULL * 1024ULL * 8ULL * 128ULL;
  uint32_t numModesA_;
  uint32_t numModesB_;
  uint32_t numModesC_;
  bool isInitialized_;
  std::array<int, kMaxNumModes_> modesA_;
  std::array<int, kMaxNumModes_> modesB_;
  std::array<int, kMaxNumModes_> modesC_;
  std::array<int64_t, kMaxNumModes_> extentA_;
  std::array<int64_t, kMaxNumModes_> extentB_;
  std::array<int64_t, kMaxNumModes_> extentC_;
  //store plan / cache line in future

  bool DoTsrInternal(Stream *stream, tsr::Handle *handle); //to fulfill later
  template <typename Dummy>
  cutensorHandle_t DoTsrInternalGetHandle(Stream *stream, tsr::Handle *handle, Dummy aaa);

  template <typename ComputeType>
  bool DoCuTensorContractionInternal(Stream *stream, tsr::Handle *handle,
                                ComputeType &alpha, ComputeType &beta,
                                const void* A_raw,
                                const void* B_raw, void* C_raw,
                                void *work_raw);
  void SetTensorContents(
      const std::string &equation,
      const std::vector<int> &A_shape,
      const std::vector<int> &B_shape);
  bool isInitialized() const { return isInitialized_; }
  size_t getWorksize() const { return kWorksize_; }

  std::vector<int> getOutputShape() const {
      if (!isInitialized_) return {};
      std::vector<int> extentC(numModesC_);
      for (int i=0; i < numModesC_; ++i)
      {
          extentC[i] = extentC_.at(numModesC_ - i - 1);
      }

      return extentC;
  }

  SE_DISALLOW_COPY_AND_ASSIGN(CUDATsr);
};


} // namespace gpu
} // namespace stream_executor

#endif // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_TSR_H_
