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

// Exposes the family of FFT routines as pre-canned high performance calls for
// use in conjunction with the StreamExecutor abstraction.
//
// Note that this interface is optionally supported by platforms; see
// StreamExecutor::SupportsFft() for details.
//
// This abstraction makes it simple to entrain FFT operations on GPU data into
// a Stream -- users typically will not use this API directly, but will use the
// Stream builder methods to entrain these operations "under the hood". For
// example:
//
//  DeviceMemory<std::complex<float>> x =
//    stream_exec->AllocateArray<std::complex<float>>(1024);
//  DeviceMemory<std::complex<float>> y =
//    stream_exec->AllocateArray<std::complex<float>>(1024);
//  // ... populate x and y ...
//  Stream stream{stream_exec};
//  std::unique_ptr<Plan> plan =
//     stream_exec.AsFft()->Create1dPlan(&stream, 1024, Type::kC2CForward);
//  stream
//    .Init()
//    .ThenFft(plan.get(), x, &y);
//  SE_CHECK_OK(stream.BlockHostUntilDone());
//
// By using stream operations in this manner the user can easily intermix custom
// kernel launches (via StreamExecutor::ThenLaunch()) with these pre-canned FFT
// routines.

#ifndef TENSORFLOW_STREAM_EXECUTOER_TSR_H_
#define TENSORFLOW_STREAM_EXECUTOER_TSR_H_
#include <complex>
#include <memory>
#include <vector>
#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {

class Stream;
template <typename ElemT>
class DeviceMemory;
class ScratchAllocator;

namespace tsr {

class Handle {
 public:
  virtual ~Handle() {}
};

class TsrSupport {
 public:
  virtual ~TsrSupport() {}

  // Creates a cuTensor handle.
  virtual std::unique_ptr<Handle> CreateHandle(
      Stream *stream, const std::string equation,
      const std::vector<int64> &A_shape, const std::vector<int64> &B_shape) = 0;

  virtual bool DoTsrContraction(Stream *stream, Handle *handle,
                                     const void* A_raw,
                                     const void* B_raw, void* C_raw,
                                     void *work_raw) = 0;
 protected:
  TsrSupport() {}

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(TsrSupport);
};

#define TENSORFLOW_STREAM_EXECUTOR_GPU_TSR_SUPPORT_OVERRIDES    \
  std::unique_ptr<tsr::Handle> CreateHandle(Stream *stream,            \
      const std::string equation, const std::vector<int64> &A_shape,     \
      const std::vector<int64> &B_shape) override; \
  bool DoTsrContraction(Stream *stream, tsr::Handle *handle, const void* A_raw, \
                             const void* B_raw, void* C_raw,             \
                             void *work_raw) override;


} // namespace tsr
} // namespace stream_executor


#endif // TENSORFLOW_STREAM_EXECUTOER_TSR_H_

