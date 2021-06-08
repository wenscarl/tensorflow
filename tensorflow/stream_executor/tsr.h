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

// Exposes the family of TSR routines as pre-canned high performance calls for
// use in conjunction with the StreamExecutor abstraction.
//
// Note that this interface is optionally supported by platforms; see
// StreamExecutor::TsrSupport() for details.
//
// This abstraction makes it simple to entrain Einsum operation on GPU data into
// a Stream -- users typically will not use this API directly, but will use the
// Stream builder methods to entrain these operations "under the hood".
// By using stream operations in this manner the user can easily intermix custom
// kernel launches (via StreamExecutor::ThenLaunch()) with these pre-canned TSR
// routines.

#ifndef TENSORFLOW_STREAM_EXECUTOER_TSR_H_
#define TENSORFLOW_STREAM_EXECUTOER_TSR_H_

#include <complex>
#include <memory>
#include <vector>

#include "tensorflow/stream_executor/lib/status.h"


namespace stream_executor {

class Stream;

namespace tsr {

struct TsrTypeHelper {
  enum TsrTypeAlias {
    INVALID_TYPE = 0,
    FLOAT_TYPE,
    DOUBLE_TYPE,
    HALF_TYPE,
    COMPLEX64_TYPE,
    COMPLEX128_TYPE
  };

  template <typename T>
  static TsrTypeAlias TsrInitType() {
    if (std::is_same<T, double>::value) {
      return TsrTypeAlias::DOUBLE_TYPE;
    } else if (std::is_same<T, float>::value) {
      return TsrTypeAlias::FLOAT_TYPE;
    } else if (std::is_same<T, Eigen::half>::value) {
      return TsrTypeAlias::HALF_TYPE;
    } else if (std::is_same<T, std::complex<float>>::value) {
      return TsrTypeAlias::COMPLEX64_TYPE;
    } else if (std::is_same<T, std::complex<double>>::value) {
      return TsrTypeAlias::COMPLEX128_TYPE;
    } else {
      return TsrTypeAlias::INVALID_TYPE;
    }
  }
};


class TsrSupport {
 public:
  virtual ~TsrSupport() {}

  virtual port::Status InitializeModes(
              Stream *stream, std::vector<int64> &output_dims,
              TsrTypeHelper::TsrTypeAlias type,
              const std::string &equation,
              const std::vector<int> &A_shape,
              const std::vector<int> &B_shape) = 0;

  virtual bool EstimateWorkSpace(
      Stream *stream, size_t &worksize, const void* A_raw,  const void* B_raw,
      void* C_raw) = 0;

  virtual bool DoTsrContraction(Stream *stream,
                                const void* A_raw,
                                const void* B_raw, void* C_raw,
                                void *work_raw) = 0;

 protected:
  TsrSupport() {}

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(TsrSupport);
};

#define TENSORFLOW_STREAM_EXECUTOR_GPU_TSR_SUPPORT_OVERRIDES                   \
  port::Status InitializeModes(                                                \
      Stream *stream, std::vector<int64> &output_dims,                         \
      tsr::TsrTypeHelper::TsrTypeAlias type,                                   \
      const std::string &equation,                                             \
      const std::vector<int> &A_shape,                                         \
      const std::vector<int> &B_shape) override;                               \
  bool EstimateWorkSpace(                                                      \
      Stream *stream, size_t &worksize,                                        \
      const void* A_raw,  const void* B_raw, void* C_raw) override;            \
  bool DoTsrContraction(Stream *stream,                                 \
                        const void* A_raw,  const void* B_raw,                 \
                        void* C_raw, void *work_raw) override;
} // namespace tsr
} // namespace stream_executor

#endif // TENSORFLOW_STREAM_EXECUTOER_TSR_H_