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
#ifndef TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

#include "third_party/gpus/cuda/include/cutensor.h"
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

#define HANDLE_ERROR(x) { const auto err = x;\
    if (err == CUTENSOR_STATUS_NOT_SUPPORTED) { return false; }\
    if (err != CUTENSOR_STATUS_SUCCESS) {printf("cutensor_python: Error %s in line %d\n", cutensorGetErrorString(err), __LINE__); return false; } }


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
struct CuTensorTypeTraits<complex64> {
  static const cudaDataType_t cudaType = CUDA_C_32F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_C_MIN_32F;
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<complex128> {
  static const cudaDataType_t cudaType = CUDA_C_64F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_C_MIN_64F;
  typedef double ScalarType;
};
template<typename ComputeType,
         typename IntType, int kMaxNumModes_>
struct Einsum
{
    static const std::vector<IntType> emptyVec;

    Einsum(const std::string &equation,
           const std::vector<IntType> &A_shape,
           const std::vector<IntType> &B_shape = emptyVec) :
        numModesA_(A_shape.size()),
        numModesB_(B_shape.size()),
        numModesC_(0),
        isInitialized_(false)
    {
        const auto arrow_pos = equation.find("->");
        const auto comma_pos = equation.find(",");
        const auto dots = equation.find("...");
        const bool isBroadcast = (dots != std::string::npos);
        const bool isImplicit = (arrow_pos == std::string::npos);
        if (isBroadcast) // TODO
        {
            return;
        }
        const bool usesB = (comma_pos != std::string::npos);
        if (! usesB)
        {
            numModesB_ = 0;
        }

        size_t a_start = 0;
        size_t a_end = isImplicit ? ((comma_pos == std::string::npos) ? equation.size() : comma_pos) : 
                                    ((comma_pos == std::string::npos) ? arrow_pos : comma_pos);
        size_t b_start = usesB ? comma_pos + 1 : 0;
        size_t b_end   = usesB ? (isImplicit ? equation.size() : arrow_pos) : 0;
        size_t c_start = isImplicit ? equation.size() : arrow_pos + 2;
        size_t c_end = equation.size();


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

        isInitialized_ = true;
    }

    size_t getWorksize() const { return kWorksize_; }

    std::vector<IntType> getOutputShape() const
    {
        if (!isInitialized_) return {};
        std::vector<IntType> extentC(numModesC_);
        for (int i=0; i < numModesC_; ++i)
        {
            extentC[i] = extentC_.at(numModesC_ - i - 1);
        }

        return extentC;
    }

    /**
     * Computes the einsum call A,B->C
     *
     * \param[in] A_raw device pointer of A
     * \param[in] B_raw device pointer of B
     * \param[out] C_raw device pointer of C
     * \param[out] wor_raw device pointer to the scratchpad memory
     * Dispatch to contraction
     */
    bool execute(const cutensorHandle_t *handle,
                 const void* A_raw,
                 const void* B_raw,
                 void* C_raw,
		 void *work_raw, cudaStream_t stream) 
    {
        if (!isInitialized_) return false;

        cudaDataType_t cudaType = CuTensorTypeTraits<ComputeType>::cudaType;
        cutensorComputeType_t computeType = CuTensorTypeTraits<ComputeType>::cutensorType;

        cutensorTensorDescriptor_t descA;
        HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                    &descA,
                    numModesA_,
                    extentA_.data(),
                    NULL /* = stride */,
                    cudaType, CUTENSOR_OP_IDENTITY));

        cutensorTensorDescriptor_t descC;
        HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                    &descC,
                    numModesC_,
                    extentC_.data(),
                    NULL /* = stride*/,
                    cudaType, CUTENSOR_OP_IDENTITY));

        uint32_t alignmentRequirementA;
        HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    A_raw, &descA, &alignmentRequirementA));

        uint32_t alignmentRequirementC;
        HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    C_raw, &descC, &alignmentRequirementC));


        cutensorTensorDescriptor_t descB;
        uint32_t alignmentRequirementB;
        if (numModesB_ > 0)
        {
            // dispatch to contraction
            HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                        &descB,
                        numModesB_,
                        extentB_.data(),
                        NULL /* = stride*/,
                        cudaType, CUTENSOR_OP_IDENTITY));

            HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                        B_raw, &descB, &alignmentRequirementB));

            cutensorContractionDescriptor_t desc;
            HANDLE_ERROR(cutensorInitContractionDescriptor(handle, &desc,
                        &descA, modesA_.data(), alignmentRequirementA,
                        &descB, modesB_.data(), alignmentRequirementB,
                        &descC, modesC_.data(), alignmentRequirementC,
                        &descC, modesC_.data(), alignmentRequirementC,
                        computeType));

            cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
            cutensorContractionFind_t find;
            HANDLE_ERROR(cutensorInitContractionFind( 
                        handle, &find, 
                        algo));

            cutensorContractionPlan_t plan;
            HANDLE_ERROR(cutensorInitContractionPlan(handle,
                        &plan, &desc, &find, kWorksize_));

            typename CuTensorTypeTraits<ComputeType>::ScalarType alpha = 1;
            typename CuTensorTypeTraits<ComputeType>::ScalarType beta = 0;

            HANDLE_ERROR(cutensorContraction(handle, &plan,
                        (void*) &alpha, A_raw, B_raw,
                        (void*) &beta,  C_raw, C_raw,
                        work_raw, kWorksize_, stream));
        }
        else
        {
            // dispatch to reduction
            typename CuTensorTypeTraits<ComputeType>::ScalarType alpha = 1;
            typename CuTensorTypeTraits<ComputeType>::ScalarType beta = 0;
            HANDLE_ERROR(cutensorReduction(handle,
                        (const void*)&alpha, A_raw, &descA, modesA_.data(),
                        (const void*)&beta,  A_raw, &descC, modesC_.data(), // beta == 0 => will not be used
                        C_raw, &descC, modesC_.data(),
                        CUTENSOR_OP_ADD, computeType, work_raw, kWorksize_, stream));
        }

	//nullinit();
        return true;
    }

    bool isInitialized() const { return isInitialized_; }
    void nullinit() { isInitialized_ = false; }

    private:
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
};

namespace functor {

//template <typename Device, typename T, int N>
//struct CutensorEinsumFunctor {
//  static EIGEN_ALWAYS_INLINE Status Compute();
//};

template <typename Device, typename T, int N>
struct StrideFunctor {
  void operator()(const Device& d, typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    output.device(d) = input.stride(strides);
  }
};

template <typename Device, typename T, int N>
struct InflateFunctor {
  void operator()(const Device& d, typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    output.device(d) = input.inflate(strides);
  }
};

//template <typename Device, typename T>
//struct EinsumCutensorFunctor {
//  static EIGEN_ALWAYS_INLINE Status Compute(
//      OpKernelContext* ctx, typename TTypes<T>::Flat input_a,
//      typename TTypes<T>::Flat input_b, TTypes<float>::Flat work_tensor,
//      typename TTypes<T>::Flat output_c,
//      std::vector<int64> input_a_shape,
//      std::vector<int64> input_b_shape)
//      {}
//};
template <typename Device, typename T>
struct EinsumCutensorFunctor {
  static EIGEN_ALWAYS_INLINE Status Compute(
      OpKernelContext* ctx, const T* input0, const T* input1, 
      T* out, float* works, string equation_, std::vector<int64> xx, 
      std::vector<int64> yy) {}
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_EINSUM_OP_H_
