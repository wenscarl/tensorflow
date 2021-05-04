#!/bin/bash

# This script was created by referring to
# /opt/tensorflow/qa/L3_self_tests/test.sh in the NGC TF container

OUTPUT_USER_ROOT=/tf_cache/bazel

TARGETS=""

KERNEL_TEST_PATH=//tensorflow/python/kernel_tests
# Note that it's possible to add comments between lines, or at the end of lines below.
# ty_py_test is for CPU-only tests
# cuda_py_test is for GPU tests (not sure right now if they also run ops on CPU)
KERNEL_TESTS=(
  # bias_op_test
  # bias_op_deterministic_test
  # cudnn_deterministic_test
  # cudnn_deterministic_ops_test
  #
  # init_ops_test                       # cuda_py_test
  # conv1d_test                         # cuda_py_test
  # conv2d_backprop_filter_grad_test    # cuda_py_test
  # conv2d_transpose_test               # cuda_py_test
  # conv3d_backprop_filter_v2_grad_test # cuda_py_test
  # conv3d_transpose_test               # cuda_py_test
  # conv_ops_test                       # cuda_py_test
  # conv_ops_3d_test                    # cuda_py_test
  ## atrous_conv2d_test                 # cuda_py_test / has no_gpu tag
  # atrous_convolution_test             # cuda_py_test
  ## neon_depthwise_conv_op_test        # tf_py_test
  ## depthwise_conv_op_test             # tf_py_test
  # pool_test                           # cuda_py_test
  # pooling_ops_test                    # cuda_py_test
  # pooling_ops_3d_test                 # cuda_py_test
  ## fractional_max_pool_op_test        # tf_py_test
)

if [ ${#KERNEL_TESTS[@]} -gt 0 ] ; then
  for KERNEL_TEST in "${KERNEL_TESTS[@]}" ; do
    TARGETS="${TARGETS} ${KERNEL_TEST_PATH}:${KERNEL_TEST}"
  done
fi

PYTHON_TEST_PATH=//tensorflow/python/kernel_tests
PYTHON_TESTS=(
#  image_ops_test
#  image_grad_test
#  image_grad_deterministic_test
  cudnn_deterministic_test
  sparse_tensor_dense_matmul_op_test
  # add other tests
)

if [ ${#PYTHON_TESTS[@]} -gt 0 ] ; then
  for PYTHON_TEST in "${PYTHON_TESTS[@]}" ; do
    TARGETS="${TARGETS} ${PYTHON_TEST_PATH}:${PYTHON_TEST}"
  done
fi

# TARGETS=//tensorflow/python/eager:forwardprop_test_gpu
# TARGETS=//tensorflow/python/keras:pooling_test
# TARGETS=//tensorflow/python:gradient_checker_v2_test
# TARGETS=//tensorflow/python/keras:convolutional_test

TEST_OUTPUT=all # other options: errors
CUDNN_LOG='--action_env=CUDNN_LOGINFO_DBG=1 --action_env=CUDNN_LOGDEST_DBG=stdout'

# Use the following to make the cache persistent outside the container:
# bazel --output_user_root=${OUTPUT_USER_ROOT} test

# Notes on how to run on XLA:GPU:
#
# Add the following two options after 'bazel test' and before '--':
#   --config=xla
#   --action_env=TF_XLA_FLAGS=--tf_xla_auto_jit=2
#
# The second option is necessary, but I have not yet confirmed that the first 
# option is necessary.
#
# Search for XLA in the log output to find print statements that confirm that
# XLA has been enabled. Also try breaking RequireDeterminism() to always return
# false in tensorflow/compiler/xla/service/gpu/gpu_compiler.cc and confirm that
# bias_op_deterministic_test fails.
#
# For running on XLA:CPU, TF_XLA_FLAGS needs to be expanded as follows, but I
# don't currently know how to achieve that using --action_env (e.g. can these
# three options be put in quotations, assigned to TF_XLA_FLAGS and then all of
# that be assigned to --action_env ?):
#   --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluser_size=1
#
# For more information see:
# https://confluence.nvidia.com/display/DL/XLA+team+knowledge-base+and+best+practices#XLAteamknowledge-baseandbestpractices-EnablingXLA
# https://confluence.nvidia.com/display/DL/XLA+team+knowledge-base+and+best+practices#XLAteamknowledge-baseandbestpractices-TestingwithXLAonly

bazel test --config=cuda                                                          \
           -c opt                                                                 \
           --verbose_failures                                                     \
           --test_verbose_timeout_warnings                                        \
           --local_test_jobs=1                                                    \
           --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
           --test_tag_filters=-no_gpu,-benchmark_test                             \
           --cache_test_results=no                                                \
           --build_tests_only                                                     \
           --test_output=${TEST_OUTPUT}                                           \
           --                                                                     \
           ${TARGETS}
