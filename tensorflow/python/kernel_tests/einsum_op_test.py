# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for conv3d_cu_tensor ops."""

from parameterized import parameterized
from parameterized import param
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.platform import test


#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



class EinsumcuTENSORTest(test.TestCase):

    # Assert TF test methods:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/test_util.py#L2474

    @parameterized.expand(
        # yapf: disable
        [
            param(
                "test 0",
                a_size=(50, 50),
                b_size=(50, 50),
                equation="ik,kj->ij",
                dtype=np.float32,
            ),
#            param(
#                "test 1",
#                a_size=(50, 50, 50),
#                b_size=(50, 50, 50),
#                equation="lik,lkj->lij",
#                dtype=np.float32,
#            ),
#            param(
#                "test 2",
#                a_size=(50, 50, 50, 20),
#                b_size=(50, 50, 50, 20),
#                equation="likm,lkjm->lij",
#                dtype=np.float32,
#            ),
#            param(
#                "test 3",
#                a_size=(20, 50, 50, 50),
#                b_size=(50, 50, 50, 20),
#                equation="mlik,lkjm->lij",
#                dtype=np.float32,
#            ),
#            param(
#                "test 4",
#                a_size=(50, 50),
#                b_size=(50, 50),
#                equation="ik,kj->ij",
#                dtype=tf.float16,
#            ),
#            param("test 5", a_size=(50, 50, 50), b_size=(50, 50, 50), equation="lik,lkj->lij", dtype=tf.float16),
#            param(
#                "test 6",
#                a_size=(50, 50, 50, 20),
#                b_size=(50, 50, 50, 20),
#                equation="likm,lkjm->lij",
#                dtype=tf.float16,
#            ),
#            param(
#                "test 7",
#                a_size=(20, 50, 50, 50),
#                b_size=(50, 50, 50, 20),
#                equation="mlik,lkjm->lij",
#                dtype=tf.float16,
#            ),
        ]
        # yapf: enable
    )
    def test_einsum_equivalent_results(self, _, a_size, b_size, equation, dtype=np.float32):
   #     A = tf.compat.v1.get_variable("A", shape=a_size, initializer=tf.random_normal_initializer, dtype=dtype)
        A = np.random.random(size=a_size).astype(dtype)
  #      B = tf.compat.v1.get_variable("B", shape=b_size, initializer=tf.random_normal_initializer, dtype=dtype)
        B = np.random.random(size=b_size).astype(dtype)

        tf_native_rslt = gen_linalg_ops.einsum([A,B],equation)
#        tf_native_grads = tf.gradients(tf_native_rslt, [A, B])

        tf_cutensor_rslt = np.einsum(equation, A, B)
#        tf_cutensor_grads = tf.gradients(tf_cutensor_rslt, [A, B])

        self.assertEqual(tf_native_rslt.get_shape(), tf_cutensor_rslt.shape)

        self.assertEqual(tf_native_rslt.dtype, tf_cutensor_rslt.dtype)
        self.assertAllClose(tf_native_rslt, tf_cutensor_rslt, rtol=5e-03, atol=5e-03)
     #   self.assertEqual(len(tf_cutensor_grads), len(tf_native_grads))

#        with self.session(use_gpu=True) as sess:
#
#            sess.run(tf.compat.v1.global_variables_initializer())
#
#            # mismatch 0.001885741949081421%
#            self.assertAllClose(tf_native_rslt, tf_cutensor_rslt, rtol=5e-03, atol=5e-03)

#            for tf_native_grad, tf_cutensor_grad in zip(tf_native_grads, tf_cutensor_grads):
#                self.assertAllClose(tf_native_grad, tf_cutensor_grad, rtol=5e-03, atol=5e-03)
#                self.assertEqual(tf_native_grad.dtype, tf_cutensor_grad.dtype)


if __name__ == '__main__':
    test.main()

