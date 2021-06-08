# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for tensorflow.ops.Einsum."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

class EinsumcuTENSORTest(test.TestCase):

    def _check(self, s, *input_shapes, **kwargs):
      dtype = kwargs.pop('dtype', np.float32)
      inputs = []
      for shape in input_shapes:
        with self.subTest(s=s, shape=shape):
          arr = np.array(np.random.random(shape)).astype(dtype)
          if dtype == np.complex64 or dtype == np.complex128:
            arr += 1j * np.array(np.random.random(shape)).astype(dtype)
          inputs.append(arr)
      input_tensors = [constant_op.constant(x, shape=x.shape) for x in inputs]
      a = np.einsum(s, *inputs)
      b = self.evaluate(gen_linalg_ops.einsum(input_tensors, s))
      self.assertAllClose(a, b, atol=5e-3, rtol=5e-3)

    def _check_gradient(self, s, *input_shapes, **kwargs):
      dtype = kwargs.pop('dtype', np.float32)
      with self.cached_session():
        with self.subTest(s=s, dtype=dtype):
          tol = 10 * np.sqrt(np.finfo(dtype).resolution)
          if dtype in (np.complex64, np.complex128):
            inputs = [
                np.array(r.random.random(shape), dtype) +
                1j * np.array(np.random.random(shape), dtype) for shape in input_shapes
            ]
          else:
            inputs = [
                np.array(np.random.random(shape), dtype) for shape in input_shapes]
          input_tensors = [
              constant_op.constant(x, shape=x.shape) for x in inputs]
          analytical, numerical = gradient_checker_v2.compute_gradient(
              lambda *xs: gen_linalg_ops.einsum(xs, s), input_tensors)
          self.assertLess(
              gradient_checker_v2.max_error(analytical, numerical), tol)


    def testBinary(self):
      # Binary cases in XLA mode must have either (a) each index appearing exactly
      # once in both the inputs (batch or contraction index), or (b) appearing
      # exactly once in an input and in the output (free index).
      self._check("ik,kj->ij", (50, 50), (50, 50))
      self._check("lik,lkj->lij", (50, 50, 50), (50, 50, 50))
      self._check("likm,lkjm->lij", (50, 50, 50, 20),(50, 50, 50, 20))
      self._check("mlik,lkjm->lij", (20, 50, 50, 50), (50, 50, 50, 20))
      self._check("ik,kj->ij", (50, 50), (50, 50), dtype=np.float16)
      self._check("lik,lkj->lij", (50, 50, 50), (50, 50, 50), dtype=np.float16)
      self._check("likm,lkjm->lij", (50, 50, 50, 20), (50, 50, 50, 20),
                  dtype=np.float16)
      self._check("mlik,lkjm->lij", (20, 50, 50, 50), (50, 50, 50, 20),
                  dtype=np.float16)
      self._check("mlik,lkjm->lij", (20, 50, 50, 50), (50, 50, 50, 20),
                  dtype=np.complex64)
      self._check("mlik,lkjm->lij", (20, 50, 50, 50), (50, 50, 50, 20),
                  dtype=np.complex128)

    def testUnary(self):
      self._check("mlik->imkl", (50, 40, 40, 50))
      self._check("mlik->kl", (20, 40, 40, 50))

    def testBroadcasting(self):
      self._check("...ij,...jk->...ik", (11, 7 ,5, 30), (11, 7, 30, 2),
                  dtype=np.float16)

    def testBinaryGrad(self):
      self._check_gradient('a,a->a', (3,), (3,))
      self._check_gradient('ab,b->a', (3, 4), (4,))
      self._check_gradient('ab,ab->', (3, 4), (3, 4))
      self._check_gradient('ab,bc->ac', (3, 4), (4, 5))
      self._check_gradient('nij,jk->nik', (5, 2, 3), (3, 4))
      self._check_gradient('abc,bad->abcd', (1, 2, 3), (2, 1, 4))

    def testUnaryGrad(self):
       self._check_gradient('abcd->da', (3, 5, 4, 2))

    def testBroadcastingGrad(self):
      self._check_gradient('...ij,...jk->...ik', (3, 2), (2, 4))
      self._check_gradient('ij...,jk...->ik...', (3, 2, 1), (2, 4))
      self._check_gradient('...ij,...jk->...ik', (3, 1, 3, 2), (1, 5, 2, 4))
      self._check_gradient('i...j,j...k->i...k', (3, 1, 2, 2), (2, 2, 3, 1, 4))


if __name__ == '__main__':
    test.main()


