# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for checkpointing the TFRecordDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import zlib

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests import reader_dataset_ops_test_base
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class TFRecordDatasetCheckpointTest(
    reader_dataset_ops_test_base.TFRecordDatasetTestBase,
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

  def _build_iterator_graph(self,
                            num_epochs,
                            batch_size=1,
                            compression_type=None,
                            buffer_size=None):
    filenames = self._createFiles()
    if compression_type == "ZLIB":
      zlib_files = []
      for i, fn in enumerate(filenames):
        with open(fn, "rb") as f:
          cdata = zlib.compress(f.read())
          zfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.z" % i)
          with open(zfn, "wb") as f:
            f.write(cdata)
          zlib_files.append(zfn)
      filenames = zlib_files

    elif compression_type == "GZIP":
      gzip_files = []
      for i, fn in enumerate(self.test_filenames):
        with open(fn, "rb") as f:
          gzfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.gz" % i)
          with gzip.GzipFile(gzfn, "wb") as gzf:
            gzf.write(f.read())
          gzip_files.append(gzfn)
      filenames = gzip_files

    return core_readers.TFRecordDataset(
        filenames, compression_type,
        buffer_size=buffer_size).repeat(num_epochs).batch(batch_size)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordWithoutBufferCore(self):
    num_epochs = 5
    batch_size = num_epochs
    num_outputs = num_epochs * self._num_files * self._num_records // batch_size
    # pylint: disable=g-long-lambda
    self.run_core_tests(
        lambda: self._build_iterator_graph(num_epochs, batch_size,
                                           buffer_size=0),
        num_outputs)
    self.run_core_tests(
        lambda: self._build_iterator_graph(num_epochs, buffer_size=0),
        num_outputs * batch_size)
    # pylint: enable=g-long-lambda

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordWithBufferCore(self):
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records
    self.run_core_tests(lambda: self._build_iterator_graph(num_epochs),
                        num_outputs)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordWithCompressionCore(self):
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records
    self.run_core_tests(
        lambda: self._build_iterator_graph(num_epochs, compression_type="ZLIB"),
        num_outputs)
    self.run_core_tests(
        lambda: self._build_iterator_graph(num_epochs, compression_type="GZIP"),
        num_outputs)


if __name__ == "__main__":
  test.main()
