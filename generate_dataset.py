#
# Copyright (c) 2025, NVIDIA CORPORATION.
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
#
#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import math
import argparse
import sklearn


def generate_dataset(filename, n_samples, n_features, dtype=np.float32):
    """ Creates a dataset in big-ann-benchmark binary format.
    """
    dirname = os.path.dirname(filename)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    itemsize = np.dtype(dtype).itemsize
    total_size = n_samples * n_features * itemsize / (1 << 30)

    print(
        "Generating dataset {0} shape=({1},{2}), dtype={3}, size={4:6.1f} GiB".format(
            filename, n_samples, n_features, dtype, total_size
        )
    )

    with open(filename, "wb") as f:
        np.asarray([n_samples, n_features], dtype=np.uint32).tofile(f)

    fp = np.memmap(
        filename, dtype=dtype, mode="r+", shape=(n_samples, n_features), offset=8
    )
    n = 1000000
    i = 0
    
    n_iter = n_samples // n
    n_centers = int(max(math.sqrt(n_samples) / n_iter, 1))

    while i < n_samples:
        n_batch = n if i + n <= n_samples else n_samples - i

        tmp, y = sklearn.datasets.make_blobs(
            n_batch,
            n_features,
            centers=n_centers,
            cluster_std=3,
            shuffle=True,
            random_state=i
        )

        fp[i : i + n_batch, :] = tmp
        i += n_batch
        print(
            "Step {0}/{1}: {2:6.1f} GiB written".format(
                i // n, n_samples // n, i * n_features * itemsize / (1 << 30)
            )
        )

    fp.flush()
    del fp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate dataset", description="Generate random dataset"
    )
    parser.add_argument("filename", type=str, help="filename")

    parser.add_argument(
        "-N",
        "--rows",
        default=1000000,
        type=int,
        help="number of vectors in the dataset",
    )
    parser.add_argument(
        "-D",
        "--cols",
        default=256,
        type=int,
        help="number of features (dataset columns)",
    )

    args = parser.parse_args()

    generate_dataset(args.filename, args.rows, args.cols)