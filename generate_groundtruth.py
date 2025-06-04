#!/usr/bin/env python
#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import argparse
import os
import sys

from utils import memmap_bin_file, suffix_from_dtype, write_bin
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

from cuvs.common import Resources
from cuvs.neighbors.brute_force import build, search
import cupy as xp


def generate_random_queries(n_queries, n_features, dtype=xp.float32):
    print("Generating random queries")
    if xp.issubdtype(dtype, xp.integer):
        queries = xp.random.randint(
            0, 255, size=(n_queries, n_features), dtype=dtype
        )
    else:
        queries = xp.random.uniform(size=(n_queries, n_features)).astype(dtype)
    return queries


def choose_random_queries(dataset, n_queries):
    print("Choosing random vector from dataset as query vectors")
    query_idx = xp.random.choice(
        dataset.shape[0], size=(n_queries,), replace=False
    )
    return dataset[xp.asnumpy(query_idx), :]


def calc_truth(dataset, queries, k, metric="sqeuclidean"):
    n_samples = dataset.shape[0]
    n = 500000  # batch size for processing neighbors
    i = 0
    indices = None
    distances = None
    queries = xp.asarray(queries, dtype=xp.float32)

    resources = Resources()

    while i < n_samples:
        print("Step {0}/{1}:".format(i // n, n_samples // n))
        n_batch = n if i + n <= n_samples else n_samples - i

        X = xp.asarray(dataset[i : i + n_batch, :], xp.float32)

        index = build(X, metric=metric, resources=resources)
        D, Ind = search(index, queries, k, resources=resources)
        resources.sync()

        D, Ind = xp.asarray(D), xp.asarray(Ind)
        Ind += i  # shift neighbor index by offset i

        if distances is None:
            distances = D
            indices = Ind
        else:
            distances = xp.concatenate([distances, D], axis=1)
            indices = xp.concatenate([indices, Ind], axis=1)
            idx = xp.argsort(distances, axis=1)[:, :k]
            distances = xp.take_along_axis(distances, idx, axis=1)
            indices = xp.take_along_axis(indices, idx, axis=1)

        i += n_batch

    return distances, indices


def main():
    pool = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(), initial_pool_size=2**30
    )
    rmm.mr.set_current_device_resource(pool)
    xp.cuda.set_allocator(rmm_cupy_allocator)
 
    parser = argparse.ArgumentParser(
        prog="generate_groundtruth",
        description="Generate true neighbors using exact NN search. "
        "The input and output files are in big-ann-benchmark's binary format.",
        epilog="""Example usage
    # With existing query file
    python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.\
fbin --output=groundtruth_dir --queries=/dataset/query.public.10K.fbin

    # With randomly generated queries
    python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.\
fbin --output=groundtruth_dir --queries=random --n_queries=10000

    # Using only a subset of the dataset. Define queries by randomly
    # selecting vectors from the (subset of the) dataset.
    python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.\
fbin --nrows=2000000 --cols=128 --output=groundtruth_dir \
--queries=random-choice --n_queries=10000
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("dataset", type=str, help="input dataset file name")
    parser.add_argument(
        "--queries",
        type=str,
        default="random",
        help="Queries file name, or one of 'random-choice' or 'random' "
        "(default). 'random-choice': select n_queries vectors from the input "
        "dataset. 'random': generate n_queries as uniform random numbers.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="output directory name (default current dir)",
    )

    parser.add_argument(
        "--n_queries",
        type=int,
        default=10000,
        help="Number of quries to generate (if no query file is given). "
        "Default: 10000.",
    )

    parser.add_argument(
        "-N",
        "--rows",
        default=None,
        type=int,
        help="use only first N rows from dataset, by default the whole "
        "dataset is used",
    )
    parser.add_argument(
        "-D",
        "--cols",
        default=None,
        type=int,
        help="number of features (dataset columns). "
        "Default: read from dataset file.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Dataset dtype. When not specified, then derived from extension."
        " Supported types: 'float32', 'float16', 'uint8', 'int8'",
    )

    parser.add_argument(
        "-k",
        type=int,
        default=100,
        help="Number of neighbors (per query) to calculate",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sqeuclidean",
        help="Metric to use while calculating distances. Valid metrics are "
        "those that are accepted by cuvs.neighbors.brute_force.knn. Most"
        " commonly used with cuVS are 'sqeuclidean' and 'inner_product'",
    )
    parser.add_argument(
        "--dim_slice",
        type=int,
        default=0,
        help="slice the number of columns in the dataset"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if args.rows is not None:
        print("Reading subset of the data, nrows=", args.rows)
    else:
        print("Reading whole dataset")

    # Load input data
    dataset = memmap_bin_file(
        args.dataset, args.dtype, shape=(args.rows, args.cols)
    )
    n_features = dataset.shape[1]
    dtype = dataset.dtype

    if len(args.output) > 0:
        os.makedirs(args.output, exist_ok=True)

    if args.dim_slice > 0:
        print("Slicing dataset to {} dim".format(args.dim_slice))
        dataset = dataset[:,:args.dim_slice]
        write_bin(
            os.path.join(args.output, "dataset_{}".format(args.dim_slice) +  suffix_from_dtype(dataset.dtype)),
            dataset)


    print(
        "Dataset size {:6.1f} GB, shape {}, dtype {}".format(
            dataset.size * dataset.dtype.itemsize / 1e9,
            dataset.shape,
            xp.dtype(dtype),
        )
    )


    if args.queries == "random" or args.queries == "random-choice":
        if args.n_queries is None:
            raise RuntimeError(
                "n_queries must be given to generate random queries"
            )
        if args.queries == "random":
            queries = generate_random_queries(
                args.n_queries, n_features, dtype
            )
        elif args.queries == "random-choice":
            queries = choose_random_queries(dataset, args.n_queries)

        queries_filename = os.path.join(
            args.output, "queries" + suffix_from_dtype(dtype)
        )
        print("Writing queries file", queries_filename)
        write_bin(queries_filename, queries)
    else:
        print("Reading queries from file", args.queries)
        queries = memmap_bin_file(args.queries, dtype)
        if args.dim_slice > 0:
            print("Slicing queries to {} dim".format(args.dim_slice))
            queries = queries[:,:args.dim_slice]
            write_bin(
                os.path.join(args.output, "queries_{}".format(args.dim_slice) + suffix_from_dtype(queries.dtype)),
                queries)

    print("Calculating true nearest neighbors")
    distances, indices = calc_truth(dataset, queries, args.k, args.metric)

    write_bin(
        os.path.join(args.output, "groundtruth.neighbors.ibin"),
        indices.astype(xp.uint32),
    )
    write_bin(
        os.path.join(args.output, "groundtruth.distances.fbin"),
        distances.astype(xp.float32),
    )


if __name__ == "__main__":
    main()
