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

import numpy as np
import cupy as cp
import math

from utils import memmap_bin_file, calc_recall, BenchmarkTimer
from timeit import default_timer as timer
import faiss
import argparse
from scipy.stats import describe

import rmm
pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaMemoryResource(), initial_pool_size=8*(1<<30)
)
rmm.mr.set_current_device_resource(pool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main_single_faiss", description="Run faiss ANN search"
    )

    parser.add_argument('dataset', type=str, help="dataset file name")
    parser.add_argument('queries', type=str, help="queries file name")
    parser.add_argument("ground_truth", type=str, help="ground_truth file name")
    parser.add_argument(
        "-N",
        "--rows",
        default=0,
        type=int,
        help="use only first N rows from dataset, by default the whole dataset is used",
    )
    parser.add_argument(
        "-D",
        "--cols",
        default=0,
        type=int,
        help="number of features (dataset columns). Must be specified if --rows is given.",
    )
    parser.add_argument(
        "-C",
        "--clusters",
        default=0,
        type=int,
        help="number of clusters to use. (default sqrt(n_rows))",
    )
    parser.add_argument(
        "-M",
        "--pq_dim",
        default=64,
        type=int,
        help="pq dimensions parameter"
    )
    parser.add_argument(
        "-k",
        "--top_k",
        default=2048,
        type=int,
        help="number of neighbors to search"
    )
    parser.add_argument('--batch_size', type=lambda s: [int(item) for item in s.split(',')],
                    help='Comma-separated list of batch size for calling search()', 
                    default=[1])
    

    parser.add_argument("-r", "--read_index", action="store_true")
    parser.add_argument("-f", "--index_file", default="tmp_index_file")
    parser.add_argument("-v", "--verbose", help="Print more details on measured times", action="store_true")

  
    args = parser.parse_args()
    if args.rows != 0 and args.cols == 0:
        raise RuntimeError(
            "Number of columns has to be specified with the --cols argument"
        )
   
    n_samples = args.rows
    n_features = args.cols
    if n_samples != 0:
        shape = (n_samples, n_features)
        print("Reading subset of the data, shape=", shape)
    else:
        print("Reading whole dataset")
        shape = None

    dtype = np.float32
    dataset = memmap_bin_file(args.dataset, dtype, shape=shape)
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]
    queries = np.asarray(memmap_bin_file(args.queries, dtype))
    gt_indices = memmap_bin_file(args.ground_truth, dtype=np.int32)

    if args.clusters > 0:
        n_clusters = args.clusters
    else:
        n_clusters = max(int(math.sqrt(n_samples)) // 1024, 1) * 1024
    
    print("n_clusters:", n_clusters)

    print(
        "Dataset shape {0}, size {1:6.1f} GiB".format(
            dataset.shape,
            dataset.size * dataset.dtype.itemsize / (1024 * 1024 * 1024),
        )
    )
    print("queries shape", queries.shape)

    # we need only k columns from the groundthruth files
    k = args.top_k
    gt_indices = np.asarray(gt_indices[:, :k])

    # Set parameters for ANN search
    build_params = {
        "n_lists": n_clusters,
        "metric": faiss.METRIC_L2,
        "pq_dim": args.pq_dim,
        "pq_bits": 8,
    }
    search_params = {
        "n_probes": 64,
    }

    if args.read_index:
        index = faiss.read_index(args.index_file)
        print("Index read from", args.index_file)
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        # co.useFloat16 = True
        # co.usePrecomputed = True
        # make it an IVF GPU index
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
        print("Index moved to GPU")
    else:
        print("\nBuilding index")

        quantizer = faiss.IndexFlatL2(n_features)

        index = faiss.IndexIVFPQ(
            quantizer,
            n_features,
            build_params["n_lists"],
            build_params["pq_dim"],
            build_params["pq_bits"],
            build_params["metric"],
        )

        start = timer()
        
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        # co.useFloat16 = True
        # co.usePrecomputed = True
        # make it an IVF GPU index
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
        index.train(dataset)

        assert index.is_trained
        build_time_1 = timer() - start
        print("Index trained in {0:6.1f} s, adding vectors".format(build_time_1))

        start = timer()
        build_batch_size = 1000000
        i = 0
        step_i = 0
        total_steps = dataset.shape[0] / build_batch_size
        while i < dataset.shape[0]:
            if i + build_batch_size <= dataset.shape[0]:
                current_batch_size = build_batch_size
            else:
                current_batch_size = dataset.shape[0] - i
            index.add(dataset[i : i + current_batch_size, :])
            dt = timer() - start
            step_i += 1
            i += current_batch_size
            t_remaining = dt / step_i * total_steps - dt
            print(
                "step {} out of {} completed, time {:6.1f} s, t_remaining {:6.1f} s".format(
                    step_i, int(math.ceil(total_steps)), dt, t_remaining
                )
            )
        build_time_2 = timer() - start

        print(
            "Add time {0:6.1f} s, total time {1:6.1f} s".format(
                build_time_2, build_time_1 + build_time_2
            )
        )

        print("Saving index to", args.index_file)
        
        index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index_cpu, args.index_file)
        del index_cpu
        
    


    neighbors = np.zeros((queries.shape[0], k), dtype=np.int64)

    cp.cuda.profiler.start()

    res_list = ["batch_size,n_clusters,n_probes,k,recall,qps,avg_time (ms),std_time (ms)"]

    for query_batch_size in args.batch_size:
        print("query with batch size", query_batch_size)

        timer = BenchmarkTimer(reps=1000, warmup=10)
        index.nprobe = search_params["n_probes"]
    
        i = 0
        for rep in timer.benchmark_runs():
            if i + query_batch_size > queries.shape[0]:
                i = 0
                if i + query_batch_size > queries.shape[0]:
                    raise RuntimeError("Too large batch size")
            (
                distances,
                neighbors[i : i + query_batch_size, :],
            ) = index.search(queries[i : i + query_batch_size, :], k)
            i += query_batch_size

        r = calc_recall(neighbors[:i,:], gt_indices[:i,:])
        print("recall", r)

        timings = np.asarray(timer.timings)
        if args.verbose:
            print(describe(timings))
            print(timings[:20])
        avg_time = timings.mean() * 1000
        std_time = timings.std() * 1000
        qps = query_batch_size / (avg_time / 1000)
        print(
            "Average search time: {0:7.3f} +/- {1:7.3} ms".format(avg_time, std_time)
        )
        print(
            "Queries per second (QPS): {0:8.0f}".format(qps)
        )
        res_list.append(
            "{},{},{},{},{:.4f},{:.0f},{:.3f},{:.3f}".format(
                query_batch_size, n_clusters, search_params['n_probes'], k, r, qps, avg_time, std_time
            )
        )

    print("\nResult table")
    print("\n".join(res_list))



