# FAISS-cuVS synthetic benchmarks


## Requirements

### Install FAISS with cuVS backend
```
conda create -n faiss_test -c pytorch -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia -c conda-forge pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version>=12.0,<=12.5' scikit-learn cuvs-bench=25.04 cuml=25.04
conda activate faiss_test
```

### Get benchmark scripts
```
git clone https://github.com/tfeher/faiss_cuvs_test.git
```

## Create dataset

```
cd faiss_cuvs_test
python generate_dataset.py blobs.100M.fbin -N 100000000 -D 256
python generate_groundtruth.py blobs.100M.fbin --queries random-choice -k 4096 --metric euclidean --dtype=float32 --output blobs_256_100M

```
 ## Run test

 ### Build and Search

```
python faiss_test.py -N 100000000 -D 256 -C 16384 blobs.100M.fbin blobs_256_100M/queries.fbin blobs_256_100M/groundtruth.neighbors.ibin
```

 Expected output:
```
Reading subset of the data, shape= (100000000, 256)
n_clusters: 16384
Dataset shape (100000000, 256), size   95.4 GiB
queries shape (10000, 256)

Building index
Index trained in    8.2 s, adding vectors
step 1 out of 100 completed, time    0.8 s, t_remaining   81.6 s
...
step 100 out of 100 completed, time   68.0 s, t_remaining    0.0 s
Add time   68.0 s, total time   76.1 s
Saving index to tmp_index_file
query with batch size 1
recall 0.7394985689975248
Average search time:   1.601 +/-   0.101 ms
Queries per second (QPS):      625

Result table
batch_size,n_clusters,n_probes,k,recall,qps,avg_time (ms),std_time (ms)
1,16384,64,2048,0.7395,625,1.601,0.101
 ```

 ### Search using saved index

```
python faiss_test.py -N 100000000 -D 256 -C 16384 blobs.100M.fbin blobs_256_100M/queries.fbin blobs_256_100M/groundtruth.neighbors.ibin --read_index --batch_size 1,1,200,200
```
Expected output
```
python faiss_test.py -N 100000000 -D 256 -C 16384 blobs.100M.fbin blobs_256_100M/queries.fbin blobs_256_100M/groundtruth.neighbors.ibin --read_index --batch_size 1,1,200,200
Reading subset of the data, shape= (100000000, 256)
n_clusters: 16384
Dataset shape (100000000, 256), size   95.4 GiB
queries shape (10000, 256)
Index read from tmp_index_file
Index moved to GPU
query with batch size 1
recall 0.7394874497215347
Average search time:   0.574 +/-   0.254 ms
Queries per second (QPS):     1741
query with batch size 1
recall 0.7394874497215347
Average search time:   0.574 +/-   0.254 ms
Queries per second (QPS):     1742
query with batch size 200
recall 0.741208984375
Average search time:  10.685 +/-   0.197 ms
Queries per second (QPS):    18718
query with batch size 200
recall 0.741208984375
Average search time:  10.703 +/-    0.19 ms
Queries per second (QPS):    18687

Result table
batch_size,n_clusters,n_probes,k,recall,qps,avg_time (ms),std_time (ms)
1,16384,64,2048,0.7395,1741,0.574,0.254
1,16384,64,2048,0.7395,1742,0.574,0.254
200,16384,64,2048,0.7412,18718,10.685,0.197
200,16384,64,2048,0.7412,18687,10.703,0.190
```


## Benchmark using cuvs-bench

### Build the index
```
 ${CONDA_PREFIX}/bin/ann/CUVS_IVF_PQ_ANN_BENCH --build --force --data_prefix=. --benchmark_filter=cuvs_ivf_pq.d64-nlist16K --benchmark_counters_tabular=true --benchmark_out_format=csv --benchmark_out=res.cvs --override_kv=dataset_memory_type:"mmap"  blobs_256_100M_large_k.json
```

## Search
```
/home/scratch.tfeher_gpu_2/miniforge3/envs/faiss_test_n2/bin/ann/CUVS_IVF_PQ_ANN_BENCH --search --data_prefix=. --benchmark_filter=cuvs_ivf_pq.d64-nlist16K --benchmark_counters_tabular=true --benchmark_out_format=csv --benchmark_out=res.cvs --override_kv=dataset_memory_type:"mmap" --override_kv=n_queries:1 /home/scratch.tfeher_gpu_2/bugs/faiss/blobs_256_100M_large_k.json
```

Expected output
```
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Benchmark                                                  Time             CPU   Iterations        GPU    Latency     Recall end_to_end items_per_second          k  n_queries     nprobe refine_ratio total_queries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
cuvs_ivf_pq.d64-nlist16K/0/process_time/real_time      0.196 ms        0.196 ms         3559   185.877u   195.635u   0.741148   0.696264       5.11159k/s     2.048k          1         20            1        3.559k dataset_memory_type="mmap"#internalDistanceDtype="half"#smemLutDtype="half"
cuvs_ivf_pq.d64-nlist16K/1/process_time/real_time      0.233 ms        0.233 ms         2998   223.561u    233.43u   0.743556   0.699822       4.28396k/s     2.048k          1         40            1        2.998k dataset_memory_type="mmap"#internalDistanceDtype="half"#smemLutDtype="half"
cuvs_ivf_pq.d64-nlist16K/2/process_time/real_time      0.259 ms        0.259 ms         2706   249.067u   259.041u   0.744379   0.700966        3.8604k/s     2.048k          1         64            1        2.706k dataset_memory_type="mmap"#internalDistanceDtype="half"#smemLutDtype="half"
cuvs_ivf_pq.d64-nlist16K/3/process_time/real_time      0.325 ms        0.325 ms         2150   315.236u   325.258u   0.741794   0.699306       3.07449k/s     2.048k          1        100            1         2.15k dataset_memory_type="mmap"#internalDistanceDtype="half"#smemLutDtype="half"
```


