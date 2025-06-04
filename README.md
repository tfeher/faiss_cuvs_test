# FAISS-cuVS synthetic benchmarks


## Requirements

### Install FAISS with cuVS backend
```
 conda create -n faiss_test -c pytorch -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia -c conda-forge pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version>=12.0,<=12.5' scikit-learn
 conda activate faiss_test
 conda install -c rapidsai -c conda-forge -c nvidia cuvs-bench=25.04
 ```

### Get benchmark scripts
```
git clone https://github.com/tfeher/faiss_cuvs_test.git
```

## Create dataset

```
cd faiss_cuvs_test
python generate_dataset.py blobs_100m.fbin -N 100000000 -D 256
python generate_groundtruth.py blobs_100m.fbin --queries random-choice -k 4096 --metric euclidean --dtype=float32 --output blobs_256_100M

```
 ## Run test

 ### Build and Search

 ```
 python faiss_test.py -N 100000000 -D 256 -C 16384 blobs.100m.fbin blobs_256_100M/queries.fbin blobs_256_100M/groundtruth.neighbors.ibin
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
python faiss_test.py -N 100000000 -D 256 -C 16384 blobs.100m.fbin blobs_256_100M/queries.fbin blobs_256_100M/groundtruth.neighbors.ibin --read_index --batch_size 1,1,200,200
```
Expected output
```
python faiss_test.py -N 100000000 -D 256 -C 16384 blobs.100m.fbin blobs_256_100M/queries.fbin blobs_256_100M/groundtruth.neighbors.ibin --read_index --batch_size 1,1,200,200
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

