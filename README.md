# Tiny CUDA k-means

This is a one-file fun project that implements k-means clustering on the GPU.
It creates three small 2D clusters, assigns points with a CUDA kernel, updates
the centroids, and logs progress.

## Run in Google Colab

Select **Runtime → Change runtime type → GPU**, upload `simple_kmeans.cu`, and run:

```bash
!nvcc -O2 simple_kmeans.cu -o simple_kmeans
!./simple_kmeans
```

Expected output includes iteration shifts, cluster counts, and final centers.
