# Neural Deferred Shading

## [Project Page](https://fraunhoferhhi.github.io/neural-deferred-shading) &nbsp;|&nbsp; [Paper](https://mworchel.github.io/assets/papers/neural_deferred_shading_with_supp.pdf) 

![alt text](docs/static/images/collection_large_bright_small.jpg)

Official code for the CVPR 2022 paper "[Multi-View Mesh Reconstruction with Neural Deferred Shading](https://openaccess.thecvf.com/content/CVPR2022/html/Worchel_Multi-View_Mesh_Reconstruction_With_Neural_Deferred_Shading_CVPR_2022_paper.html)", a method for fast multi-view reconstruction with analysis-by-synthesis.

## Installation

Setup the environment and install basic requirements using conda

```bash
conda env create -f environment.yml
conda activate nds
```

### Nvdiffrast

To install [Nvdiffrast](https://github.com/NVlabs/nvdiffrast) from source, run the following in the main directory:

```bash
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
python -m pip install .
```

### pyremesh

Option 1 (preferred): Install [pyremesh](https://github.com/sgsellan/botsch-kobbelt-remesher-libigl) from pre-built packages in the `pyremesh` subdirectory.

From the main directory, run:

```bash
python -m pip install --no-index --find-links ./ext/pyremesh pyremesh
```

Option 2: Install pyremesh from source.

Follow the instructions at https://github.com/sgsellan/botsch-kobbelt-remesher-libigl.

## Reconstructing DTU Scans

Download the sample data [here](https://www.dropbox.com/s/x5hrx26l1pmz1id/data.zip) and unzip the content into the main directory. For example, after unzipping you should have the directory `./data/65_skull`. This dataset contains two samples but we will release the full dataset soon.

To start the reconstruction for the skull, run:
```bash
python reconstruct.py --input_dir ./data/65_skull/views --input_bbox ./data/65_skull/bbox.txt
```
or for a general scan:
```bash
python reconstruct.py --input_dir ./data/{SCAN-ID}_{SCAN-NAME}/views --input_bbox ./data/{SCAN-ID}_{SCAN-NAME}/bbox.txt
```

You will find the output meshes in the directory `./out/{SCAN-ID}_{SCAN-NAME}/meshes.obj`.

## Reconstructing Custom Scenes

Description of the data format coming soon...

## Citation

If you find this code or our method useful for your academic research, please cite our paper

```bibtex
@InProceedings{worchel:2022:nds,
      author    = {Worchel, Markus and Diaz, Rodrigo and Hu, Weiwen and Schreer, Oliver and Feldmann, Ingo and Eisert, Peter},
      title     = {Multi-View Mesh Reconstruction with Neural Deferred Shading},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2022}
}
```

## Troubleshooting

### CUDA Out of Memory

The reconstruction can be quite heavy on GPU memory and in our experiments we used a GPU with 24 GB.

The memory usage can be reduced by reconstructing with a smaller image resolution. Try passing `--image_scale 2` or `--image_scale 4` to `reconstruct.py`, which uses 1/2th or 1/4th of the original resolution. Expect lower memory consumption and better runtime but degraded reconstruction accuracy.

### Reconstruction Hangs at Remeshing

While the remeshing step can take some time especially at higher mesh resolutions, it sometimes hangs indefinitely. This issue comes from calling the function `remesh_botsch` in the `pyremesh` package, which does not return.

For now, the reconstruction has to be aborted and restarted.