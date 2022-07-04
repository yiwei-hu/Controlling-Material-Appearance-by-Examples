# Controlling Material Appearance by Examples

A by-example material appearance transfer method
![teaser](teaser.jpg)

Yiwei Hu, Miloš Hašan, Paul Guerrero, Holly Rushmeier, Valentin Deschaintre

In Computer Graphics Forum (Proc. of Eurographics Symposium on Rendering 2022). [[Project page]](https://yiweihu.netlify.app/project/hu2022control/)

---
## Installation
```
conda create -n mat_transfer python=3.8
conda activate mat_transfer
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install ninja kornia imageio imageio-ffmpeg opencv matplotlib -c conda-forge
```
## Usage
Please download pretrained models here and put models into `./pretrained`. To run the code, see main.py for details. 
We include some examples in `./sample`

## Citation
```
@article {hu2022control,
	title = {Controlling Material Appearance by Examples},
	author = {Hu, Yiwei and Hašan, Miloš and Guerrero, Paul and Rushmeier, Holly and Deschaintre, Valentin},
	journal = {Computer Graphics Forum (Proc. of Eurographics Symposium on Rendering 2022)},
	volume = {41},
	number = {4},
	year = {2022},
}
```

## Contact
If you have any question, feel free to contact yiwei.hu@yale.edu