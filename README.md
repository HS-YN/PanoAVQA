# Pano-AVQA

Official repository of PanoAVQA: Grounded Audio-Visual Question Answering in 360° Videos (ICCV 2021)

![Data_fig](https://raw.githubusercontent.com/HS-YN/PanoAVQA/main/assets/data.png)

### [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Yun_Pano-AVQA_Grounded_Audio-Visual_Question_Answering_on_360deg_Videos_ICCV_2021_paper.html) [[Poster]](https://hs-yn.github.io/assets/pdf/2021iccv_panoavqa_poster.pdf) [Video]


## Getting Started

This code is based on following libraries:

* `python=3.8`
* `pytorch=1.7.0` (with cuda 10.2)

To create virtual environment with all necessary libraries:
 
```bash
conda env create -f environment.yml
```

By default data should be saved under `data/feat/{audio,label,visual}` directory and logs (w/ cache, checkpoint) are saved under `data/{cache,ckpt,log}` directory. Using symbolic link is recommended:

```bash
ln -s {path_to_your_data_directory} data
```

We use single TITAN RTX for training, but GPUs with less memory are still doable with smaller batch size (provided precomputed features).


## Dataset

We plan to release the Pano-AVQA dataset public within this year, including Q&A annotation, precomputed features, etc. Please stay tuned!


## Model

### Training

Default configuration is provided in `code/config.py`. To run with this configuration:

```bash
python cli.py
```

To run with custom configuration, either modify `code/config.py` or execute:

```bash
python cli.py with {{flags_at_your_disposal}}
```

### Inference

Model weight is saved under `./data/log` directory. To run inference only:

```bash
python cli.py eval with ckpt_file=../data/log/{experiment}/{ckpt}.pth
```


## Citation

If you find our work useful in your research, please consider citing:

```tex
@InProceedings{Yun2021PanoAVQA, 
    author = {Yun, Heeseung and Yu, Youngjae and Yang, Wonsuk and Lee, Kangil and Kim, Gunhee},
    title = {Pano-AVQA: Grounded Audio-Visual Question Answering on 360$^\circ$ Videos},
    booktitle = {ICCV},
    year = {2021}
}
```


## Contact

If you have any inquiries, please don't hesitate to contact us via heeseung.yun at vision.snu.ac.kr.
