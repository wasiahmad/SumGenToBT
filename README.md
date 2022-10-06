<div align="center">

<h1>Summarize and Generate to Back-translate</h1>

Official code release of our work, [Summarize and Generate to Back-translate: Unsupervised Translation of Programming Languages](https://arxiv.org/abs/2205.11116). 

<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#train">Train</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#license">License</a> • 
  <a href="#citation">Citation</a>
</p>

</div>

## Setup

Setting up a conda environment is recommended to run experiments. 
We assume [anaconda](https://www.anaconda.com/) is installed. The additional requirements 
(noted in [requirements.txt](https://github.com/wasiahmad/SumGenToBT/blob/main/requirements.txt)) can be installed by 
running the following script:

```
bash install_env.sh
```

Then build `tree_sitter` library for Java and Python languages by running:

```
python build.py
```

Finally, download the pre-trained PLBART checkpoints.

```bash
cd plbart
bash download.sh
```

There are two model sizes, so we can perform experiments with **MODEL_SIZE=base|large**.

## Train

### Step1. Summarization and Generation

```bash
cd sumgen
bash run.sh GPU_ID [MODEL_SIZE]
```

### Step2. Back-translation

```bash
cd plbart
bash train.sh GPU_ID [MODEL_SIZE]
```

## Evaluation

#### Evaluate SumGen model

```bash
cd sumgen/evaluation
bash decode.sh GPU_ID SOURCE TARGET MODEL_SIZE BEAM_SIZE
bash evaluate.sh SAVE_DIR SOURCE TARGET
```

For example, run the following commands to get results with default settings.

```bash
cd sumgen/evaluation
# to evaluate base model
bash decode.sh 0 java python base 10
bash evaluate.sh base_java_python_b10 java python
# to evaluate large model
bash decode.sh 0 java python large 10
bash evaluate.sh large_java_python_b10 java python
```

#### Evaluate PLBART

```bash
cd scripts
bash run.sh GPU_ID
```

### Results



## License

Contents of this repository is under the [MIT license](https://opensource.org/licenses/MIT). The license 
applies to the pre-trained and fine-tuned models as well.


## Citation

If you use any of the datasets, models or code modules, please cite the following paper:

```
@article{ahmad2022sumgen,
  author    = {Wasi Uddin Ahmad and Saikat Chakraborty and Baishakhi Ray and Kai-Wei Chang},
  title     = {Summarize and Generate to Back-translate: Unsupervised Translation of Programming Languages},
  journal   = {CoRR},
  volume    = {abs/2205.11116},
  year      = {2022},
  url       = {https://arxiv.org/abs/2205.11116},
  eprinttype = {arXiv},
  eprint    = {2205.11116}
}
```



