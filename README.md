# Abstractive Bangla Text Summarizer
An **abstractive text summarizer** built for Bangla language powered by a **Transformer** model.

![Python](https://img.shields.io/badge/Python-20232A?style=for-the-badge&logo=python)
![Tensorflow](https://img.shields.io/badge/TensorFlow-20232A?style=for-the-badge&logo=tensorflow&logoColor=FF6F00)
![Numpy](https://img.shields.io/badge/Numpy-20232A?style=for-the-badge&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-20232A?style=for-the-badge&logo=pandas)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
      <ul>
        <li><a href="#contributors">Contributors</a></li>
        <li><a href="#more-information">More Information</a></li>
      </ul>
    </li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#checkpoint">Checkpoint</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#setup">Setup</a></li>
        <li><a href="#run">Run</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

***

# About
This repository is dedicated to my undergraduate thesis research about **Neural Machine Summarization** for the Bangla language. Previous research endeavors on Bangla text summarization were limited to Recurrent Neural Networks and their many variants. A new breakthrough in Natural Language Processing was brought about in 2017 with the famous paper [Attention is All you Need](https://arxiv.org/abs/1706.03762). My undergraduate thesis was focused on researching this revolutionary paper and creating a transformer model to summarize Bangla text.

## Contributors
This research was made possible with the help of my two teammates [Ahmed Sadman Muhib](https://github.com/ahmedsadman) and [AKM Nahid Hasan](https://github.com/shishir9159). We completed our thesis under the supervision of [Dr. Abu Raihan Mostofa Kamal](https://cse.iutoic-dhaka.edu/profile/raihan-kamal/educations) while being a part of the **Network and Data Analysis** lab.

## More Information
* My [thesis report](https://drive.google.com/file/d/1-mWQxW6n-rojOzdfa2t2KnSjxgu5UJqw/view?usp=sharing) goes into more details about 
my undergraduate thesis reseach. 
* My [thesis defence slide](https://docs.google.com/presentation/d/1FUDkCxxXU61i2e86HaZ4WYdikhz2FYuM/edit?usp=sharing&ouid=102705287918414412487&rtpof=true&sd=true)
highlights the important sections and achievements of my thesis.

<p align="right">(<a href="#abstractive-bangla-text-summarizer">back to top</a>)</p>

***

## Dataset
To train the transformer model we created a text corpus consisting of over 1 million Bangla news articles scraped from the popular news publisher Prothom-Alo. We made the dataset open-source and it can be found at Kaggle called [Prothom Alo News Articles](https://www.kaggle.com/datasets/ishfar/prothom-alo-news-articles). The dataset was mined using [this tool](https://github.com/ahmedsadman/news-scraper) created by my teammate [Ahmed Sadman Muhib](https://github.com/ahmedsadman/news-scraper).

<p align="right">(<a href="#abstractive-bangla-text-summarizer">back to top</a>)</p>

***

# Checkpoint
Saved weights for the transformer model can be found [here](https://drive.google.com/drive/folders/1QN5ZZ6NossaElW6LHR4YVTTwKhmDKB9e?usp=sharing).

***

# Getting Started

## Prerequisites
* Hardware Requirements
    1. NVidia GPU
    2. At least 32GB of RAM
* Software Requirements
    * Operating System: Linux
    * Python: 3.7.7
    * CUDA: 10.1
    * CUDNN: 7.0

**Personal configuration**
* GPU: NVidia RTX 3070 with 8GB VRAM
* RAM: 32GB DDR4 2400MHz
* OS: Manjaro Linux

## Setup
1. Install `python 3.7.7`, `cuda-10.1` and `cudnn-7.0`
2. Clone the repository into local machine.
3. Create a python virtual environment.
    ```bash
    $ python3 -m venv venv
    $ pip install --upgrade pip
    $ pip install -r requirements.txt
    ```

## Run
* Generate TFRecords from csv
    ```bash
    $ python3 src/data_manipulation/create_tfrecords_from_csv.py \
    --input_csv_dir /path/to/csvs \
    --output_dir /path/to/output/dir
    ```

* Train model: Execute `train.ipynb` with appropriate path to tfrecords and output directories.

<p align="right">(<a href="#abstractive-bangla-text-summarizer">back to top</a>)</p>

***

# Contact
For inqueries consider reaching out 
* [Shakleen Ishfar](mailto:shakleenishfar@iut-dhaka.edu)
* [Ahmed Sadman Muhib](mailto:sadmanmuhib@iut-dhaka.edu)
* [AKM Nahid Hasan](mailto:nahidhasan43@iut-dhaka.edu)

<p align="right">(<a href="#abstractive-bangla-text-summarizer">back to top</a>)</p>

***

# Acknowledgments
1. Thank you to my supervisor **Dr. Abu Raihan Mostofa Kamal** for his support and guidance in my thesis work.
2. I'm immensely grateful to the [Rasa](https://www.youtube.com/c/RasaHQ) team for making [this playlist](https://www.youtube.com/playlist?list=PL75e0qA87dlG-za8eLI6t0_Pbxafk-cxb) with such valuable information free for anyone to learn.
3. A huge thank you to [Hugging Face](https://huggingface.co/) for making their implementation of SOTA papers open source.

<p align="right">(<a href="#abstractive-bangla-text-summarizer">back to top</a>)</p>