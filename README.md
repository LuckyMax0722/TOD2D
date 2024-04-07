
## Introduction
In this project, we propose an approach for data preprocessing based on nuImages database.

## Installation
Environment requirements

* Ubuntu 20.04
* Python 3.8
* Pytorch 2.1.2
* CUDA 12.1

The following installation guild suppose ``Ubuntu=20.04`` ``python=3.8`` ``pytorch=2.1.2`` and ``cuda=12.1``. You may change them according to your system, but linux is mandatory.

1. Create a conda virtual environment and activate it.
```
conda create -n TOD2D python=3.8
conda activate TOD2D
```

2. Clone the repository.
```
git clone https://github.com/LuckyMax0722/TOD2D.git
```

3. Install the PyTorch and PyTorch Lightning
```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pytorch-lightning==2.1.4
```

4. Install the dependencies.
```
pip install opencv-python
pip install easydict
pip install matplotlib
pip install pandas
pip install ipython
pip install psutil
pip install seaborn
```

5. Install the visualization
```angular2html
pip install tensorboard
pip install protobuf==3.19.6
```

## Data Preparation
1. First, you need to register/login on DriveU to download the [DriveU Traffic Light Dataset (DTLD)](https://www.uni-ulm.de/in/iui-drive-u/projekte/driveu-traffic-light-dataset/).

The dataset divides the data according to German cities. You can download the data and labels for individual cities, e.g. **DTLD/Berlin**, or the entire dataset **DTLD**.

For detail information, please refer to [DTLD](https://github.com/julimueller/dtld_parsing)

Your folder should look like this:
```
data
├── Berlin
│   ├── Berlin1
│       ├── 2015-04-17_10-50-05  
│           ├── DE_..._k0.tiff
│           ├── DE_..._nativeV2.tiff
│           ├── .......
│       ├── .......
│   ├── Berlin2
├── Bochum
│   ├── .......
├── DTLD_Labels_v2.0
```

2. Before processing the data, please set the base path of the project in ``lib/config.py``.
```angular2html
# Main Path
...
CONF.PATH.BASE = '.../TOD2D'  # TODO: change this
...
CONF.PATH.LABELS = os.path.join(CONF.PATH.DATA, 'DTLD_Labels_v2.0/v2.0/DTLD_all.json') # TODO: change this if use different data
```

3. First you need to use a data converter to convert the DTLD into a Classifier format dataset.
```angular2html
cd TOD2D
python tools/converter_dtld2cls.py
```
Your folder should look like this:
```
dataset_cls
├── dtld_cls
│   ├── images
│       ├── DE_..._k0_0.jpg     
│       ├── DE_..._k0_1.jpg    
│       ├── .......
│   ├── labels
│       ├── dtld_cls.txt     
```

5. We manually divided the labels into the categories shown in the table below:

|        | Circle | Left | Right | Straight | Other |
|:------:|:------:|:----:|:-----:|:--------:|:-----:|
|  Red   |  0,0   | 0,1  |  0,2  |   0,3    |  0,4  |
| Yellow |  1,0   | 1,1  |  1,2  |   1,3    |  1,4  |
| Green  |  2,0   | 2,1  |  2,2  |   2,3    |  2,4  |
|  Off   |        |      |       |          |  3,4  |

