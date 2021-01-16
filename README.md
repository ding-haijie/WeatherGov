
# WeatherGov

TO TEST GENERALIZATION ABILITY FOR SAN_D2T.

The repository for SAN_D2T is at [SAN_D2T](https://github.com/dinghaij/SAN_D2T).

## Dependencies

* [PyTorch](https://pytorch.org) - Models, computational graphs are built with Pytorch.
* [Numpy](https://numpy.org) - Numpy provides the most-frequently-used operations for tensors.
* [Matplotlib](https://matplotlib.org/) - Matplotlib provides toolkits for visualizations in Python.
* [NLTK](https://www.nltk.org/) - Natural Language Toolkit, BLEU-Score calculator needed.

## Instructions

### Prepare Data

Download dataset: [WeatherGov](https://cs.stanford.edu/~pliang/data/weather-data.zip) released by Percy Liang.

Create a folder named 'data', then copy original data files to it.

Run ***data_preprocess/preprocess.py*** to process original & separated files to one pickle file into 'data' folder.

### Train Model

Run ***train_models.py*** to continue training/save model, and evaluate the pre-trained models with metrics **BLEU**.

Run ***test_model.py*** to generate utterances and examine the pre-trained models.
