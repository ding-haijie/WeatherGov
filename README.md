
# WeatherGov

***Test Generalization Ability For [SAN_D2T](https://github.com/ding-haijie/SAN_D2T).***

## Instructions

### Prepare Data

Download dataset: [WeatherGov](https://cs.stanford.edu/~pliang/data/weather-data.zip) released by Percy Liang.

Copy the original data files into a folder named 'data'.

Run `python data_preprocess/preprocess.py` to process original files to one pickle file into 'data' folder.

### Train Model

Run `python train_models.py` to continue training/save model, and evaluate the pre-trained models with metrics **BLEU**.

Run `python test_model.py` to generate utterances and examine the pre-trained models.
