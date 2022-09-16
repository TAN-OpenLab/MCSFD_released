
# MDFEND: Multi-domain Fake News Detection
This is an official implementation for [**Multi-domain consistency supervised feature disentanglement framework for generalizable rumor detection**] which has been accepted by IP&MC.

## Dataset
 The datasets we used in our paper are Pheme, Weibo16 and Weibo18. We provide the link to the original dataset and data processing code. 
 The Pheme dataset can be download from https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619.
 and the Weibo18 dataset is avilable at https://github.com/thunlp/Chinese_Rumor_Dataset, 
 the weibo16 dataset is avalable at https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0.


## Code
### Requirements
python 3.6
pytorch 1.7.1

### Data Preparation
After you download the **Pheme** dataset, run 'raw_process.py' - 'Bert_sentences.py' - 'data_generator.py' - 'data_split.py'.
### Run
You can run this model through:
```python
python main.py 
```
### Reference
