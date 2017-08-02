# dyn_pose
Training and inference code for [LSTM-based action recognition model](http://www.contrib.andrew.cmu.edu/~gengshay/wordpress/index.php/424-2/).
Online version to be released later.

## Steps
- pre-process data: preprocess.py / split.py
- extract pose: extract.py
- train model: train_lstm.py
- evaluate model: infer_lstm.py
- offline demo: example.py

## Notes
- modify label_num in model/lstm.config

## Utils
- display.ipynb: display pose estimation results
- run.sh: evaluation script
- test.ipynb: miscellaneous

## TODO
- reduce spatial net size
