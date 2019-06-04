# Deep-Robust-Subjective-Visual-Property-Prediction-in-Crowdsourcing
This is a PyTorch implementation of the model described in our paper:

>Q. Xu, Z. Yang, Y. Jiang, X. Cao, Q. Huang, and Y. Yao. Deep Robust Subjective Visual Property Prediction in Crowdsourcing. CVPR 2019.

## Dependencies
- PyTorch >= 0.4.0
- numpy
- scipy
- Pillow

## Data
We convert the train/test split in the [original LFW10 dataset](http://cvit.iiit.ac.in/images/Projects/relativeParts/LFW10.zip) to our train and test file in the `data/` folder. The label `0` is ignored in the converted version. Please unzip `data/images.zip` before training.

The data format is `(file_1, file_2, label, attr_id, pair_id, strength)`.
- `file_1`/`file_2`: file name of images
- `label`: pairwise comparison label (-1/1)
- `attr_id`: attribute id, starting from 0.
- `pair_id`: pair id given a specific attribute, starting from 0. Note that (i,j,-1) and (i,j,1) are the same pair but different samples for the same attribute.
- `strength`: how many times a label is annotated for a pair given a specific attribute.

For custom dataset, you can write a class that inherits from the base class `PairwiseDataset` in `utils.py` like we do in `LFWDataSet`.

## Train
Here is an example to train the model with logistic loss.
```
python train_gamma_lfw.py --outer=10 --lamda=0.5 -e=5 --loss="logit" --binary_label=True -L=20
```

## Outlier Ratio
We provide a function `get_outlier_num()` in the base class `utils.py/PairwiseDataset` to calculate the true outlier number for the training set. This function can be slightly modified to get the total sample number. Then it is easy to obtain the true outlier ratio.

After the training, the outlier ratio indicated by `gamma` is also calculated. The hyperparameter `lambda` controls the sparsity of `gamma` and thus influences the predicted outlier ratio.

## Citation
Please cite our paper if you use this code in your own work:

```
@inproceedings{xu2019deep,
  title={Deep Robust Subjective Visual Property Prediction in Crowdsourcing},
  author={Xu, Qianqian and Yang, Zhiyong and Jiang, Yangbangyan and Cao, Xiaochun and Huang, Qingming and Yao, Yuan},
  booktitle={IEEE/CVF International Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
