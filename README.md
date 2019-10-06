# Structured Prediction Helps 3D Human Motion Modelling 
Code repository for our paper presented at ICCV '19.

## Preparing the Data
Download the data from the [DIP website](http://dip.is.tue.mpg.de/) and unzip it into a folder of your choice. Let's call that folder `DIP_DATA_IN`. Create a folder where you want to store the processed data, `DIP_DATA_OUT`. In the folder [`preprocessing`](./preprocessing), run the script

```
python preprocess_dip.py --input_dir DIP_DATA_IN --output_dir DIP_DATA_OUT
```

By default the script generates the data using rotation matrix representations. If you want to convert the data to angle-axis or quaternions, use the `--as_aa` or `--as_quat` flags.

This script creates the training, validation and test splits used to produce the results in the paper. Note that data split is deterministic and determined by the files `training_fnames.txt`, `validation_fnames.txt`, and `test_fnames.txt` under [`preprocessing`](./preprocessing).

When running the script it creates two versions of the validation and test split: One where we split each motion sequence into subsequences of size 180 (3 seconds) using a sliding window and one where we do not split the sequence (referred to as `dynamic` split). When we load data during training or evaluation, we always only extract one window of size `W` from each sequence. Hence, the splitting with a sliding window ''blows up'' the number of samples. Thus, the dynamic split has effectively less samples, which is sometimes convenient (for debugging, visualization etc.).

A note on the data: The data published on the DIP website is an early version of the official AMASS dataset. When we submitted the paper, the official [AMASS dataset](https://amass.is.tue.mpg.de/) was not published yet. We are planning to evaluate our model and baseline models on the official AMASS dataset and report results here. 
If you plan to use the latest version of AMASS, we are happy to provide assistance if required. However, it shouldn't be too hard to adapt `preprocess_dip.py` to parse the AMASS data. 

## Citation
If you use code from this repository, please cite 

```
@inproceedings{Aksan_2019_ICCV,
  title={Structured Prediction Helps 3D Human Motion Modelling},
  author={Aksan, Emre and Kaufmann, Manuel and Hilliges, Otmar},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  month={Oct},
  year={2019},
  note={First two authors contributed equally.}
}
```

If you use data from DIP or AMASS, please cite the original papers as detailed on their website.

## Contact
Please file an issue or contact [Emre Aksan (emre.aksan@inf.ethz.ch)](mailto:emre.aksan@inf.ethz.ch) or [Manuel Kaufmann (manuel.kaufmann@inf.ethz.ch)](mailto:manuel.kaufmann@inf.ethz.ch)