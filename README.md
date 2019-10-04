# Structured Prediction Helps 3D Human Motion Modelling 
Code repository for our paper presented at ICCV '19.

# Preparing the Data
Download the data from the [DIP website](http://dip.is.tue.mpg.de/) and unzip it into a folder of your choice. Let's call that folder `DIP_DATA_IN`. Create a folder where you want to store the processed data, `DIP_DATA_OUT`. In the folder [`preprocessing`](./preprocessing), run the script

```
python preprocess_dip.py --input_dir DIP_DATA_IN --output_dir DIP_DATA_OUT
```

By default the script generates the data using rotation matrix representations. If you want to convert the data to angle-axis or quaternions, use the `--as_aa` or `--as_quat` modifier.

This script produces the training, validation and test data used to produce the results in the paper. Note that data split is deterministic and stored in the files `training_fnames.txt`, `validation_fnames.txt`, and `test_fnames.txt` under [`preprocessing`](./preprocessing).

When running the script it creates two versions of the validation and test split. One where we split each motion sequence into subsequences of size 180 (3 seconds) using a sliding window and one where we do not split sequence (referred to as `dynamic` split). The dynamic split has less samples, which is sometimes convenient (for debugging, visualization etc.).

A note on the data: The data published on the DIP website is an early (and smaller) version of the official AMASS dataset. When we submitted the paper, the official AMASS dataset was not published yet. If you plan to build your own motion prediction models, we highly recommend using the latest version of AMASS, which can be downloaded from [their website](https://amass.is.tue.mpg.de/). It shouldn't be too hard to adapt `preprocess_dip.py` to parse the AMASS data, but we are happy to provide assistance.

# Citation
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

# Contact
Please file an issue or contact [Emre Aksan (emre.aksan@inf.ethz.ch)](mailto:emre.aksan@inf.ethz.ch) or [Manuel Kaufmann (manuel.kaufmann@inf.ethz.ch)](mailto:manuel.kaufmann@inf.ethz.ch)