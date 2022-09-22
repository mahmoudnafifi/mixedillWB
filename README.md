# Auto White-Balance Correction for Mixed-Illuminant Scenes

[Mahmoud Afifi](https://sites.google.com/view/mafifi), 
[Marcus A. Brubaker](https://mbrubake.github.io/), 
and [Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)

York University  &nbsp;&nbsp; 

[Video](https://www.youtube.com/watch?v=bhMdH0ZY51s)





Reference code for the paper [Auto White-Balance Correction for Mixed-Illuminant Scenes.](https://arxiv.org/abs/2109.08750) Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. If you use this code or our dataset, please cite our paper:
```
@inproceedings{afifi2022awb,
  title={Auto White-Balance Correction for Mixed-Illuminant Scenes},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2022}
}
```


![teaser](https://user-images.githubusercontent.com/37669469/129296945-ae85e148-ff4c-4e94-8887-0313a477e3e4.jpg)




The vast majority of white-balance algorithms assume a single light source illuminates the scene; however, real scenes often have mixed lighting conditions. Our method presents an effective auto white-balance method to deal with such mixed-illuminant scenes. A unique departure from conventional auto white balance, our method does not require illuminant estimation, as is the case in traditional camera auto white-balance modules. Instead, our method proposes to render the captured scene with a small set of predefined white-balance settings. Given this set of small rendered images, our method learns to estimate weighting maps that are used to blend the rendered images to generate the final corrected image. 


![method](https://user-images.githubusercontent.com/37669469/129645729-470a7eee-a541-4d94-b44c-620774b52bc4.jpg)



Our method was built on top of the modified camera ISP proposed [here](https://github.com/mahmoudnafifi/ColorTempTuning). This repo provides the source code of our deep network proposed in our [paper](https://arxiv.org/abs/2109.08750). 

## Code


### Training

To start training, you should first download the [Rendered WB dataset](https://github.com/mahmoudnafifi/WB_sRGB/), which includes ~65K sRGB images rendered with different color temperatures. Each image in this dataset has the corresponding ground-truth sRGB image that was rendered with an accurate white-balance correction. From this dataset, we selected 9,200 training images that were rendered with the "camera standard" photofinishing and with the following white-balance settings: tungsten (or incandescent), fluorescent, daylight, cloudy, and shade. To get this set, you need to only use images ends with the following parts: `_T_CS.png`, `_F_CS.png`, `_D_CS.png`, `_C_CS.png`, `_S_CS.png` and their associated ground-truth image (that ends with `_G_AS.png`). 

Copy all training input images to `./data/images` and copy all ground truth images to `./data/ground truth images`. Note that if you are going to train on a subset of these white-balance settings (e.g., tungsten, daylight, and shade), there is no need to have the additional white-balance settings in your training image directory. 

Then, run the following command:

`python train.py --wb-settings <WB SETTING 1> <WB SETTING 2> ... <WB SETTING N> --model-name <MODEL NAME> --patch-size <TRAINING PATCH SIZE> --batch-size <MINI BATCH SIZE> --gpu <GPU NUMBER>`

where, `WB SETTING i` should be one of the following settings: `T`, `F`, `D`, `C`, `S`, which refer to tungsten, fluorescent, daylight, cloudy, and shade, respectively. Note that daylight (`D`) should be one of the white-balance settings. For instance, to train a model using tungsten and shade white-balance settings + daylight white balance, which is the fixed setting for the high-resolution image (as described in the [paper](https://arxiv.org/abs/2109.08750)), you can use this command:

`python train.py --wb-settings T D S --model-name <MODEL NAME>`

### Testing

Our pre-trained models are provided in [./models](https://github.com/mahmoudnafifi/mixedillWB/tree/main/models). To test a pre-trained model, use the following command:

`python test.py --wb-settings <WB SETTING 1> <WB SETTING 2> ... <WB SETTING N> --model-name <MODEL NAME> --testing-dir <TEST IMAGE DIRECTORY> --outdir <RESULT DIRECTORY> --gpu <GPU NUMBER>`

As mentioned in the paper, we apply ensembling and edge-aware smoothing (EAS) to the generated weights. To use ensembling, use `--multi-scale True`. To use EAS, use `--post-process True`. Shown below is a qualitative comparison of our results with and without the ensembling and EAS.


![weights_ablation](https://user-images.githubusercontent.com/37669469/129297902-a6b60667-d99b-4937-9c73-a58fc71378d9.jpg)


Experimentally, we found that when ensembling is used it is recommended to use an image size of 384, while when it is not used, 128x128 or 256x256 give the best results. To control the size of input images at inference time, use `--target-size`. For instance, to set the target size to 256, use `--target-size 256`. 

## Network

Our network has a [GridNet](https://arxiv.org/pdf/1707.07958.pdf)-like architecture. Our network consists of six columns and four rows. As shown in the figure below, our network includes three main units, which are: the residual unit (shown in blue), the downsampling unit (shown in green), and the upsampling unit (shown in yellow). If you are looking for the Pythorch implementation of GridNet, you can check [src/gridnet.py](https://github.com/mahmoudnafifi/mixedillWB/blob/main/src/gridnet.py).


#### UPDATE: There is a bug in the decoder forward function, it makes the decoder always has a single layer in depth. Please refer to this [issue](https://github.com/mahmoudnafifi/mixedillWB/issues/4) for more details. To fix it, please update the code in lines [149](https://github.com/mahmoudnafifi/mixedillWB/blob/aeee3b8ab16e9d8e8d462e7ad32f0cb1a91b1654/src/gridnet.py#L149)-[150](https://github.com/mahmoudnafifi/mixedillWB/blob/aeee3b8ab16e9d8e8d462e7ad32f0cb1a91b1654/src/gridnet.py#L150) with the following code:
```
        if j == 0:
          x_latent = latent_forward[k]
        x_latent = res_blck(x_latent)
```
#### Thanks [denkorzh](https://github.com/denkorzh) for catching this mistake. 

![net](https://user-images.githubusercontent.com/37669469/129297286-b82441e3-fe02-4900-9b07-3bd0928731d2.jpg)

## Results

Given this set of rendered images, our method learns to produce weighting maps to generate a blend between these rendered images to generate the final corrected image. Shown below are examples of the produced weighting maps.

![weights](https://user-images.githubusercontent.com/37669469/129297900-c5ab58ef-bafa-409d-bdf9-bee66efa5489.jpg)


Qualitative comparisons of our results with the camera auto white-balance correction. In addition, we show the results of applying post-capture white-balance correction by using the [KNN white balance](https://github.com/mahmoudnafifi/WB_sRGB/) and [deep white balance](https://github.com/mahmoudnafifi/Deep_White_Balance).

![qualitative_5k_dataset](https://user-images.githubusercontent.com/37669469/129297898-b33ae6f9-db8f-4750-b8f9-2de00ee809ad.jpg)


Our method has the limitation of requiring a modification to an ISP to render the additional small images with our predefined set of white-balance settings. To process images that have already been rendered by the camera (e.g., JPEG images), we can employ one of the sRGB white-balance editing methods to synthetically generate our small images with the target predefined WB set in post-capture time. 

In the shown figure below, we illustrate this idea by employing the [deep white-balance editing](https://github.com/mahmoudnafifi/Deep_White_Balance) to generate the small images of a given sRGB camera-rendered image taken from Flickr. As shown, our method produces a better result when comparing to the camera-rendered image (i.e., traditional camera AWB) and the deep WB result for post-capture WB correction. If the input image does not have the associated small images (as described above), the provided source code runs automatically [deep white-balance editing](https://github.com/mahmoudnafifi/Deep_White_Balance) for you to get the small images. 

![qualitative_flickr](https://user-images.githubusercontent.com/37669469/129298104-9ec5186b-092f-4906-a6a4-ca8072b5b1a3.jpg)


## Dataset

![dataset](https://user-images.githubusercontent.com/37669469/129298211-2cbbdc06-915e-4d6e-9a0e-34f910e89512.jpg)

We generated a synthetic testing set to quantitatively evaluate white-balance methods on mixed-illuminant scenes. Our test set consists of 150 images with mixed illuminations. The ground-truth of each image is provided by rendering the same scene with a fixed color temperature used for all light sources in the scene and the camera auto white balance. Ground-truth images end with `_G_AS.png`, while input images ends with `_X_CS.png`, where `X` refers to the white-balance setting used to render each image. 


You can download our test set from one of the following links:
* [8-bit JPG images](https://ln4.sync.com/dl/327ce3f30/jd7rvtf6-7tgz43nf-e9ahtm3j-tv8uzxwe)
* [16-bit PNG images](https://ln4.sync.com/dl/02f0af5f0/4hhpe83r-8ymvskfz-naqpdrqt-nxvq8h4x)



## Acknowledgement
A big thanks to Mohammed Hossam for his help in generating our synthetic test set. 


## Commercial Use
This software and data are provided for research purposes only and CANNOT be used for commercial purposes.


## Related Research Projects
- [C5](https://github.com/mahmoudnafifi/C5): A self-calibration method for cross-camera illuminant estimation (ICCV 2021).
- [Deep White-Balance Editing](https://github.com/mahmoudnafifi/Deep_White_Balance): A multi-task deep learning model for post-capture white-balance correction and editing (CVPR 2020).
- [Interactive White Balancing](https://github.com/mahmoudnafifi/Interactive_WB_correction): A simple method to link the nonlinear white-balance correction to the user's selected colors to allow interactive white-balance manipulation (CIC 2020).
- [White-Balance Augmenter](https://github.com/mahmoudnafifi/WB_color_augmenter): An augmentation technique based on camera WB errors (ICCV 2019).
- [When Color Constancy Goes Wrong](https://github.com/mahmoudnafifi/WB_sRGB): The first work to directly address the problem of incorrectly white-balanced images; requires a small memory overhead and it is fast (CVPR 2019).
- [Color temperature tuning](https://github.com/mahmoudnafifi/ColorTempTuning): A modified camera ISP to allow white-balance editing in post-capture time (CIC 2019).
- [SIIE](https://github.com/mahmoudnafifi/SIIE): A learning-based sensor-independent illumination estimation method (BMVC 2019).


