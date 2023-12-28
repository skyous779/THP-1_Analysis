# The distribution of THP-1 mφs
## Main Work
Firstly, we manually annotated a dataset comprising 100 bright-field images for subsequent neural network training. During the network training process, we opted for the PaddleSeg 2.9 training framework and selected the PP-LiteSeg model as the segmentation network model. Following the completion of training, we conducted model predictions on the unannotated bright-field images to acquire the corresponding bright-field masks for each image.

Subsequently, we utilized the bright-field masks to partition the bright-field images into three distinct regions. Specifically, we initially determined the centroid and major and minor axes of the bright-field masks. The bright-field masks were segmented into three sub-regions in proportion to 1:0.67:0.33. Notably, the centroid of each sub-region coincided with that of the original bright-field mask.

In the process of calculating the average fluorescence intensity for these three sub-regions in the fluorescence images, we initially extracted the values in the G channel of the fluorescence images. The fluorescence images were subjected to element-wise multiplication with each corresponding sub-region mask, resulting in the fluorescence images for the respective sub-regions. Finally, the average fluorescence intensity for each sub-region was computed according to Formula:

$Intensity_i= \frac{sum(img_i)}{sum(mask_i)}$

Where $Intensity_i$, $img_i$ and $mask_i$ respectively represent the average fluorescence intensity of a sub-region, the sub-part of the fluorescence image, and the sub-part of the mask.


## Start
### PaddleSeg
1. Enter the PaddleSeg directory
   ```bash
   cd ./PaddleSeg
   ```
2. Prepare the dataset and config files
   In this EM analysis, the corresponding configuration file path is ./configs/quick_start/bisenet_optic_disc_512x512_1k_EM.yml.

3. Start training
   After preparing the configuration file, execute the following command in the PaddleSeg root directory to train the model on a single GPU using the tools/train.py script.
   
   Note: Commands for model training, evaluation, prediction, and export in PaddleSeg all require execution in the PaddleSeg root directory.
   ```bash
   export CUDA_VISIBLE_DEVICES=0 # Set 1 available card on Linux
   # set CUDA_VISIBLE_DEVICES=0  # Set 1 available card on Windows

   python tools/train.py \
          --config configs/quick_start/bisenet_optic_disc_512x512_1k_EM.yml \
          --save_interval 500 \
          --do_eval \
          --use_vdl \
          --save_dir output
   ```
4. Model prediction

   The predict.py script is specifically used for visualizing predictions, and the command format is as follows.
   ```bash
   python tools/predict.py \
          --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
          --model_path output/best_model/model.pdparams \
          --image_path data/optic_disc_seg/JPEGImages/H0002.jpg \
          --save_dir output/result
   ```
   Where image_path can be a path to an image or a directory. If it is a directory, visual results will be predicted and saved for all images in the directory.

### Distribution calculation of THP-1 macrophages
1. Enter the THP-1_Analysis directory
   ```bash
   cd ./THP-1_Analysis
   ```

2. Prepare the dataset
   You need to prepare the path for the distribution images of THP-1 macrophages in the sphere, as well as the corresponding mask segmentation files (obtained from PaddleSeg).

3. Run the EM analysis script
   ```bash
   python ./THP-1_Analysis.py \
       --THP_1_path THP-1_DIR
       --mask_path THP-1_MASK_DIR
       --save_xlsv_path SAVE_XLSV_PATH
   ```
   Here, THP-1_DIR is the path to the distribution images of THP-1 macrophages in the sphere, mask_path is the path to the corresponding mask segmentation files for THP-1 macrophages, and SAVE_XLSV_PATH is the path to save the xlsv file. The final calculation results will be saved in the xlsv file.

## Citation

```bash
@misc{liu2021paddleseg,
      title={PaddleSeg: A High-Efficient Development Toolkit for Image Segmentation},
      author={Yi Liu and Lutao Chu and Guowei Chen and Zewu Wu and Zeyu Chen and Baohua Lai and Yuying Hao},
      year={2021},
      eprint={2101.06175},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{paddleseg2019,
    title={PaddleSeg, End-to-end image segmentation kit based on PaddlePaddle},
    author={PaddlePaddle Authors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleSeg}},
    year={2019}
}
```





