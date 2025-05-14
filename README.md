Dataset 
1. Bean (Healthy, Angular leaf spot, Rust)    https://www.kaggle.com/datasets/therealoise/bean-disease-dataset
2. Strawberry (Healthy, Angular leaf spot)  https://www.kaggle.com/datasets/nirmalsankalana/plant-diseases-training-dataset?select=data (healthy), https://www.kaggle.com/datasets/caozhihao/strawberry-disease-data (angular leaf spot)
3. Coffee (Healthy, Rust)  https://www.kaggle.com/datasets/nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset

Segmented Dataset
https://drive.google.com/drive/folders/1wKFDDZOx-tbPDjsfV8Y9Txnqql6hGb6L?usp=sharing




# Augmentation Strategies for Plant Disease Classification

Abstract: *This study explores the effectiveness of various data augmentation strategies for enhancing plant disease classification using the LeafGAN model. We propose a novel approach that integrates leaf region and disease symptom masking to improve the quality of synthetic images and, consequently, the performance of plant disease models. Three different configurations of the LeafGAN model were tested, with each model applying distinct masking techniques: Model 1 uses basic LeafGAN outputs, Model 2 applies leaf region masking to isolate the leaf from the background, and Model 3 combines both leaf region and disease symptom masking for enhanced disease simulation. The models were evaluated based on their ability to generate realistic disease progression and recovery images, which were then used for data augmentation. Results show that Model 3, incorporating both masking strategies, produced the most realistic and high-fidelity images, leading to superior augmentation quality. These findings highlight the potential of advanced data augmentation strategies in improving plant disease simulation, emphasizing the importance of targeted feature masking in enhancing the generalization and robustness of disease classification models in agricultural applications.*


## LFLSeg module
Tutorial of how to create dataset and train the LFLSeg module is available in the [LFLSeg](https://github.com/IyatomiLab/LeafGAN/tree/master/LFLSeg)

## YOLOv5 model
Tutorial of how to train the YOLOv5 model and get the segmentation result. [YOLOv5](https://github.com/ultralytics/yolov5)

## Datasets
- original dataset:
    - Bean Leaf  [Healthy](https://www.kaggle.com/datasets/therealoise/bean-disease-dataset) ,
                 [Angular Leaf Spot](https://www.kaggle.com/datasets/therealoise/bean-disease-dataset)
    - Strawberry Leaf [Healthy](https://universe.roboflow.com/university-of-cordilleras/strawberryleafdisease-no-other/browse?queryText=class%3ALeafSpot&pageSize=50&startingIndex=0&browseQuery=true) , 
                      [Angular Leaf Spot](https://www.kaggle.com/datasets/caozhihao/strawberry-disease-data)
- Segmented dataset:
- Diseased symptoms annotate though [Robotflow](https://app.roboflow.com/yolov5plantdoc/disease-region/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

```bash
/path/to/healthy2brownspot/trainA
/path/to/healthy2brownspot/testA
/path/to/healthy2brownspot/trainB
/path/to/healthy2brownspot/testB
```
- Masked dataset: This dataset is normal dataset + pre-generated mask images. First, you need to generate your own mask images using the [prepare_mask.py](https://github.com/IyatomiLab/LeafGAN/blob/master/prepare_mask.py). An example of the masked dataset named `healthy2brownspot_mask`
```bash
/path/to/healthy2ALS/trainA
/path/to/healthy2ALS/trainA_mask # mask images of trainA
/path/to/healthy2ALS/testA
/path/to/healthy2ALS/trainB
/path/to/healthy2ALS/trainB_mask # mask images of trainB
/path/to/healthy2ALS/testB
```
## LeafGAN/CycleGAN train/test
- Make sure to prepare the dataset first
- Train a model (example with the dataset `healthy2brownspot`):
```bash
python train.py --dataroot /path/to/healthy2brownspot --name healthy2brownspot_leafGAN --model leaf_gan
```
- Train model with mask images (example with the dataset `healthy2brownspot_mask`):
```bash
python train.py --dataroot /path/to/healthy2brownspot --name healthy2brownspot_leafGAN --model leaf_gan --dataset_mode unaligned_masked
```
To see more intermediate results, check out `./checkpoints/healthy2brownspot_leafGAN/web/index.html`.
- Test the model:
```bash
python test.py --dataroot /path/to/healthy2brownspot --name healthy2brownspot_leafGAN --model leaf_gan
```
- The test results will be saved to a html file here: `./results/healthy2brownspot_leafGAN/latest_test/index.html`.

## Citation

```
@article{cap2020leafgan,
  title   = {LeafGAN: An Effective Data Augmentation Method for Practical Plant Disease Diagnosis},
  author  = {Quan Huu Cap and Hiroyuki Uga and Satoshi Kagiwada and Hitoshi Iyatomi},
  journal = {IEEE Transactions on Automation Science and Engineering},
  year    = {2020},
  doi     = {10.1109/TASE.2020.3041499}
}
```

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
