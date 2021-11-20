# Feature-Normalization

## Introduction

### Requirements
This codebase is tested with:  
1.  h5py==3.1.0
2.  joypy==0.2.5
3.  matplotlib==3.4.2
4.  numpy==1.21.0
5.  pandas==1.2.3
6.  Pillow==8.4.0
7.  scikit_learn==1.0.1
8.  scipy==1.6.0
9.  seaborn==0.11.2
10. torch==1.8.1
11. torchvision==0.9.1
12. tqdm==4.60.0

To install all requirements, use "pip install -r requirements.txt"

## Running Experiments 
### Dataset Preparation
**MiniImageNet and CD-FSL:** Download the datasets for CD-FSL benchmark following step 1 and step 2 here: https://github.com/IBM/cdfsl-benchmark.
**Set datasets path:** Set the appropriate dataset pathes in "configs.py".

**Source dataset names:** "ImageNet", "miniImageNet"

**Target dataset names:** "EuroSAT", "CropDisease", "ChestX", "ISIC", "ImageNet_test"

|               | Baseline BN                                                                                                                                                                                                                       |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | python baseline.py --dir ./logs/baseline --bsize 128 --epochs 1000 --model resnet10                                                                                                                                             |
| Ro Fine-Tune: | python finetune.py --save_dir ./logs/baseline --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/baseline/checkpoint_best.pkl --freeze_backbone  |
|               | [Pre-Trained Dictionary](https://github.com/MosyMosy/FN_Model_Zoo/blob/main/Dictionaries/baseline/checkpoint_best.pkl)                                                                                                          |

|               | Baseline FN                                                                                                                                                                                                                      |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | python baseline_na.py --dir ./logs/baseline_na --bsize 128 --epochs 1000 --model resnet10                                                                                                                                             |
| Ro Fine-Tune: | python finetune.py --save_dir ./logs/baseline_na --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/baseline_na/checkpoint_best.pkl --freeze_backbone  |
|               | [Pre-Trained Dictionary](https://github.com/MosyMosy/FN_Model_Zoo/blob/main/Dictionaries/baseline_na/checkpoint_best.pkl)                                                                                                          |

|               | AdaBN BN                                                                                                                                                                                                                      |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | python AdaBN.py --dir ./logs/AdaBN/{dataset Name} --base_dictionary logs/baseline/checkpoint_best.pkl --target_dataset $target_testset --target_subset_split datasets/split_seed_1/$target_testset\_unlabeled_20.csv --bsize 256 --epochs 10 --model resnet10 &                                                                                                                                             |
| Ro Fine-Tune: | python finetune.py --save_dir ./logs/AdaBN/{Target dataset name} --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/AdaBN/{Target dataset name}/checkpoint_best.pkl --freeze_backbone  |
|               | Pre-Trained Dictionary: https://github.com/MosyMosy/FN_Model_Zoo/blob/main/Dictionaries/AdaBN/{Target dataset name}/checkpoint_best.pkl                                                                                                        |

|               | AdaBN FN                                                                                                                                                                                                                      |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | python AdaBN_na.py --dir ./logs/AdaBN_na/{dataset Name} --base_dictionary logs/baseline_na/checkpoint_best.pkl --target_dataset $target_testset --target_subset_split datasets/split_seed_1/$target_testset\_unlabeled_20.csv --bsize 256 --epochs 10 --model resnet10                                                                                                                                             |
| Ro Fine-Tune: | python finetune.py --save_dir ./logs/AdaBN_na/{Target dataset name} --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/AdaBN_na/{Target dataset name}/checkpoint_best.pkl --freeze_backbone  |
|               | Pre-Trained Dictionary: https://github.com/MosyMosy/FN_Model_Zoo/blob/main/Dictionaries/AdaBN/{Target dataset name}/checkpoint_best.pkl                                                                                                        |
