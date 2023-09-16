# Multi Subject Domain Adaptation for Facial Expression Recognition 
# Biovid Pain and Heat Dataset

## Dependencies

### Using conda

```
conda env create -f env.yml
```


### Using pip

```
pip install -r req.txt
```

### Dataset

```
Biovid datasets PartA can be downloaded from here: https://www.nit.ovgu.de/BioVid.html#PubACII17
```

### Training of Source Subjects

```
CUDA_VISIBLE_DEVICES=0 python methods/msbda.py --sub_domains_datasets_path=$BIOVID_DATASET_PATH --sub_domains_label_path=$BIOVID_DATASET_LABEL_PATH --pain_db_root_path=$BIOVID_ROOT_FOLDER_PATH
```
