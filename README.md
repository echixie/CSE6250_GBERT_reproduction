# CSE6250 G-Bert Reproduction
Reproduce G-Bert model for Medication Recommendation

## Intro
Medication recommendation is a big and important machine learning application on healthcare
area. However, most existing models reply on a small number of patient’s historical electronic
health records(EHRs) to train the model and make predication. A large number of patients with
only single visit records are excluded from model training. To address this challenge, the original
paper brings the language model pre-training schema into the healthcare area for the first time. It
introduce a new model called G-Bert, which combines the power of graph neural network for
medical code representation and BERT(Bidirectional Encoder Representations from Transformers)
for EHR representation. In this project challenge, we reproduced this innovative model and verify if
it can achieve state-of-the-art performance on the medication recommendation task.
## Requirements
- pytorch==3.7.13
- python==1.11.0

For other dependency requrements, refer to environment.yml

## Structure
We list the main part of this repo as follows:
```latex
.
├──  code/
│   ├──  gnn_model.py % GNN Model
│   ├──  modeling_gbert.py % Basic GBert model
│   ├──  modeling_gbert_pretrain.py % GBert pretrain model
│   ├──  modeling_gbert_downstream.py % GBert downstream model/ medication predict
│   ├──  run_gbert_pretraining.py % running script for Gbert pretraining
│   ├──  run_gbert_predict.py % running script for Gbert medication predict
├──  data/
│   ├──  etl.py  etl script
│   ├──  data-multi-visit.pkl % multi-visit record
│   ├──  data-single-visit.pkl % singe-visit record
│   ├──  dx-vocab.txt % diagnosis codes vocabulary
│   ├──  rx-vocab.txt % medication codes vocabulary
│──  saved/
│   ├──  GBert-predict % saved GBERT model
├──  environment.yml % python dependency
```
### Dataset description
data/etl.py is the script to process data using raw files from MIMIC-III dataset. You need request and  download data files from [MIMIC](https://mimic.physionet.org/gettingstarted/dbsetup/) and get necessary mapping files from [GAMENet](https://github.com/sjy1203/GAMENet).
This is the files you need and put under dir data/raw/:

- PRESCRIPTIONS.csv, DIAGNOSES_ICD.csv 
- ndc2atc_level4.csv, Data_final.pkl, Ndc2rxnorm_mapping.txt 

### Guide
we reproduced four models in this experiments, this is how you can train them from stratch:

- GBERT
```bash
cd code/
python run_gbert_pretraining.py --model_name GBert-predict --do_train --graph
python run_gbert_predict.py --model_name GBert-predict --use_pretrain --pretrain_dir ../saved/GBert-predict --do_train --graph
```
You can also validate this model with our saved model file:
```bash
cd code/
python run_gbert_predict.py --model_name GBert-predict --use_pretrain --pretrain_dir ../saved/GBert-predict  --graph
```

- G-BERTp-(GBERT model without pretrain)
```bash
cd code/
python run_gbert_predict.py --model_name GBert-predict --do_train --graph
```

- G-BERTg-(GBERT model without GNN embedding)
```bash
cd code/
python run_gbert_pretraining.py --model_name GBert-predict --do_train 
python run_gbert_predict.py --model_name GBert-predict --use_pretrain --pretrain_dir ../saved/GBert-predict --do_train
```

- G-BERTp-g- (GBERT model without pretrain and GNN embedding)
```bash
cd code/
python run_gbert_predict.py --model_name GBert-predict --do_train
```





