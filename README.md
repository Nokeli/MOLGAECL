# MOLGAECL
The code of MOLGAECL: Molecular Graph Contrastive Learning via Graph Auto-Encoder Pretraining and Fine-Tuning Based on Drug–Drug Interaction Prediction (https://pubs.acs.org/doi/10.1021/acs.jcim.5c00043)
### 前提条件
- Python 3.8+
- [RDKit](https://www.rdkit.org/docs/Install.html)
- 其他依赖：`numpy`, `pandas`, `tqdm`, `joblib`
- 通过pip install -r requirements.txt
### 运行
1. 预训练
python molgaecl.py
2. 微调：
python finetune/2finetune_classify_ddi.py 
