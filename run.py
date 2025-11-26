import os

# command = "pip install -r req.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
# result = os.system(command)
# result = os.system("pip install -U bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple")

from unimol_tools import MolTrain

# 配置训练参数
trainer = MolTrain(
    task='multilabel_regression',
    data_type='molecule',
    epochs=300,
    learning_rate=1e-4,
    batch_size=64,
    save_path='./multi_model',
    metrics="mse,mae,r2",
    kfold=10,
    split='scaffold',
    target_normalize='standard',
    smiles_col='smiles',
    target_cols='density,DetoD,DetoP,DetoQ,DetoT,DetoV,HOF_S'
)

data_path = "train_set.csv"
trainer.fit(data_path)

print("训练完成！模型保存在 ./multi_model")