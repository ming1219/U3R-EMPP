import os

# command = "pip install -r req.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
# result = os.system(command)
# result = os.system("pip install -U bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple")


import pandas as pd
from unimol_tools.predict import MolPredict  # 导入预测类


MODEL_PATH = './multi_model'

predictor = MolPredict(load_model=MODEL_PATH)

test_data_path = "test_set.csv"

test_df = pd.read_csv(test_data_path)
required_columns = ['smiles', 'density', 'DetoD', 'DetoP', 'DetoQ', 'DetoT', 'DetoV', 'HOF_S']
if not all(col in test_df.columns for col in required_columns):
    print(f"错误：测试集必须包含以下列: {required_columns}")
    print(f"当前列名: {test_df.columns.tolist()}")
    exit(1)

try:
    print("开始预测...")
    target_cols = [c for c in required_columns if c != 'smiles']
    if test_df[target_cols].isna().all().all():
        metrics_arg = None
    else:
        metrics_arg = "mse,mae,r2"
    results = predictor.predict(
        data=test_data_path,
        save_path='./predictions',
        metrics=metrics_arg
    )
    print("预测完成！结果保存在 ./predictions 目录")

except Exception as e:
    print(f"预测出错: {str(e)}")