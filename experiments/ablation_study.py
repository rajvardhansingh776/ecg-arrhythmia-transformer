import pandas as pd
from models.transformer import ECGTransformer
from models.patch_transformer import PatchECGTransformer

def run_ablation(train_fn,eval_fn):

    experiments=[
        {"name":"baseline_transformer","patch":False},
        {"name":"patch_transformer","patch":True}
    ]

    results=[]

    for exp in experiments:

        if exp["patch"]:
            model=PatchECGTransformer(seq_len=1000,classes=5)
        else:
            model=ECGTransformer(seq_len=1000,classes=5)

        train_fn(model)

        metrics=eval_fn(model)

        results.append({
            "experiment":exp["name"],
            "f1":metrics["f1"]
        })

    df=pd.DataFrame(results)

    df.to_csv("results/tables/ablation_results.csv",index=False)

    return df