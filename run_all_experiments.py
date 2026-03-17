import os
import subprocess
import sys

def run(cmd):

    print("\n--------------------------------------------------")
    print("Running:",cmd)
    print("--------------------------------------------------\n")

    result=subprocess.run(cmd,shell=True)

    if result.returncode!=0:
        print("Error while running:",cmd)
        sys.exit(1)


def main():

    print("\n========================================")
    print("ECG ARRHYTHMIA TRANSFORMER PIPELINE")
    print("========================================\n")

    # STEP 1
    print("STEP 1: Setup environment and download datasets")

    if os.path.exists("setup.sh"):
        run("bash setup.sh")
    else:
        print("setup.sh not found, skipping setup")


    # STEP 2
    print("\nSTEP 2: Preprocessing ECG datasets")
    run("python preprocessing/preprocess_dataset.py")


    # STEP 3
    print("\nSTEP 3: Self-Supervised Pretraining")
    run("python training/pretrain_ssl.py")


    # STEP 4
    print("\nSTEP 4: Training Model")
    run("python training/train.py")


    # STEP 5
    print("\nSTEP 5: Cross Dataset Evaluation")
    run("python experiments/cross_dataset.py")


    # STEP 6
    print("\nSTEP 6: Robustness Testing")
    run("python evaluation/robustness.py")


    # STEP 7
    print("\nSTEP 7: Ablation Study")
    run("python experiments/ablation_study.py")


    # STEP 8
    print("\nSTEP 8: Explainability (SHAP)")
    run("python explainability/shap_explain.py")


    # STEP 9
    print("\nSTEP 9: Attention Visualization")
    run("python visualization/attention_maps.py")


    print("\n========================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("========================================")

    print("\nResults saved in:")
    print("results/figures/")
    print("results/tables/")


if __name__=="__main__":
    main()