import os
import json
import shutil
import argparse
import subprocess

nnUNet_raw = os.environ.get("nnUNet_raw")
nnUNet_preprocessed = os.environ.get("nnUNet_preprocessed")
nnUNet_results = os.environ.get("nnUNet_results")


def dataset_json(dataset_name):
    cases = set()
    modalities = set()
    for file in os.listdir(os.path.join(nnUNet_raw, dataset_name, "imagesTr")):
        if file.endswith(".nii.gz"):
            cases.add(file[:-12])  # _0000.nii.gz
            modalities.add(file[-11:-7])  # 0000
    print(cases, modalities)

    if len(modalities) == 1:
        channel_names = {"0": "abdCT"}
    elif len(modalities) == 3:
        channel_names = {"0": "abdCT", "1": "vessel", "2": "distance"}

    dataset = {
        "channel_names": channel_names,
        "labels": {"background": 0, "lymphnode": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    with open(os.path.join(nnUNet_raw, dataset_name, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=4)


def nnUNetPlans_json(dataset_name):
    shutil.copy(
        os.path.join(nnUNet_preprocessed, dataset_name, "nnUNetPlans.json"),
        os.path.join(nnUNet_preprocessed, dataset_name, "nnUNetPlans_old.json"),
    )

    with open(os.path.join(nnUNet_preprocessed, dataset_name, "nnUNetPlans.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    data["configurations"]["3d_fullres_vesseldecattn"] = {
        "inherits_from": "3d_fullres",
        "UNet_class_name": "VesselDecAttnUNet",
    }
    with open(os.path.join(nnUNet_preprocessed, dataset_name, "nnUNetPlans.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def split_final_json(dataset_name):
    with open("data_split.json", "r") as file:
        data = json.load(file)["idx"]

    train, val = [], []
    for file in sorted(
        os.listdir(os.path.join(nnUNet_raw, dataset_name, "labelsTr")),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    ):
        case = file.replace(".nii.gz", "")
        if int(case.split("_")[-1]) in data["private_va_idx"]:
            val.append(case)
        else:
            train.append(case)

    splits = [{"train": train, "val": val}]
    with open(os.path.join(nnUNet_preprocessed, dataset_name, "splits_final.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="", help="dataset id")
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--ref_id", type=str, default="", help="ref dataset id")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--config", type=str, default="3d_fullres")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer_oversample")
    args = parser.parse_args()

    dataset_id = args.id
    ref_dataset_id = args.ref_id
    dataset_name = "Dataset%s_LNM_LYMPH" % dataset_id
    ref_dataset_name = "Dataset%s_LNM_LYMPH" % ref_dataset_id

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "dataset":
        dataset_json(dataset_name)

    elif args.mode == "preprocess":
        subprocess.run(["nnunetv2_lnm_extract_fingerprint", "-d", dataset_id], check=True)
        
        if dataset_id in ["310", "312"]:
            subprocess.run(["nnunetv2_lnm_plan_experiment", "-d", dataset_id], check=True)
            
            if dataset_id in ["312"]:
                nnUNetPlans_json(dataset_name)

        elif dataset_id in ["311"]:
            subprocess.run(
                [
                    "nnunetv2_lnm_move_plans_between_datasets",
                    "-s",
                    ref_dataset_id,
                    "-t",
                    dataset_id,
                    "-sp",
                    "nnUNetPlans",
                    "-tp",
                    "nnUNetPlans",
                ],
                check=True,
            )
            shutil.copyfile(
                os.path.join(nnUNet_raw, dataset_name, "dataset.json"),
                os.path.join(nnUNet_preprocessed, dataset_name, "dataset.json"),
            )

        split_final_json(dataset_name)

        subprocess.run(["nnunetv2_lnm_preprocess", "-d", dataset_id, "-c", args.config, "-np", "8"], check=True)

    elif args.mode == "train":
        if not ref_dataset_id:
            subprocess.run(
                ["nnunetv2_lnm_train", dataset_id, args.config, args.fold, "-tr", args.trainer], env=env, check=True
            )
        else:
            subprocess.run(
                [
                    "nnunetv2_lnm_train",
                    dataset_id,
                    args.config,
                    args.fold,
                    "-tr",
                    args.trainer,
                    "-pretrained_weights",
                    os.path.join(
                        nnUNet_results,
                        ref_dataset_name,
                        "nnUNetTrainer_oversample__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                    ),
                ],
                env=env,
                check=True,
            )

    elif args.mode == "test":
        suffix = args.trainer.replace("nnUNetTrainer_oversample", "") + args.config.replace("3d_fullres", "")
        subprocess.run(
            [
                "nnunetv2_lnm_predict",
                "-d",
                dataset_id,
                "-c",
                args.config,
                "-f",
                args.fold,
                "-tr",
                args.trainer,
                "-i",
                os.path.join(nnUNet_raw, dataset_name, f"imagesTs"),
                "-o",
                os.path.join(nnUNet_raw, dataset_name, f"predsTs{suffix}"),
                "-chk",
                f"checkpoint_best.pth",
            ],
            env=env,
            check=True,
        )