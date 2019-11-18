import gc
import pathlib

from sklearn.model_selection import KFold

import cfg
import dataset
import evaluate
import models
import train
import utils


def prepare():
    if cfg.is_train:
        utils.create_new_dir(cfg.main_dir)

    return cfg.main_dir


def prepare_subdir(main_dir: pathlib.Path, sub_dir_name: str):
    """
    prepare sub results dir for KFold or Leave One Out.
    """
    sub_dir = main_dir.joinpath(sub_dir_name)
    if cfg.is_train:
        utils.create_new_dir(sub_dir)

    return sub_dir


def main():
    main_path = prepare()

    # generate data
    ori_paths, seg_paths = dataset.get_all_data_path()

    kf = KFold(n_splits=cfg.n_kfold, shuffle=True, random_state=cfg.random_state)
    for ki, (train_idx, test_idx) in enumerate(kf.split(ori_paths, seg_paths)):
        X_train, X_test = ori_paths[train_idx], ori_paths[test_idx]
        y_train, y_test = seg_paths[train_idx], seg_paths[test_idx]
        splited = (X_train, X_test, y_train, y_test)

        test_names = [x.stem for x in X_test]
        sub_dir = prepare_subdir(main_path, str(ki))

        # build model
        model = models.create(cfg.model_name)

        # build dataloader
        train_dataloader, test_dataloader = dataset.image_dataloader(splited)

        # train and val model
        if cfg.is_train:
            train.train_val(model, train_dataloader, test_dataloader, sub_dir)

        # test model
        train.test_batch(model, test_dataloader, sub_dir)

        # calc metric
        evaluate.calc_dice(test_names, sub_dir)

        # TODO: tensorboard

        # count all dice of each k
        utils.parse_and_save_sub(sub_dir, cfg.sub_summary_name)

        del model
        gc.collect()

    # count all dice of all k
    utils.parse_and_save_all(main_path, cfg.summary_name)


if __name__ == "__main__":
    main()
