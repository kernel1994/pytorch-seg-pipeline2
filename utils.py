import csv
import pathlib
import shutil

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from lxml import objectify
from PIL import Image


def normalize(image, ranges=(-1., 1.)):
    """
    do image normalize, image to [min, max], default is [-1., 1.]
    :param image: ndarray
    :param ranges: tuple, (min, max)
    :return:
    """
    _min = ranges[0]
    _max = ranges[1]
    return (_max - _min) * (image - image.min()) / (image.max() - image.min()) + _min


def read_mha(mha_path: pathlib.Path, norm: tuple=None, reshape: tuple=None, dtype=np.float, tranpose: bool=True) -> np.ndarray:
    image_itk = sitk.ReadImage(str(mha_path))
    image_array = sitk.GetArrayFromImage(image_itk)
    if tranpose:
        # SimpleITK read image as (z, y, x), need to be transposed to (x, y, z)
        image_array = image_array.transpose((2, 1, 0))
    image_array = image_array.astype(dtype)

    if norm:
        image_array = normalize(image_array, norm)
    if reshape:
        image_array = image_array.reshape(reshape)

    return image_array


def least_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n least indices from a numpy array.

    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select

    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    return np.unravel_index(indices, array.shape)


def create_dir(path, parents=True):
    """
    create directory if dose not exists.
    :param path: pathlib.Path: create dir path object
    :param parents: boolean: weather create parents dir, default is True.
    :return:
    """
    if not path.exists():
        path.mkdir(parents=parents)


def create_new_dir(path, parents=True):
    """
    create directory and delete it if exists.
    :param path: Path: create dir path object
    :param parents: boolean: weather create parents dir, default is True.
    :return:
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=parents)


def postprocess(prd_ary, save_path, thd=None):
    if not thd:
        thd = prd_ary.mean()
    prd_ary[np.where(prd_ary < thd)] = 0
    prd_ary[np.where(prd_ary >= thd)] = 1
    prd_ary = prd_ary.astype(np.int8)

    out_img = sitk.GetImageFromArray(prd_ary)

    cnt_filter = sitk.ConnectedComponentImageFilter()
    out_img = cnt_filter.Execute(out_img)

    rlb_filter = sitk.RelabelComponentImageFilter()
    out_img = rlb_filter.Execute(out_img)

    bin_filter = sitk.BinaryThresholdImageFilter()
    out_img = bin_filter.Execute(out_img, 1, 1, 1, 0)

    prd_postprocessed = sitk.GetArrayFromImage(out_img)
    fig_save(prd_postprocessed, save_path)


def parse(xml_file_path):
    with xml_file_path.open('r', encoding='utf-8') as f:
        root = objectify.fromstring(f.read())
        dice_value = root.metrics.DICE.attrib['value']
    return (xml_file_path.stem, float(dice_value))


def parse_and_save_sub(results_dir, cvs_file_name):
    # specifying the fields for csv file
    fields = ['id', 'name', 'dice']

    all_values = []
    idx = 0
    for p_path in results_dir.iterdir():
        if p_path.suffix != '.xml':
            continue
        xml_file_path = p_path
        parse_item = parse(xml_file_path)
        all_values.append({fields[0]: idx,
                           fields[1]: parse_item[0],
                           fields[2]: parse_item[1]})
        idx += 1

    # calc mean and std value
    value_list = [item[fields[2]] for item in all_values]
    mean = np.mean(value_list)
    std = np.std(value_list)
    all_values.append({
        fields[0]: -2,
        fields[1]: 'mean',
        fields[2]: round(mean, 5)
    })
    all_values.append({
        fields[0]: -1,
        fields[1]: 'std',
        fields[2]: round(std, 5)
    })

    # save to csv file
    with results_dir.joinpath(cvs_file_name).open('w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(all_values)


def parse_and_save_all(main_dir, cvs_file_name):
    # specifying the fields for csv file
    fields = ['id', 'kid', 'name', 'dice']

    all_values = []
    idx = 0
    for k_path in main_dir.iterdir():
        for p_path in k_path.iterdir():
            if p_path.suffix != '.xml':
                continue
            xml_file_path = p_path
            parse_item = parse(xml_file_path)
            all_values.append({fields[0]: idx,
                               fields[1]: k_path.stem,
                               fields[2]: parse_item[0],
                               fields[3]: parse_item[1]})
            idx += 1

    # calc mean and std value
    value_list = [item[fields[3]] for item in all_values]
    mean = np.mean(value_list)
    std = np.std(value_list)
    all_values.append({
        fields[0]: -2,
        fields[1]: -2,
        fields[2]: 'mean',
        fields[3]: round(mean, 5)
    })
    all_values.append({
        fields[0]: -1,
        fields[1]: -1,
        fields[2]: 'std',
        fields[3]: round(std, 5)
    })

    # save to csv file
    with main_dir.joinpath(cvs_file_name).open('w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(all_values)


def draw_loss(loss_csv_path, save_path):
    plt.clf()

    df = pd.read_csv(loss_csv_path)
    epoch = df['epoch'].drop_duplicates().values
    train_loss = df.loc[df['stage'] == 'train']['loss'].values
    val_loss = df.loc[df['stage'] == 'val']['loss'].values

    lw = 1
    plt.plot(epoch, train_loss, color='darkorange', lw=lw, label='train loss')
    plt.plot(epoch, val_loss, color='skyblue', lw=lw, label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.savefig(str(save_path), format='png')


def fig_save(image_ary, save_path):
    # convert to Pillow image
    image_obj = Image.fromarray(image_ary.astype('uint8'), mode='L')

    # save to file
    image_obj.save(save_path)
