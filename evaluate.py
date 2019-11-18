import os

import cfg


def calc_dice(test_names, results_dir):
    for p_name in test_names:
        # TODO: feature: ori data required to be process
        truth = str(cfg.seg_dir.joinpath(f'{p_name}_seg.mha'))
        predict = str(results_dir.joinpath(f'{p_name}_prd_bin.png'))
        output_xml = str(results_dir.joinpath(f'{p_name}.xml'))

        cli = f'{cfg.evaluator} {truth} {predict} -use all -xml {output_xml}'
        print(f'excute command: {cli}')
        os.system(cli)
