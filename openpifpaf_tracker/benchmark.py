"""Benchmark."""

import argparse
from collections import namedtuple
import datetime
import logging
import os

import openpifpaf.benchmark

LOG = logging.getLogger(__name__)


DEFAULT_CHECKPOINTS = [
    'tshufflenetv2k16',
]


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.benchmark',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=openpifpaf.__version__))

    parser.add_argument('--output', default=None,
                        help='output file name')
    parser.add_argument('--checkpoints', default=DEFAULT_CHECKPOINTS, nargs='+',
                        help='checkpoints to evaluate')
    parser.add_argument('--ablation-1', default=False, action='store_true')
    parser.add_argument('--ablation-2', default=False, action='store_true')
    group = parser.add_argument_group('logging')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args, eval_args = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # default eval_args
    if not eval_args:
        eval_args = ['--loader-workers=8']

    # default loader workers
    if not any(l.startswith('--loader-workers') for l in eval_args):
        LOG.info('adding "--loader-workers=8" to the argument list')
        eval_args.append('--loader-workers=8')

    # default dataset
    if not any(l.startswith('--dataset') for l in eval_args):
        LOG.info('adding "--dataset=posetrack2018" to the argument list')
        eval_args.append('--dataset=posetrack2018')
        if not any(l.startswith('--keypoint-threshold') for l in eval_args):
            LOG.info('adding "--keypoint-threshold=0.2" to the argument list')
            eval_args.append('--keypoint-threshold=0.2')
        if not any(l.startswith('--instance-threshold') for l in eval_args):
            LOG.info('adding "--instance-threshold=0.01" to the argument list')
            eval_args.append('--instance-threshold=0.01')
        if not any(l.startswith('--seed-threshold') for l in eval_args):
            LOG.info('adding "--seed-threshold=0.4" to the argument list')
            eval_args.append('--seed-threshold=0.4')
        if not any(l.startswith('--no-reverse-match') for l in eval_args):
            LOG.info('adding "--no-reverse-match" to the argument list')
            eval_args.append('--no-reverse-match')
        if not any(l.startswith('--write-predictions') for l in eval_args):
            LOG.info('adding "--write-predictions" to the argument list')
            eval_args.append('--write-predictions')

    # generate a default output filename
    if args.output is None:
        now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        args.output = 'outputs/benchmark-{}/'.format(now)
        os.makedirs(args.output)

    return args, eval_args


Config = namedtuple('Config', ['checkpoint', 'suffix', 'args'])


class Benchmark(openpifpaf.benchmark.Benchmark):
    def print_md(self):
        """Pretty printing markdown"""
        stats = self.stats()

        checkpoint_w = max(len(c) for c in stats.keys()) + 2
        checkpoint_title = 'Checkpoint'
        labels = ''.join(['{0: <8} |'.format(l) for l in stats['text_labels']])
        print(
            f'| {checkpoint_title: <{checkpoint_w}} | {labels}'
            ' t_{total} [ms]  | t_{dec} [ms] |     size |'
        )

        reference = None
        if self.reference_config is not None:
            reference = stats[self.reference_config.checkpoint + self.reference_config.suffix]

        for checkpoint, data in sorted(stats.items(), key=lambda b_d: b_d[1]['stats'][0]):
            stats = list(data['stats'])
            t = 1000.0 * data['total_time'] / data['n_images']
            tdec = 1000.0 * data['decoder_time'] / data['n_images']
            file_size = data['file_size'] / 1024 / 1024
            checkpoint_link = '[' + checkpoint + ']'

            if self.reference_config is not None \
               and self.reference_config.checkpoint != checkpoint:
                stats = [v - r for v, r in zip(stats, reference['stats'])]
                t -= 1000.0 * reference['total_time'] / reference['n_images']
                tdec -= 1000.0 * reference['decoder_time'] / reference['n_images']
                file_size -= reference['file_size'] / 1024 / 1024

                values = '__{0: <+2.1f}__'.format(stats[0])
                if len(stats) > 1:
                    values += ''.join(['{0: <+6.1f} |'.format(v) for v in stats[1:]])
            else:
                values = '__{0: <2.1f}__'.format(stats[0])
                if len(stats) > 1:
                    values += ''.join(['{0: <6.1f} |'.format(v) for v in stats[1:]])
            print(
                f'| {checkpoint_link: <{checkpoint_w}} | {values} '
                f'{t: <+15.0f} | {tdec: <+12.0f} | {file_size: >+6.1f}MB |'
            )

        return self


def main():
    args, eval_args = cli()
    Ablation = namedtuple('Ablation', ['suffix', 'args'])
    ablations = [Ablation('', eval_args)]

    if args.ablation_1:
        ablations += [
            Ablation('.greedy', eval_args + ['--greedy']),
            Ablation('.greedy.dense', eval_args + ['--greedy', '--dense-connections']),
            Ablation('.dense', eval_args + ['--dense-connections']),
            Ablation('.dense.hierarchy', eval_args + ['--dense-connections=0.1']),
        ]
    if args.ablation_2:
        ablations += [
            Ablation('.nr.nms', eval_args + ['--ablation-cifseeds-no-rescore',
                                             '--ablation-cifseeds-nms',
                                             '--ablation-caf-no-rescore']),
        ]

    configs = [
        Config(checkpoint, ablation.suffix, ablation.args)
        for checkpoint in args.checkpoints
        for ablation in ablations
    ]
    Benchmark(configs, args.output, reference_config=configs[0]).run()


if __name__ == '__main__':
    main()
