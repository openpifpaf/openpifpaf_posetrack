# Install

```sh
pip install 'openpifpaf_tracker[test,train]'

# from source:
pip install --editable '.[test,train]'
```

# CrowdPose Data

```sh
mkdir data-crowdpose
cd data-crowdpose
# download links here: https://github.com/Jeff-sjtu/CrowdPose
unzip annotations.zip
unzip images.zip
```

# Train posetrack2018-cocokp

```sh
python3 -m openpifpaf_tracker.imagetotracking \
  --checkpoint ../pifpaf_run/outputs/shufflenetv2k16-201008-120711-cocokp-o10s-58034177.pkl
```

```sh
time CUDA_VISIBLE_DEVICES=0,1 python3 -m openpifpaf.train \
  --lr=0.000025 --momentum=0.98 --b-scale=10.0 \
  --epochs=50 \
  --lr-decay 40 45 \
  --lr-decay-epochs=5 \
  --batch-size=8 \
  --weight-decay=1e-5 \
  --dataset=posetrack2018-cocokp --posetrack-upsample=2 --cocokp-upsample=2 --cocokp-orientation-invariant=0.5 --cocokp-blur=0.1 --cocokp-square-edge=513 \
  --checkpoint outputs/tshufflenetv2k16-201008-120711-cocokp-o10s-58034177.pkl
```

```sh
time CUDA_VISIBLE_DEVICES=3 python -m openpifpaf.eval \
  --checkpoint outputs/tshufflenetv2k30-200926-224057-posetrack2018-cocokp-edge513-o50-484ddecb.pkl \
  --dataset=posetrack2018 \
  --batch-size=1 --loader-workers=8 \
  --decoder trackingpose \
  --seed-threshold=0.4 --keypoint-threshold=0.2 --instance-threshold=0.01 \
  --write-predictions
```

```sh
while true; do \
  CUDA_VISIBLE_DEVICES=3 find outputs/ -name "tshufflenetv2k30-201009-162243-posetrack2018-cocokp-edge513-o50.pkl.epoch??[0,5]" -exec \
    python -m openpifpaf.eval \
      --checkpoint {} \
      --skip-existing \
      --dataset=posetrack2018 \
      --batch-size=1 --loader-workers=8 \
      --decoder trackingpose \
      --seed-threshold=0.4 --keypoint-threshold=0.2 --instance-threshold=0.01 \
      --write-predictions \; \
  ; \
  sleep 300; \
done
```

Demo:

```sh
MPLBACKEND=macosx python -m openpifpaf.video --checkpoint outputs/tshufflenetv2k30-200923-230634-posetrack2018-cocokp-edge513-o50-3fbacca9.pkl --long-edge=1201 --decoder trackingpose --show-only-decoded-connections --save-all --source "data-posetrack2018/images/val/001001_mpii_test/%06d.jpg" --skip-frames=5 --image-min-dpi=200 --show-file-extension=jpeg --white-overlay=0.7 --show-multitracking  --image-height=3 --skeleton-solid-threshold=0.0 --text-color=black --monocolor-connections

MPLBACKEND=macosx python -m openpifpaf.video --checkpoint outputs/tshufflenetv2k30-200923-230634-posetrack2018-cocokp-edge513-o50-3fbacca9.pkl --decoder trackingpose --show-only-decoded-connections --save-all --source "data-shadows2/%06d.jpeg" --image-min-dpi=200 --show-file-extension=jpeg --show-multitracking --image-height=3 --textbox-alpha=1.0 --long-edge=321 --skeleton-solid-threshold=0.0 --text-color=black
```

===

```sh
srun --account=vita --gres gpu:1 singularity exec --nv pytorch_latest.sif python test.py
srun --account=vita --gres gpu:1 nvidia-smi
```

===

Full training:

```sh
time CUDA_VISIBLE_DEVICES=0,1 python -m openpifpaftracker.train \
  --pose-checkpoint ../pifpaf_run/outputs/shufflenetv2k30w-200510-104256-cif-caf-caf25-o10s-0b5ba06f.pkl \
  --head-quad=1 \
  --batch-size=8 \
  --headnets cif17,caf18,tcaf17,caf23 cif,caf,caf25 \
  --square-edge=513 \
  --datasets posetrack cocokp \
  --grouped-lambdas 1,1,0.2,1,1,1,0.2,0.2,1,1,1,0.2,0.2,1,1,1,0.2,0.2    1,1,0.2,1,1,1,0.2,0.2,1,1,1,0.2,0.2 \
  --auto-tune-mtl \
  --lr=0.003 \
  --momentum=0.975 \
  --epochs=50 \
  --lr-decay 40 45 \
  --lr-decay-epochs=5 \
  --dataset-weights 1.0 1.0 \
  --weight-decay=1e-5 \
  --ema=0.01 \
  --stride-apply=2

time CUDA_VISIBLE_DEVICES=0,1 python -m openpifpaftracker.train \
  --checkpoint outputs/tshufflenetv2k30w-200516-093254-cif17caf18tcaf17caf23-cifcafcaf25-d192f64c.pkl \
  --head-quad=1 \
  --batch-size=8 \
  --square-edge=513 \
  --datasets posetrack cocokp \
  --grouped-lambdas 1,1,0.2,1,1,1,0.2,0.2,1,1,1,0.2,0.2,1,1,1,0.2,0.2    1,1,0.2,1,1,1,0.2,0.2,1,1,1,0.2,0.2 \
  --auto-tune-mtl \
  --lr=0.003 \
  --momentum=0.975 \
  --epochs=100 \
  --lr-decay 75 90 \
  --lr-decay-epochs=10 \
  --dataset-weights 1.0 1.0 \
  --weight-decay=1e-5 \
  --ema=0.01 \
  --stride-apply=2
```

For local tests:
```sh
python -m openpifpaftracker.train \
   --pose-checkpoint=shufflenetv2x2 \
   --head-quad=1 \
   --batch-size=1 \
   --headnets pif17,pafs18,tpafs17,tskeleton \
   --long-edge=401 \
   --datasets posetrack \
   --loader-workers=0 \
   --grouped-lambdas 1,1,1,1,1,1,1,1,1,1,1,1,1 \
   --auto-tune-mtl \
   --posetrack-train-annotations data-posetrack/annotations/val/001001_mpii_test.json \
   --debug
```

Eval:

```sh
time python -m openpifpaftracker.predict_posetrack_parallel \
  --n-workers=24 \
  --checkpoint outputs/tshufflenetv2k30w-200525-100140-cif17caf18tcaf17caf23-cifcafcaf25-e326f2a2.pkl \
  --batch-size=1 \
  --loader-workers=8 \
  --long-edge=801 \
  --instance-threshold=0.01 \
  --single-pose-threshold=0.3 \
  --multi-pose-threshold=0.2 \
  --keypoint-threshold=0.2 \
  --seed-threshold=0.4 \
  --frames 0 -1 -2 -3 \
  --gpus 2 3  # for test add: --annotations "data-posetrack/annotations/test/*.json"

# or for 2017 formatting
time python -m openpifpaftracker.predict_posetrack_parallel \
  --n-workers=24 \
  --checkpoint outputs/tshufflenetv2k30w-200525-100140-cif17caf18tcaf17caf23-cifcafcaf25-e326f2a2.pkl \
  --annotations "data-posetrack2017/annotations/val/*.json" --format2017 \
  --batch-size=1 \
  --loader-workers=8 \
  --long-edge=801 \
  --instance-threshold=0.01 \
  --single-pose-threshold=0.3 \
  --multi-pose-threshold=0.2 \
  --keypoint-threshold=0.2 \
  --seed-threshold=0.4 \
  --frames 0 -1 -2 -3 \
  --gpus 2 3

# in a Python2 environment
time python evaluate.py \
  --groundTruth ../../openpifpaftracker_run/data-posetrack/annotations/val/ \
  --predictions ../../openpifpaftracker_run/outputs/posetrack-predict-191101-191757/ \
  --outputDir ../../openpifpaftracker_run/outputs/posetrack-predict-191101-191757-poseval/ \
  --evalPoseTracking \
  --evalPoseEstimation \
  --saveEvalPerSequence
```

For test-server submission, zip the files with the Mac Finder.
It's 375 files for 2018 and 214 for 2017.

Sometimes something goes wrong and you just want to rerun the missing files.

```sh
diff \
  <(cd ../../openpifpaftracker_run/data-posetrack/annotations/val/ && ls -1 *.json) \
  <(cd ../../openpifpaftracker_run/outputs/posetrack-predict-191106-085718/ && ls -1 *.json)
```


# Train MOT

```sh
python -m openpifpaftracker.train \
    --pose-checkpoint=resnet101 \
    --head-quad=1 \
    --train-annotations 'data-mot/train/MOT17-0*-DPM/gt/gt.txt' \
    --val-annotations 'data-mot/train/MOT17-1*-DPM/gt/gt.txt' \
    --batch-size=1 \
    --headnets baf skeleton \
    --long-edge=401 \
    --freeze-base=1 \
    --debug \
    --dataset=mot \
    --lambdas 10 3 3 10 3 3 10 3 3 10 3 3
```

Eval:

```sh
python -m openpifpaftracker.predict_mot --checkpoint outputs/tresnet101block5-bafi-190228-105342.pkl --annotations 'data-mot/train/MOT17-1*-DPM/gt/gt.txt' --batch-size=8 --long-edge=721 --dataset=mot --instance-threshold=0.0
python -m motmetrics.apps.eval_motchallenge data-mot/train outputs/mot-predict-<timestamp> --fmt mot16
```


# MOT

Data.

```
mkdir data-mot
cd data-mot
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
```

Evaluation is a Python package that already is a dependency of this tracker.


# Posetrack

Data. Follow the Posetrack instructions to download and untar the images.
Labels:

```sh
mkdir data-posetrack
cd data-posetrack
wget https://posetrack.net/posetrack18-data/posetrack18_v0.45_public_labels.tar.gz
tar -xvf posetrack18_v0.45_public_labels.tar.gz
mv posetrack_data/* .
rm -r posetrack_data
```

Evaluation. DO NOT FOLLOW POSETRACK INSTALL INSTRUCTIONS. NEVER TOUCH PYTHON_PATH!
Keep the eval code in a separate folder from this tracker code. Install:

```sh
git clone https://github.com/leonid-pishchulin/poseval.git --recursive posetrack-poseval
cd posetrack-poseval/py
virtualenv venv  # has to be Python2 ... :(
source venv/bin/activate
pip install shapely numpy click tqdm enum
pip install -e ../py-motmetrics
```

Run (change the predictions path to your actual predictions):
```sh
time python evaluate.py \
  --groundTruth ../../openpifpaftracker/data-posetrack/annotations/val/ \
  --predictions ../../openpifpaftracker/data-posetrack/annotations/val/ \
  --evalPoseTracking \
  --evalPoseEstimation \
  --saveEvalPerSequence
```

To print all the outputs:

```sh
for fn in out/*.json
do
  echo ${fn}
  cat ${fn}
done
```

Generate PoseTrack2017 json data of the ground truth.
Usage of `octave` instead of `matlab` is not documented, but this seems to work:

```sh
cd matlab
octave --no-gui --eval "addpath('./external/jsonlab'); mat2json('your_relative_path/data-posetrack2017/annotations/val/'); quit"
```

This takes a long time. It is faster on the test set:

```sh
octave --no-gui --eval "addpath('./external/jsonlab'); mat2json('your_relative_path/data-posetrack2017/annotations/test/'); quit"
```


# Partial Data

The training script supports ``--train-annotations`` and ``--val-annotations``
to restrict the used annotation files. This is useful for local testing.


# Person Skeletons

The Posetrack skeleton used here:

<img src="docs/skeleton_posetrack.png" height=250 alt="Posetrack skeleton" />

Created with ``python -m openpifpaf.data``.
