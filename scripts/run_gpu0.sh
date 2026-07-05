SINGLE_TRANSFORM_VERSION=52
PAIR_TRANSFORM_VERSION=213

python train.py \
  --single_transform_version "$SINGLE_TRANSFORM_VERSION" \
  --pair_transform_version "$PAIR_TRANSFORM_VERSION" \
  --gpu "0"
