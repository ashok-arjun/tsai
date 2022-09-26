export CUDA_VISIBLE_DEVICES=7

TASK="regression"
for DATA_DIR in "PM10"
do
    for TRAIN_START in 0.2 0.3 0.4
    do
        for LR in 0.1 0.01 # 0.001
        do
            for METHOD in 'lstm' 'inceptiontime'
            do
                python examples/forecasting/beijing_dataset.py --method $METHOD \
                --data "datasets/${DATA_DIR}/${DATA_DIR}_train_start_${TRAIN_START}/data.csv" \
                --run_name "${DATA_DIR}-${TASK}-TrainStart-${TRAIN_START}-LR-${LR}-Horizon-${FORECAST_HORIZON}" \
                --lr $LR --tags "Beijing-ERA5-Train" "New-Sets" --task $TASK
            done
        done
    done
done
