export CUDA_VISIBLE_DEVICES=7

TASK="forecasting"
for DATA_DIR in "PM2.5" "PM10"
do
    for FORECAST_HORIZON in 24 48 168 336 720
    do
        for TRAIN_START in 0 # 0.1 0.2 0.3 0.4
        do
            for LR in 0.1 0.01 # 0.001
            do
                for METHOD in 'tstplus'
                do
                    python examples/forecasting/beijing_dataset.py --method $METHOD \
                    --data "datasets/${DATA_DIR}/${DATA_DIR}_train_start_${TRAIN_START}/data.csv" \
                    --run_name "${DATA_DIR}-${TASK}-TrainStart-${TRAIN_START}-LR-${LR}-Horizon-${FORECAST_HORIZON}" \
                    --lr $LR --tags "Beijing-ERA5-Train" "New-Sets" "Forecasting-Fix-lookback-1" \
                    --task $TASK --forecast_horizon $FORECAST_HORIZON \
                    --lookback_horizon 1
                done
            done
        done
    done
done
