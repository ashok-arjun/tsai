# METHODS: 'rocket', 'transformer', 'tstplus', 'rnn', 'lstm', 'rnnplus', 'lstmplus', 'rnn_fcn', 'rnn_fcnplus', 'resnet', 'resnetplus', 'inceptiontime', 'inceptiontimeplus', 'xresnet1d18', 'xresnet1d18_deep', 'xresnet1d34', 'xresnet1d34_deep', 'xresnet1d18plus', 'xresnet1d18_deepplus', 'xresnet1d34plus', 'xresnet1d34_deepplus'

# TASK="regression"
# for DATA_DIR in "Processed-Beijing-ERA5-PM2.5" "ChangpingPM10_With_ERA5_processed"
# do
#     for TRAIN_START in 0 0.1 0.2 0.3 0.4 0.5
#     do
#         for LR in 0.1 0.01 0.001
#         do
#             for METHOD in 'rnn' # 'transformer', 'inceptiontime', 'xresnet1d18'
#             do
#                 python examples/forecasting/beijing_dataset.py --method $METHOD \
#                 --data "data/${DATA_DIR}/train_start_${TRAIN_START}/data.csv" \
#                 --run_name "${DATA_DIR}-${TASK}-TrainStart-${TRAIN_START}-LR-${LR}" \
#                 --lr $LR --tags "Beijing-ERA5-Train" "New-Sets" --task $TASK
#             done
#         done
#     done
# done

# TASK="forecasting"
# for FORECAST_HORIZON in 24 48 168 336 720
# do
#     for DATA_DIR in "Processed-Beijing-ERA5-PM2.5" "ChangpingPM10_With_ERA5_processed"
#     do
#         for TRAIN_START in 0 0.1 0.2 0.3 0.4 0.5
#         do
#             for LR in 0.1 0.01 0.001
#             do
#                 for METHOD in 'rnn' # 'transformer', 'inceptiontime', 'xresnet1d18'
#                 do
#                     python examples/forecasting/beijing_dataset.py --method $METHOD \
#                     --data "data/${DATA_DIR}/train_start_${TRAIN_START}/data.csv" \
#                     --run_name "${DATA_DIR}-${TASK}-TrainStart-${TRAIN_START}-LR-${LR}-Horizon-${FORECAST_HORIZON}" \
#                     --lr $LR --tags "Beijing-ERA5-Train" "New-Sets" --task $TASK --forecast_horizon $FORECAST_HORIZON
#                 done
#             done
#         done
#     done
# done

export CUDA_VISIBLE_DEVICES=4

TASK="forecasting"
for FORECAST_HORIZON in 48 168 336 720
do
    for DATA_DIR in "PM2.5" "PM10"
    do
        for TRAIN_START in 0 # 0.1 0.2 0.3 0.4
        do
            for LR in 0.1 0.01 # 0.001
            do
                for METHOD in 'lstmplus'
                do
                    python examples/forecasting/beijing_dataset.py --method $METHOD \
                    --data "datasets/${DATA_DIR}/${DATA_DIR}_train_start_${TRAIN_START}/data.csv" \
                    --run_name "${DATA_DIR}-${TASK}-TrainStart-${TRAIN_START}-LR-${LR}-Horizon-${FORECAST_HORIZON}" \
                    --lr $LR --tags "Beijing-ERA5-Train" "New-Sets" "Forecasting-Fix" --task $TASK --forecast_horizon $FORECAST_HORIZON
                done
            done
        done
    done
done


# TASK="classification"
# DATA_DIR="ChangpingWD_With_ERA5_processed"
# for TRAIN_START in 0.3 0.4 0.5 # 0 0.1 0.2 
# do
#     for LR in 0.1 0.01 0.001
#     do
#         for METHOD in 'rnn' # 'transformer', 'inceptiontime', 'xresnet1d18'
#         do
#             python examples/forecasting/beijing_dataset.py --method $METHOD \
#             --data "data/${DATA_DIR}/train_start_${TRAIN_START}/data.csv" \
#             --run_name "${DATA_DIR}-${TASK}-TrainStart-${TRAIN_START}-LR-${LR}" \
#             --lr $LR --tags "Beijing-ERA5-Train" "New-Sets" --task $TASK --num_epochs 1
#         done
#     done
# done

