TASK="regression"

# METHODS: 'rocket', 'transformer', 'tstplus', 'rnn', 'lstm', 'rnnplus', 'lstmplus', 'rnn_fcn', 'rnn_fcnplus', 'resnet', 'resnetplus', 'inceptiontime', 'inceptiontimeplus', 'xresnet1d18', 'xresnet1d18_deep', 'xresnet1d34', 'xresnet1d34_deep', 'xresnet1d18plus', 'xresnet1d18_deepplus', 'xresnet1d34plus', 'xresnet1d34_deepplus'

for TRAIN_START in 0 0.1 0.2 0.3 0.4 0.5
do
    for LR in 0.1 0.01 0.001
    do
        for METHOD in 'xresnet1d18', 'xresnet1d18_deep', 'xresnet1d34', 'xresnet1d34_deep'
        do
            python examples/forecasting/beijing_dataset.py --method $METHOD \
            --data "data/Processed-Beijing-ERA5/train_start_${TRAIN_START}/data.csv" \
            --run_name "Processed-Beijing-ERA5-${TASK}-TrainStart-${TRAIN_START}-LR-${LR}" \
            --lr $LR --tags "Beijing-ERA5-Train" --task $TASK
        done
    done
done