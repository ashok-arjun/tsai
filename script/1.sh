for METHOD in 'rnn_fcn' 'resnet' 'xresnet1d18' 'xresnet1d34' 'lstm'
do
    python examples/forecasting/beijing_dataset.py --method $METHOD --data "data/beijing/PM10.csv"
done

for METHOD in 'rnn_fcn' 'resnet' 'xresnet1d18' 'xresnet1d34'
do
    python examples/forecasting/beijing_dataset.py --method $METHOD --data "data/beijing/PM2.5.csv"
done