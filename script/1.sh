for METHOD in 'lstm' 'rnn_fcn' 'resnet' 'xresnet1d18' 'xresnet1d34'
do
    python examples/forecasting/beijing_dataset.py --method $METHOD
done