for METHOD in 'rocket' 'transformer' 'rnn' 'xresnet1d18_deep' 'xresnet1d34_deep'
do
    python examples/forecasting/beijing_dataset.py --method $METHOD
done