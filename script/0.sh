for METHOD in 'xresnet1d18_deep' 'xresnet1d34_deep' 'rnn'
do
    python examples/forecasting/beijing_dataset.py --method $METHOD --data "data/beijing/PM10.csv"
done

for METHOD in 'xresnet1d18_deep' 'xresnet1d34_deep'
do
    python examples/forecasting/beijing_dataset.py --method $METHOD --data "data/beijing/PM2.5.csv"
done