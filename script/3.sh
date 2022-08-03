for METHOD in 'rnn_fcnplus' 'resnetplus'  'lstmplus'
do
    python examples/forecasting/beijing_dataset.py --method $METHOD --data "data/beijing/PM2.5.csv" --task 'forecasting'
done

for METHOD in 'rnn_fcnplus' 'resnetplus' 'lstmplus'
do
    python examples/forecasting/beijing_dataset.py --method $METHOD --data "data/beijing/PM10.csv" --task 'forecasting'
done