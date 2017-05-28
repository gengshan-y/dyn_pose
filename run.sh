cp model/lstm.config.origin model/lstm.config
echo "buckets = 50" >> model/lstm.config
echo "load_epoch = 5" >> model/lstm.config
./infer_lstm.py

cp model/lstm.config.origin model/lstm.config
echo "buckets = 50" >> model/lstm.config
echo "load_epoch = 10" >> model/lstm.config
./infer_lstm.py

cp model/lstm.config.origin model/lstm.config
echo "buckets = 50" >> model/lstm.config
echo "load_epoch = 15" >> model/lstm.config
./infer_lstm.py

cp model/lstm.config.origin model/lstm.config
echo "buckets = 50" >> model/lstm.config
echo "load_epoch = 20" >> model/lstm.config
./infer_lstm.py

cp model/lstm.config.origin model/lstm.config
echo "buckets = 50" >> model/lstm.config
echo "load_epoch = 25" >> model/lstm.config
./infer_lstm.py

