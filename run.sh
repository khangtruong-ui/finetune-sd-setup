echo "===== PWD: $(pwd) ====="
echo "Script argument: $1"
chmod +x setup.sh
chmod +x full_retrain.sh

./setup.sh "$1" > setup.log 2>&1
./full_retrain.sh > retrain.log 2>&1
gsutil cp *.log gs://khang-sd-ft/log
