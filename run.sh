export SAVE_DIR="gs://khang-sd-ft" 

echo "===== PWD: $(pwd) ====="
chmod +x setup.sh
chmod +x full_retrain.sh

echo "===== SETUP START ====="
./setup.sh > setup.log 2>&1
echo "===== RUN TASK ====="
./full_retrain.sh > retrain.log 2>&1
gsutil cp *.log $SAVE_DIR/log
