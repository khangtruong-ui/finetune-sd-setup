export SAVE_DIR="gs://khang-sd-ft" 

echo "===== PWD: $(pwd) ====="
chmod +x setup_env.sh
chmod +x train.sh

echo "===== SETUP START ====="
./setup_env.sh > setup.log 2>&1
echo "===== RUN TASK ====="
./train.sh > retrain.log 2>&1
gsutil cp *.log $SAVE_DIR/log
