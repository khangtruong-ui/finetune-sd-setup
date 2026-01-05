echo "===== PWD: $(pwd) ====="

echo "===== SETUP START ====="
sh ./setup_env.sh > setup.log 2>&1
echo "===== RUN TASK ====="
sh ./train.sh > retrain.log 2>&1
gsutil cp *.log $SAVE_DIR/log
