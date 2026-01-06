echo "===== WORKING DIRECTORY: $(pwd) ====="
echo "===== SAVE DIRECTORY: $SAVE_DIR ====="
set -e
echo "===== SETUP START ====="
sh ./setup.sh > setup.log 2>&1
echo "===== RUN TASK ====="
sh ./train.sh 2>&1 | grep -v "deprecated" > retrain.log 
gsutil cp *.log $SAVE_DIR/log
