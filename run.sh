sh ./setup.sh > setup.log 2>&1
sh ./full_retrain.sh > retrain.log 2>&1
gsutil cp *.log gs://khang-sd-ft/log
