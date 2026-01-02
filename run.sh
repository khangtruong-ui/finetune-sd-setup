sh ./setup.sh > setup.log
sh ./full_retrain.sh > retrain.log
gsutil cp *.log gs://khang-sd-ft/log
