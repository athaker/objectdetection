# You need to install things everything
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# Make sure you are in the correct directory object_detection
python xml_to_csv.py # You can skip this step if you are not using pascal VOC

python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record

# get the pre-trained model
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz

# unzip the pretrained model
tar -xzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz

# From models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# then navigate back to this directory

python train.py --logtostderr --train_dir=data/ --pipeline_config_path=training/pipeline.config

# Now this is going to take a long time to run, but eventually you should start seeing something that looks like:
# INFO:tensorflow:global step 1: loss = 14.4836 (135.976 sec/step)
# IINFO:tensorflow:global step 2: loss = 12.4565 (77.529 sec/step)
# ...
# ...
# ...

# Here is gets complicated and more difficult to explain
# 1. You will export the model 

# Export your model to use and score
# model.ckpt-14, is the latest model that you have, this will be different
python export_inference_graph.py --input_type image_tensor --pipeline_config_path=training/pipeline.config --trained_checkpoint_prefix=data/model.ckpt-14 --output_directory=output


# Finally you can do scoring, and will leave it to you to experiment with it
python scoring.py
