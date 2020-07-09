
#script=/opt/intel/openvino/inference_engine/samples/python/classification_sample/classification_sample.py
script=/opt/intel/openvino/inference_engine/samples/python/object_detection_sample_ssd/object_detection_sample_ssd.py

model=$1
#model=/home/xy18/workspace/git/tracker/2020/siamban/2020R1/FP32/my.xml

	source /opt/intel/openvino/bin/setupvars.sh
	python $script -i car.png -m $model

# usage
# ./run-openvino2020-python-sample.sh 2020R1/FP32/sr.xml
# ./run-openvino2020-python-sample.sh /home/xy18/workspace/git/tracker/2020/siamban/2020R1/FP32/my.xml 2>&1 | tee my.log-ov-python-sample
