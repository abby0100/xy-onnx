#!/bin/bash

onnx=$1
ov=2020R1
dataType=FP32
outDir=$ov/$dataType 
mo=/opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py

	source /opt/intel/openvino_fpga_2020.1.023/bin/setupvars.sh
	python3 $mo --input_model $onnx --output_dir $outDir --data_type $dataType \
	--input_shape [1,3,112,112]

# usage
# ./run-openvino2020-mo.sh sr.onnx
