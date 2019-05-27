#!/bin/bash

unset GREP_OPTIONS

export XPUSIM_SIMULATOR_MODE=FUNCTION
export XPUSIM_SSE_LOG_LEVEL=INFO

#models=("ResNet50" "VGG19" "AlexNet" "GoogleNet" "InceptionV4" "MobileNet" "MobileNetV2" "DistResNet" "SE_ResNeXt50_32x4d")
#models=("ResNet50")
models=("GoogleNet")

#batch_size=(1 2 4 8 16)
batch_size=(1)

#place="cuda"
#run_mode=("train" "infer")

place="xsim"
#run_mode=("train" "infer" "fused_infer")
run_mode=("fused_infer")
#run_mode=("infer")

for model in  ${models[@]}
do
    for mode in ${run_mode[@]}
    do
        for batch in ${batch_size[@]}
        do
            echo $model $batch $mode $place
            python train.py \
                --model=$model \
                --run_mode=$mode \
                --batch_size=$batch \
                --place=$place 2>&1 | tee ops_$model\_$mode.log
        done
    done
done

#echo -n "model "
#for mode in ${run_mode[@]}
#do
#    for batch in ${batch_size[@]}
#    do
#        echo -n $mode"_"$batch" "
#    done
#done
#echo
#
#for model in  ${models[@]}
#do
#    echo -n $model" "
#    for mode in ${run_mode[@]}
#    do
#        for batch in ${batch_size[@]}
#        do
#            cat perf_$place\_$model\_$batch\_$mode.log | grep -i thread0 | awk 'BEGIN{sum=0}{sum+=$3}END{printf("%f ", sum)}'
#        done
#    done
#    echo
#done
