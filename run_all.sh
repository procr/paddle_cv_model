#!/bin/bash

unset GREP_OPTIONS
export XPUSIM_SIMULATOR_MODE=SYSTEMC
export XPUSIM_SSE_LOG_LEVEL=INFO

#models=("ResNet50" "VGG19" "AlexNet" "GoogleNet" "InceptionV4" "MobileNet" "MobileNetV2" "DistResNet" "SE_ResNeXt50_32x4d")
models=("ResNet50")
#models=("ResNet50" "VGG19" "AlexNet" "GoogleNet" "InceptionV4" "MobileNet" "MobileNetV2" "DistResNet" "SE_ResNeXt50_32x4d")

#batch_size=(1 2 4 8 16)
batch_size=(1)

#place="cuda"
#run_mode=("train" "infer")

place="xsim"
#run_mode=("train" "infer" "fused_infer")
run_mode=("fused_infer")
#precision=("int8" "int16")
precision=("int16")

for model in  ${models[@]}
do
    for mode in ${run_mode[@]}
    do
        for batch in ${batch_size[@]}
        do
            for pre in ${precision[@]}
            do
                echo $model $batch $mode $place $pre
                python train.py \
                    --model=$model \
                    --run_mode=$mode \
                    --batch_size=$batch \
                    --precision=$pre \
                    --place=$place 2>&1 | tee perf_$place\_$model\_$batch\_$mode\_$pre.log
            done
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
