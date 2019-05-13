#!/bin/bash
unset GREP_OPTIONS
file=$1
tmp_file=analyze_tmp
result_file=analyze_result_tmp

set=$(grep "RunPreparedContext" $file | awk '{print $3}' | cut -d "(" -f1 | sort | uniq)

cat $file | awk 'BEGIN {n=0};
            /RunPreparedContext/{printf "%s line%d\n" ,$0, n; n++; next}
            {print $0}
            END {print n}' > $tmp_file

num=`tail -1 $tmp_file`

echo "Doing statistical analysis... [" $num "] ops in total..."
for ((i=0; i < $num; i++))
do
    p1="line"$i
    p2="line"$(($i + 1)) # or $[$i + 1];
    kernel=`cat $tmp_file | awk "/$p1$/"'{print $3}'`
    if [ ! -f "$kernel"_tmp_file"" ]; then
        echo > "$kernel"_tmp_file""
    fi
    cat $tmp_file | awk "/$p1$/"'{flag=1}'"/$p2$/"'{flag=0}flag' |
        awk 'BEGIN {sum=0; max=0; count=0;};
            {
             if ($5 == "Compute") {max=0; count++;}
             if ($4 == "DURATION]:") {if ($5>max) {max=$5;} count--; if (count == 0) {sum += max; max=0;}} 
            }
            END {printf("DURATION: %f\n", sum);}' >> "$kernel"_tmp_file
done

echo > $result_file

for kernel in $set;
do
    cat $kernel"_tmp_file" |
        grep "DURATION" |
        awk 'BEGIN {sum = 0; min = 65536; max=0; cnt=0};
            { sum += $2; cnt++;
              if ($2 + 0 < min + 0) min=$2;
              if ($2 + 0 > max + 0) max=$2; };
            END {printf("%f %f %f %d:", sum / 1000.0, min / 1000.0, max / 1000.0, cnt);}' >> $result_file
    echo $kernel >> $result_file
    rm $kernel"_tmp_file"
done;

echo
echo "Printing reslut... unit(ms)"
echo

cat $result_file | sort -nr -k 1 -t " " |
    awk 'BEGIN{sumall = 0; printf("%s %s %s cnt\n", "sum", "min", "max")}
         {print $0; sumall += $1;}
         END{printf("Total: %f ms\n", sumall)}'

rm $tmp_file
rm $result_file
