#!/bin/bash

rm -r ../Data/Raw
for symbol in $(curl -s 'https://ticks.ex2archive.com/ticks/' | jq -r .[].name | grep -v -e Raw -e Zero -e Standart | grep -v -e m$ -e c$ -e '_') ; do
   echo -e $symbol"\tProcesando."
   years=`curl https://ticks.ex2archive.com/ticks/$symbol/ -s | jq -r '.[].name' | grep -v 2024`
   if grep -q 2023 <<< $years; then
	   echo -e $symbol"\tTiene año 2023."
	   mkdir -p ../Data/Raw/$symbol
	   cd ../Data/Raw/$symbol;
	   for year in $years ; do
		echo -e $symbol $year "\tProcesando año."
		wget -q https://ticks.ex2archive.com/ticks/$symbol/$year/Exness_"$symbol"_$year.zip;
		unzip -q ./Exness_"$symbol"_$year.zip;
		rm ./Exness_"$symbol"_$year.zip;
		sed -i '1d' ./Exness_"$symbol"_$year.csv
		awk -F '[" :,]' '{s=$8" "$9":"$10; count[substr(s, 1,length(s)-1)"0"] += 1; sum[substr(s, 1,length(s)-1)"0"] += ($13+$14)/2 } END {num = asorti(sum,sum2); for (i=1; i<=num; i++) print sum2[i]","sum[sum2[i]]/count[sum2[i]]}' ./Exness_"$symbol"_$year.csv > ./Exness_"$symbol"_"$year"_raw.csv
		rm ./Exness_"$symbol"_$year.csv;
	   done ;
	   cd ./../../../Scripts/;
   fi
done
