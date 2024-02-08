# 2024-Final-Project
2024-Final-Project


## Download the raw data
This script will download all the public symbols from the company and:
1. Discard the ones that do not have year 2023
2. Merge the different ticks in 10 min slots, doing the median from the given values

To execute it:
``
cd ./Scripts
chmod +x ./*
./Download_all_symbols_slot_10m.sh
``
