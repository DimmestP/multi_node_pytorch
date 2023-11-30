#!/bin/bash

cd train

image_label_regex='[a-zA-Z, ]+$'

for foldername in n*
do
        metadata_line=$(grep "$foldername" ../imagenet_metadata.txt)
        image_label=$( echo "$metadata_line" | grep -oE "$image_label_regex")

        for filename in $foldername/*
        do
                echo "$filename $image_label" >> image_labels.txt
        done
done
