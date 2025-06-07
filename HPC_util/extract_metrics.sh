#!/bin/bash

# Input and output files
INFILE="out/ML-training_KL_improved_2.out"
BATCH_OUT="batch_metrics.csv"
EPOCH_OUT="epoch_metrics.csv"
TEMP_LR="temp_learning_rates.csv"
TEMP_METRICS="temp_metrics.csv"

# Extract batch metrics
grep -A 2 "Loss:" $INFILE | awk '
BEGIN { print "batch,epoch,loss,grad_norm,clip_percent" }
/Loss:/ {
    loss = $2
    getline
    split($0, a, " ")
    grad_norm = a[5]
    getline
    if ($0 ~ /batch:/) {
        match($0, /batch: ([0-9]+)/, batch_match)
        match($0, /epoch: ([0-9]+)/, epoch_match)
        match($0, /clipping percentage: ([0-9.]+)%/, clip_match)
        batch = batch_match[1]
        epoch = epoch_match[1]
        clip_percent = clip_match[1]
        print batch "," epoch "," loss "," grad_norm "," clip_percent
    }
}' > $BATCH_OUT

# Extract learning rates with padded epoch numbers
grep "Learning Rate:" $INFILE | awk '
BEGIN { print "epoch,learning_rate" }
{
    epoch = sprintf("%04d", NR - 1)  # Pad with zeros to 4 digits
    learning_rate = $3
    print epoch "," learning_rate
}' > $TEMP_LR

# Extract other epoch metrics with padded epoch numbers
awk '
BEGIN { print "epoch,train_loss,valid_kl,max_prob_error" }
/Epoch [0-9]*\/300:/ {
    epoch = sprintf("%04d", substr($2, 1, index($2, "/") - 1))  # Pad with zeros to 4 digits
    getline
    train_loss = $3
    getline
    valid_kl = $3
    getline
    max_prob_error = $4
    print epoch "," train_loss "," valid_kl "," max_prob_error
}' $INFILE > $TEMP_METRICS

# Sort both files (excluding headers)
tail -n +2 $TEMP_METRICS | sort -t, -k1,1 > ${TEMP_METRICS}.sorted
tail -n +2 $TEMP_LR | sort -t, -k1,1 > ${TEMP_LR}.sorted

# Join the sorted files
join -t, -a1 -e "" -o 1.1,1.2,1.3,1.4,2.2 ${TEMP_METRICS}.sorted ${TEMP_LR}.sorted > $EPOCH_OUT

# Add header and remove padding from epoch numbers
sed -i '1i epoch,train_loss,valid_kl,max_prob_error,learning_rate' $EPOCH_OUT
sed -i 's/^0*\([0-9]\),/\1,/' $EPOCH_OUT

# Clean up temporary files
rm $TEMP_LR ${TEMP_LR}.sorted $TEMP_METRICS ${TEMP_METRICS}.sorted


