#!/bin/bash

# Debug mode flag (set to true for testing)
DEBUG=false #true

# Base paths
BASE_DIR="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide"
MARTINI_DIR="${BASE_DIR}/Tetrapeptide/MARTINI22"
SCRIPT_DIR="${BASE_DIR}/Tetrapeptide/MARTINI_Setup_script"

# Read the sequences file
SEQUENCES_FILE="${BASE_DIR}/DL_for_Peptide/ML_ddEdge/selected_sequences_tetrapeptide.txt"

# Read the file and process each sequence
while IFS= read -r line; do
    # Skip comments and empty lines
    [[ $line =~ ^#.*$ ]] && continue
    [[ -z $line ]] && continue
    
    # Extract sequence from the line (assuming format "Type_Rank: Sequence (Value)")
    seq=$(echo $line | sed 's/.*: \([A-Z,]*\) .*/\1/' | tr -d ',')
    
    # Extract amino acids in positional order
    pos1=${seq:0:1}
    pos2=${seq:1:1}
    pos3=${seq:2:1}
    pos4=${seq:3:1}
    
    # Find position of C
    if [ "$pos1" = "C" ]; then
        c_pos=1
    elif [ "$pos2" = "C" ]; then
        c_pos=2
    elif [ "$pos3" = "C" ]; then
        c_pos=3
    elif [ "$pos4" = "C" ]; then
        c_pos=4
    else
        echo "Error: No C found in sequence $seq"
        continue
    fi
    
    # Set target directory based on C position
    TARGET_DIR="${MARTINI_DIR}/Tetrapeptide_mon_C${c_pos}"
    
    # Create output directory if it doesn't exist
    mkdir -p ${TARGET_DIR}/out
    
    # Change to target directory
    cd ${TARGET_DIR}
    
    if [ "$DEBUG" = true ]; then
        echo "Sequence: $seq"
        echo "Current Directory: $(pwd)"
        echo "---"
    else
        # Submit the job
        sbatch --job-name="${pos1}_${pos2}_${pos3}_${pos4}" \
            --mail-user=yuanmis1@uci.edu --mail-type=FAIL \
            --account=dtobias_lab \
            --partition=standard \
            --nodes=1 \
            --ntasks-per-node=1 \
            --cpus-per-task=40 \
            --time=24:00:00 \
            --out=out/${pos1}_${pos2}_${pos3}_${pos4}.out \
            --wrap="bash ${SCRIPT_DIR}/Cys_unstapled_tetrapeptide_MARTINI_run.sh ${pos1} ${pos2} ${pos3} ${pos4}"
    fi
done < "$SEQUENCES_FILE" 