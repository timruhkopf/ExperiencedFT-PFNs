#!/bin/bash
policies=("PFNs4BO" "MALIBO" "mPFNs4BO")
policy="PFNs4BO" # Default policy

search_space_ids=("4796" "5527" "5636" "5859" "5860" "5891" "5906" "5965" "5970" "5971" "6766" "6767" "6794" "7607" "7609" "5889") 
search_space_ids_length=${#search_space_ids[@]}

test_seeds=("test0" "test1" "test2" "test3" "test4")

test_seeds_length=${#test_seeds[@]}

# Function to check for idle partitions
find_idle_partition() {
  sinfo -h -o "%P %a %l %t %D" | grep "idle" | awk '{print $1}' | head -n 1
}

# Loop through the array elements
for ((i=0; i<search_space_ids_length; i++)); do
    for ((j=0; j<test_seeds_length; j++)); do
        # Get the current dataset and policy
        search_space_id="${search_space_ids[i]}"
        seed="${test_seeds[j]}"
	echo "Running dataset: $search_space_id, policy: $policy"
        # Define the output and error file paths based on input parameters
        logpath="../results/logs/${search_space_id}/"
        mkdir -p "$logpath"
        output_path="${logpath}/job%j_out.txt"
        error_path="${logpath}/job%j_err.txt"
        # Get an idle partition
        # partition=$(find_idle_partition)
        # if [ -z "$partition" ]; then
        #     echo "No idle partitions found, run on cpu."
        #     partition="cpu-galvani"
        # else
        #     echo "Idle partition found:$partition"
        # fi
        partition="2080-galvani"

        output=$(eval sbatch --partition=$partition --error=$error_path --output=$output_path runoncluster.sh $search_space_id $policy $seed)
        if [[ $output == *"Submitted batch job"* ]]; then
            echo $output
            echo "Succesfully"
            sleep 1
        else
            echo "Oh no! Detected error. Wait 5 mins and try again."
            # Set your_variable to some other value if needed
            ((j--))
            sleep 3600
        fi
    done
done
