#!/bin/bash
search_space_id="6767" #("4796" "5527" "5636" "5859" "5860" "5891" "5906" "5965" "5970" "5971" "6766" "6767" "6794" "7607" "7609" "5889") 
policies=("PFNs4BO") #  "mPFNs4BO" "MALIBO"
policies_length=${#policies[@]}

seeds=("test0" "test1" "test2" "test3" "test4")
seeds_length=${#seeds[@]}

# Function to check for idle partitions
find_idle_partition() {
  sinfo -h -o "%P %a %l %t %D" | grep "idle" | awk '{print $1}' | head -n 1
}

# Loop through the array elements
for ((i=0; i<seeds_length; i++)); do
    for ((j=0; j<policies_length; j++)); do
        # Get the current dataset and policy
        seed="${seeds[i]}"
        policy="${policies[j]}"
	echo "Running dataset: $search_space_id, policy: $policy, seed: $seed"
        # Define the output and error file paths based on input parameters
        logpath="../results/logs/${search_space_id}/"
        mkdir -p "$logpath"
        output_path="${logpath}/${seed}_%j_out.txt"
        error_path="${logpath}/${seed}_%j_err.txt"
        # Get an idle partition
        partition=$(find_idle_partition)
        if [ -z "$partition" ]; then
            echo "No idle partitions found, run on cpu."
            partition="cpu-galvani"
        else
            echo "Idle partition found:$partition"
        fi
        #partition="2080-galvani"

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
