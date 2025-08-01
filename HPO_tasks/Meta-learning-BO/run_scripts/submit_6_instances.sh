#!/bin/bash
search_space_ids=('4796' '5860' '5906'  '5527'  '5889'  '5859')
search_space_ids=('4796' '5889'  '5859')
#9 test 5965, 7609, 5889, 6794, 5859, 4796, 7607, 5636, and 5970.
#9 valid 5527, 5891, 5906, 5971, 6767, 6766, and 5860
policies=("pPFNs4BO-simple") #("PFNs4BO-PI-o" "PFNs4BO-PI" "PFNs4BO-PI-IW" "PFNs4BO-EI-IW") #  "mPFNs4BO" "MALIBO"
evaluations=60

seeds=("test0" "test1" "test2" "test3" "test4")

seeds_length=${#seeds[@]}
search_space_ids_length=${#search_space_ids[@]}
policies_length=${#policies[@]}

# Function to check for idle partitions
find_idle_partition() {
  sinfo -h -o "%P %a %l %t %D" | grep "idle" | awk '{print $1}' | head -n 1
}

# Loop through the array elements
for ((i=0; i<seeds_length; i++)); do
    for ((k=0; k<search_space_ids_length; k++)); do
        for ((j=0; j<policies_length; j++)); do
            # Get the current dataset and policy
            seed="${seeds[i]}"
            policy="${policies[j]}"
            search_space_id="${search_space_ids[k]}"
            echo "Running dataset: $search_space_id, policy: $policy, seed: $seed"
            # Define the output and error file paths based on input parameters
            logpath="../results/logs6/${search_space_id}/"
            mkdir -p "$logpath"
            output_path="${logpath}/${policy}_${seed}_%j_out.txt"
            error_path="${logpath}/${policy}_${seed}_%j_err.txt"
            # Get an idle partition
            # partition=$(find_idle_partition)
            # if [ -z "$partition" ]; then
            #     echo "No idle partitions found, run on cpu."
            #     partition="cpu-galvani"
            # else
            #     echo "Idle partition found:$partition"
            # fi
            partition="2080-galvani"

            output=$(eval sbatch --partition=$partition --error=$error_path --output=$output_path --exclude=galvani-cn119 runoncluster.sh $search_space_id $policy $seed "results/hpob6" $evaluations)
            if [[ $output == *"Submitted batch job"* ]]; then
                echo $output
                echo "Successfully submitted job"
                sleep 1
            else
                echo "Oh no! Detected error. Wait 5 mins and try again."
                # Set your_variable to some other value if needed
                ((j--))
                sleep 3600
            fi
        done
    done
done