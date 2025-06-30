#!/bin/bash
function_names=("Branin" "DropWave" "Levy" "Ackley" "Rastrigin"  "Rosenbrock")
policies=("Random-Search" "GP-UCB" "PFNs4BO-HEBO") #"ourPFNs")
# Get the length of the array
function_names_length=${#function_names[@]}
policies_length=${#policies[@]}

# Function to check for idle partitions
find_idle_partition() {
  sinfo -h -o "%P %a %l %t %D" | grep "idle" | awk '{print $1}' | head -n 1
}

# Loop through the array elements
for ((i=0; i<function_names_length; i++)); do
    for ((j=0; j<policies_length; j++)); do
        # Get the current dataset and policy
        function_name="${function_names[i]}"
        policy="${policies[j]}"
	echo "Running dataset: $function_name, policy: $policy"
        # Define the output and error file paths based on input parameters
        logpath="../results/logs/${function_name}/"
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

        output=$(eval sbatch --partition=$partition --error=$error_path --output=$output_path runoncluster.sh $function_name $policy)
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
