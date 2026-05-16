for curr_part in {0..6}; do
    MODEL_NAME=llama-3-8b-inst MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct DEVICES="0" PART=${curr_part} bash run_compute_weights.sh
done