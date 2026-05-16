for curr_part in {0..7}; do
    MODEL_NAME=llama-3-8b MODEL_PATH=princeton-nlp/Llama-3-Base-8B-SFT DEVICES="0" PART=${curr_part} bash run_compute_weights.sh
done