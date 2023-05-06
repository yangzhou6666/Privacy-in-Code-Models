python ../extract/merge.py  \
--model codeparrot/codeparrot-small \
--top_k 40  \
--temperature 1.0  \
--seq_len 512 \
--prompt_mode direct_prompt \
--prompt fbf9a2284a36c954f3609c8fce2a720476a2e82f \ #a0878ed405636ae5d6f4a905b3a2aa8bd4a6d66e
--internet-sampling \

python ../clone/scripts.py \
--model codeparrot/codeparrot-small   \
--top_k 40     \
--temperature 1.0     \
--seq_len 512 --tool_path /mnt/hdd1/zyang/Privacy-in-Code-Models/clone/simian-2.5.10.jar \
--prompt_mode direct_prompt \
--prompt_hash fbf9a2284a36c954f3609c8fce2a720476a2e82f \
--internet-sampling \

# 下面变成 da39a3ee5e6b4b0d3255bfef95601890afd80709，是因为上面该为prompt而不是prompt_hash
mv ./log/save/codeparrot/codeparrot-small-temp1.0-len512-k40/da39a3ee5e6b4b0d3255bfef95601890afd80709/1-20096/  ./log/save/codeparrot/codeparrot-small-temp1.0-len512-k40/da39a3ee5e6b4b0d3255bfef95601890afd80709/0-200000/ 
cp ../log/analyze.py ./log/
cd log
python analyze.py  --model codeparrot/codeparrot-small     --top_k 40     --temperature 1.0     --seq_len 512 --prompt_mode direct_prompt --prompt fbf9a2284a36c954f3609c8fce2a720476a2e82f --internet-sampling



python ../extract/merge.py  \
--model codeparrot/codeparrot-small \
--top_k 40  \
--temperature 1.0  \
--seq_len 512 \
--prompt_mode direct_prompt \
--prompt e326ca81facf0916b79371d6f7ded1a6436bd9fe \
--internet-sampling \


python ../clone/scripts.py \
--model codeparrot/codeparrot-small   \
--top_k 40     \
--temperature 1.0     \
--seq_len 512 --tool_path /mnt/hdd1/zyang/Privacy-in-Code-Models/clone/simian-2.5.10.jar \
--prompt_mode direct_prompt \
--prompt e326ca81facf0916b79371d6f7ded1a6436bd9fe \
--internet-sampling \

python analyze.py  --model codeparrot/codeparrot-small     --top_k 40     --temperature 1.0     --seq_len 512 --prompt_mode direct_prompt --prompt e326ca81facf0916b79371d6f7ded1a6436bd9fe --internet-sampling --file_end_number 5000 --file_begin_number 1



=======

python ../extract/merge.py  \
--model codeparrot/codeparrot \
--top_k 40  \
--temperature 1.0  \
--seq_len 512 \
--prompt_mode direct_prompt \
--prompt e326ca81facf0916b79371d6f7ded1a6436bd9fe \
--internet-sampling \


python ../clone/scripts.py \
--model codeparrot/codeparrot   \
--top_k 40     \
--temperature 1.0     \
--seq_len 512 --tool_path /mnt/hdd1/zyang/Privacy-in-Code-Models/clone/simian-2.5.10.jar \
--prompt_mode direct_prompt \
--prompt e326ca81facf0916b79371d6f7ded1a6436bd9fe \
--internet-sampling \

cd log
python analyze.py  --model codeparrot/codeparrot     --top_k 40     --temperature 1.0     --seq_len 512 --prompt_mode direct_prompt --prompt e326ca81facf0916b79371d6f7ded1a6436bd9fe --internet-sampling --file_end_number 20000 --file_begin_number 1
python sample.py  --model codeparrot/codeparrot     --top_k 40     --temperature 1.0     --seq_len 512 --prompt_mode direct_prompt --prompt e326ca81facf0916b79371d6f7ded1a6436bd9fe --internet-sampling --file_end_number 20000 --file_begin_number 1

====
python ../../extract/merge.py  \
--model codeparrot/codeparrot \
--top_k 40  \
--temperature 1.0  \
--seq_len 512 \
--prompt_mode direct_prompt \
--prompt ffffffffffffffffffffffffffffffffffffffff \
--internet-sampling \


python ../../clone/scripts.py \
--model codeparrot/codeparrot   \
--top_k 40     \
--temperature 1.0     \
--seq_len 512 --tool_path /mnt/hdd1/zyang/Privacy-in-Code-Models/clone/simian-2.5.10.jar \
--prompt_mode direct_prompt \
--prompt ffffffffffffffffffffffffffffffffffffffff \
--internet-sampling \

cd log
python analyze.py  --model codeparrot/codeparrot     --top_k 40     --temperature 1.0     --seq_len 512 --prompt_mode direct_prompt --prompt ffffffffffffffffffffffffffffffffffffffff --internet-sampling --file_end_number 20000 --file_begin_number 1

python sample.py  --model codeparrot/codeparrot     --top_k 40     --temperature 1.0     --seq_len 512 --prompt_mode direct_prompt --prompt ffffffffffffffffffffffffffffffffffffffff --internet-sampling --file_end_number 20000 --file_begin_number 1