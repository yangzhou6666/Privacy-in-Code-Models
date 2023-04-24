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