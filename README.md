
## ProG

python attack_to_pretrain_distribution.py --dataset CiteSeer --gnn_type TransformerConv --usage_par prompt --attack_type ours --makeTE trigger_graph --reg_param_trigger_n 0.05 --reg_param_trigger_f 0.05 --reg_param_gnn 0.5 --num_trigger_node 3 --defense_mode none --sim_thre 0 &

## GraphPrompt

python graphprompt_distribution.py --dataset CiteSeer --gnn_type TransformerConv --usage_par www23 --attack_type ours --makeTE trigger_graph --reg_param_trigger_n 0.05 --reg_param_trigger_f 0.05 --reg_param_gnn 0.5 --num_trigger_node 3 --defense_mode none --sim_thre 0 &
