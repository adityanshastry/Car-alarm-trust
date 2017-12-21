# Car_alarm_trust
Measure effectiveness of an alarm to prevent car crashes

Execution instructions:

1) cd Car_alarm_trust
2) export export PYTHONPATH="${PYTHONPATH}:directory/where/Car_alarm_trust/is/placed"
3) python common/run_trainer.py <algorithm_option> <num_jobs> <alarm_consistency>

algorithm_option: 0 - sarsa, 1 - q_learning
alarm_consistency: 0 - False, 1 - True