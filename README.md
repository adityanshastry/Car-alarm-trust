# Car Alarm Trust
This project measures the effectiveness of an alarm system used to warn drivers of an impending collision. This problem is treated as a Reinforcement Learning problem. The state, actions, and rewards are designed based on an ideal outcome of a collision detection and warning system. The data for this problem is taken from Fatality Analysis Reporting System (FARS), maintained by National Highway Traffic Safety Administration. The algorithms used are SARSA, and Q_learning. This project measures the effectiveness of an alarm in 2 settings - Consistent (alarm is on only when there is a collision), and Inconsistent (alarm is sounded randomly). This project effectively shows that Reinforcement Learning can be used to tackle this problem, with the data supplied from a car simluator in the future. 

A more detailed report is present here - [Project Report](https://drive.google.com/file/d/1msm0MryHJoJhggAmGTcICJS_qMY8J5ig/view)

Execution instructions:
1) Clone/download the project in a desired location
2) cd Car_alarm_trust
3) export PYTHONPATH="${PYTHONPATH}:directory/where/Car_alarm_trust/is/placed"
4) python common/run_trainer.py <algorithm_option> <num_jobs> <alarm_consistency>

algorithm_option: 0 or 1 to indicate the algorithm, which is SARSA or Q-Learning respectively
<num_jobs>: integer value >= 1 to indicate the number of parallel jobs to run
alarm_consistency: 0 or 1 to indicate the alarm consistency, which is False or True respectively

Details about the code:
1) Language: Python 2.7
2) Libraries: Numpy (for fast mathematical operations), Matplotlib (for graph plots), joblib (for parallel runs)
