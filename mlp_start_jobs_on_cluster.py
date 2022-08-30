import os

if __name__ == '__main__':
    

    lid_ds_versions = [
        'LID-DS-2019',
        'LID-DS-2021'
        ]
    
    job_counter = 1
    
    # Default is all.
    # number_of_play_back_alarms = [
    #     '1',
    #     '2',
    #     '3',
    #     'all'
    #     ]
    
    mlp_configs = [
        # '0',
        # '1',
        '2',
        '3',
        '4'
        ]
    
    independent_validations = [
        'True', 
        'False'
        ]
    
    learning_rates = [
        0.003,
        0.005,
        0.007
        ]
    
    modes = [
        'retraining',
        'revalidation',
        'conceptdrift'
        ]

    
    freezing = [
        'True', 
        'False'
        ]
    
    result_path = 'results_mlp'
    
    for version in lid_ds_versions:
        if version=="LID-DS-2019":
            scenario_names = [
                "Bruteforce_CWE-307",
                "CVE-2012-2122",
                "CVE-2014-0160",
                "CVE-2017-7529",
                "CVE-2018-3760",
                "CVE-2019-5418",
                "EPS_CWE-434",
                "PHP_CWE-434",
                "SQL_Injection_CWE-89",
                "ZipSlip"
           ]
        else:           
           scenario_names = [
               "Bruteforce_CWE-307",
               "CVE-2012-2122",
               "CVE-2014-0160",
               "CVE-2017-7529",
               "CVE-2017-12635_6",
               "CVE-2018-3760", 
               "CVE-2019-5418",
               "CVE-2020-9484",
               "CVE-2020-13942",
               "CVE-2020-23839",
               "CWE-89-SQL-injection",
               "Juice-Shop",
               "EPS_CWE-434",
               "PHP_CWE-434", 
               "ZipSlip",
           ]    
           
        # for play_back_count in number_of_play_back_alarms:
        for config in mlp_configs:
                for learning_rate in learning_rates:
                    for validation_mode in independent_validations:
                        for mode in modes:
                            for freeze in freezing:
                                for scenario in scenario_names:
                                    if mode == 'revalidation' or mode  == 'retraining': 
                                        if learning_rate != 0.003: # Ergibt anders keinen Sinn bei neuem Schwellenwert
                                            continue
                                    if mode == 'revalidation' and freeze == 'True':
                                        continue # Ergibt keinen Sinn bei neuem Schwellenweert.
                                    
                                    command = f'sbatch --job-name=mlp{job_counter:03} evaluation_mlp.job {version} {scenario} {config} {result_path} {learning_rate} {mode} {freeze} {validation_mode}'
                                    os.system(command)

                                    job_counter += 1
    