from multiprocessing import cpu_count
import os
import sys
import json
import datetime
from json import JSONEncoder
from pprint import pprint
from argparse import ArgumentParser
from typing import List
import torch
import numpy
import random
from copy import deepcopy
from tqdm.contrib.concurrent import process_map
from functools import reduce
from time import time
from algorithms.decision_engines.ae import AE, AEMode
from datetime import datetime
from algorithms.ids import IDS
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.one_hot_encoding import OneHotEncoding
from algorithms.persistance import save_to_json
from algorithms.performance_measurement import Performance

from dataloader.dataloader_factory import dataloader_factory
from dataloader.syscall_2021 import Syscall2021
from dataloader.direction import Direction

from torch.multiprocessing import set_start_method

from dataloader.syscall import Syscall

# CONSTANTS
LEARNING_RATE_CONSTANT = 0.001

    
    
class FalseAlertResult:
    def __init__(self, name, syscalls) -> None:
        self.name = name 
        self.syscalls = syscalls
        self.structure = {name: syscalls}
        
    def add(left: 'FalseAlertResult', right: 'FalseAlertResult') -> 'FalseAlertResult':
        result = FalseAlertResult(right.name, right.syscalls)
        result.structure = left.structure        
        result.structure[right.name] = right.syscalls

        return result
    
    def __repr__(self) -> str:
        return f"FalseAlertResult: Name: {self.name}, Structure: {self.structure}"


# Brauche ich um die Systemcalls dem Trainingsdatensatz hinzuzufügen
class ArtificialRecording:  
    def __init__(self, name, syscalls):
         self.name = name
         self._syscalls = syscalls
         
    def syscalls(self) -> list:
        return self._syscalls
        
    def __repr__(self) -> str:
        return f"ArtificialRecording, Name: {self.name}, Nr. of Systemcalls: {len(self._syscalls)}"



class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def json_default(value):
    if isinstance(value, datetime.date):
        return dict(year=value.year, month=value.month, day=value.day)
    else:
        return value.__dict__

# Speichere die künstlichen recordings um auch lange Laufzeiten möglich zu machen.
def save_artifical_recordings(recordings, path: str):
    """ Speichert die Künstlichen Recordings ab.

    Args:
        recordings (List[BaseRecording]): Die zu speichernden, künstlichen Recordings.
        path (str): Path to the file.
    """

    listofresults = []
    # Erzeuge die richtige Form
    for recording in recordings: 
        result = {
        'algorithm': str,
        'recording_name': str,
        'syscalls': [Syscall]
        }
            
        result['algorithm'] = 'AE'
        result['recording_name'] = recording.name
        result['syscalls'] = recording.syscalls()
        listofresults.append(result)
    pprint(recording.syscalls())

    # Wenn der Ordner noch nicht existiert, erzeuge ihn.
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    with open(path, 'w') as file:
        json.dump(listofresults, file, indent=2, cls=MyEncoder, default=json_default)


# def custom_decode(value):
#     pprint(f'Value here: {value}')

# Lade die schon erzeugten künstlichen Recordings.
def load_artifical_recordings(path) -> List[ArtificialRecording]:
    with open(path, 'r') as file:
        # result = json.load(file, object_hook=custom_decode)
        recordings = json.load(file)
        # pprint(result)
        # pprint(type(result))

        all_recordings = []
        for recording in recordings:
            # pprint(entry)
            # pprint(type(entry))
            # pprint(recording['syscalls'][0])
            
            systemcall_list = []
            for systemcallAsJson in recording['syscalls']:
                # Construct 2021 Systemcall
                syscall = Syscall2021(recording_path=systemcallAsJson['recording_path'], syscall_line=systemcallAsJson['syscall_line'])
                syscall._direction = systemcallAsJson['_direction']
                syscall._line_list = systemcallAsJson['_line_list']
                syscall._name = systemcallAsJson['_name']
                syscall._params = systemcallAsJson['_params']
                syscall._process_id = systemcallAsJson['_process_id']
                syscall._process_name = systemcallAsJson['_process_name']
                syscall._thread_id = systemcallAsJson['_thread_id']
                
                # Muss ich den Timestamp richtig initialisieren? Könnte einfach den 0. Eintrag aus syscall_line nehmen und den konvertieren. An sich funktionieren die aber noch nicht richtig. Produzieren die trotzdem die Anomaly-Werte?
                syscall._timestamp_datetime = int(systemcallAsJson['syscall_line'][0])
                syscall._timestamp_unix = datetime.fromtimestamp(int(systemcallAsJson['syscall_line'][0]) * 10 ** -9) 
                syscall._user_id = systemcallAsJson['_user_id']
                
                syscall.line_id = systemcallAsJson['line_id']
                

                systemcall_list.append(syscall)
                # pprint(f'Constructed Systemcall: {syscall}, original was: {systemcallAsJson}')
            
            artificial_recording = ArtificialRecording(name = recording['recording_name'], syscalls=systemcall_list)
            all_recordings.append(artificial_recording)
        # pprint(all_recordings[0]._syscalls[0])
    return all_recordings
    
    


# Argument Parser für bessere die Nutzbarkeit eines einzelnen Scripts, welches dann auf dem Cluster gecallt wird.
def parse_cli_arguments(): 
    parser = ArgumentParser(description='Playing Back False-Positives Pipeline')
    parser.add_argument('--version', '-v', choices=['LID-DS-2019', 'LID-DS-2021'], required=True, help='Which version of the LID-DS?')
    parser.add_argument('--scenario', '-s', choices=['Bruteforce_CWE-307',
                                                     'CVE-2012-2122',
                                                     'CVE-2014-0160',
                                                     'CVE-2017-7529',
                                                     'CVE-2017-12635_6',
                                                     'CVE-2018-3760',
                                                     'CVE-2019-5418',
                                                     'CVE-2020-9484',
                                                     'CVE-2020-13942',
                                                     'CVE-2020-23839',
                                                     'CWE-89-SQL-injection',
                                                     "SQL_Injection_CWE-89",
                                                     'EPS_CWE-434',
                                                     'Juice-Shop',
                                                     'PHP_CWE-434',
                                                     'ZipSlip'], required=True, help='Which scenario of the LID-DS?')
    # parser.add_argument('--algorithm', '-a', choices=['stide',
    #                                                   'mlp',
    #                                                   'ae',
    #                                                   'som'], required=True, help='Which algorithm shall perform the detection?')
    parser.add_argument('--play_back_count_alarms', '-p' , choices=['1', '2', '3', 'all'], default='all', help='Number of False Alarms that shall be played back or all.')
    parser.add_argument('--results', '-r', default='results', help='Path for the results of the evaluation')
    parser.add_argument('--base-path', '-b', default='/work/user/lz603fxao/Material', help='Base path of the LID-DS')
    parser.add_argument('--config', '-c', choices=['0', '1', '2'], default='0', help='Configuration of the MLP which will be used in this evaluation')
    parser.add_argument('--use-independent-validation', '-u', choices=['True', 'False'], default='False', required=False, help='Indicates if the AE will use the validation dataset for threshold AND stop of training or only for threshold.')
    parser.add_argument('--learning-rate', '-l', default=LEARNING_RATE_CONSTANT, type=float, choices=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009], help='Learning rate of the mlp algorithm of the new IDS')
    parser.add_argument('--mode', '-m', default = 'retraining', choices=['retraining', 'revalidation', 'conceptdrift'], help='Decides in which mode the playing back will work.')
    parser.add_argument('--freeze-on-retraining', '-f', default='False', choices=['True', 'False'], help='After the retraining of the IDS, will you freeze the original threshold or calculate a new one?')

    return parser.parse_args()
    

# Startpunkt
if __name__ == '__main__':
   
    set_start_method('forkserver')
    
    args = parse_cli_arguments()
    
    # Verwaltet hier die Abbrüche irreführender Kombinationen.
     
    # Check ob die Kombination vorhanden ist.
    if args.version == 'LID-DS-2019' and args.scenario in ['CWE-89-SQL-injection', 'CVE-2020-23839', 'CVE-2020-9484', 'CVE-2020-13942' , 'Juice-Shop' , 'CVE-2017-12635_6']:
        sys.exit('This combination of LID-DS Version and Scenario aren\'t available.')
     
    # Check ob ins Valid-Set gespielt wird, dabei aber Freeze Threshold verlangt wird oder eine andere learning rate.
    if args.mode == 'revalidation':
        if args.freeze_on_retraining == 'True':
            sys.exit('This combination can\'t be played since we want to play back the examples in the validation set. Therefore we MUST NOT freeze the threshold.')
        elif args.learning_rate != LEARNING_RATE_CONSTANT:
            sys.exit(f'Can\'t change the learning rate when we play back in the validation set. This should only influence the threshold. Default learning rate is {LEARNING_RATE_CONSTANT}.')
    
    if args.mode == 'retraining':
        if args.learning_rate != LEARNING_RATE_CONSTANT:
            sys.exit(f'Can\'t change the learning rate when use retraining mode. Default learning rate is {LEARNING_RATE_CONSTANT}.')
    
     
    pprint("Performing Host-based Intrusion Detection with AE:")
    pprint(f"Version: {args.version}") 
    pprint(f"Scenario: {args.scenario}")
    pprint(f"Configuration: {args.config}")
    pprint(f"State of independent validation: {args.use_independent_validation}")
    pprint(f"Number of maximal played back false alarms: {args.play_back_count_alarms}")
    pprint(f"Playing back mode is: {args.mode}")
    if args.mode == 'conceptdrift':
        pprint(f"Learning-Rate of the new IDS: {args.learning_rate}")
    pprint(f"Treshold freezing on seconds IDS: {args.freeze_on_retraining}")
    pprint(f"Results path: {args.results}")
    pprint(f"Base path: {args.base_path}")
    
    scenario_path = f"{args.base_path}/{args.version}/{args.scenario}"     
    dataloader = dataloader_factory(scenario_path,direction=Direction.BOTH) # Results differ, currently BOTH was the best performing
    
    #--------------------
        
    # Configuration of chosen decision engines. Choosing best configs in MLPs for equalness. Grimmers Paper.
    ####################
    
    # load FalseAlertResults into All_recordings
    false_alert_path = f'{args.base_path}/False_Alerts/{args.version}/{args.scenario}/ae_false_alerts.dump'
    all_recordings = load_artifical_recordings(false_alert_path)
    
    dataloader.set_retraining_data(all_recordings)
    # dataloader.set_revalidation_data(all_recordings)
    
    settings_dict = {} # Enthält die Konfig-Infos
    
    ##################################### Config 0 ######################################### 
    if args.config == '0':
            
        ngram_length = 3
        thread_aware = True
        batch_size = 256
        window_length = 1
    
    ##################################### Config 1 ######################################### 
    elif args.config == '1':
            
        ngram_length = 5
        thread_aware = True
        batch_size = 256
        window_length = 1
    
    ##################################### Config 2 ######################################### 
    elif args.config == '2':
            
        ngram_length = 7
        thread_aware = True
        batch_size = 256
        window_length = 1
        
    else:
        exit('Unknown configuration of MLP. Exiting.')
        
    settings_dict['ngram_length'] = ngram_length
    settings_dict['thread_aware'] = thread_aware
    settings_dict['batch_size'] = batch_size
    settings_dict['window_length'] = window_length
        
    # Building Blocks
    inte = IntEmbedding()
        
    ohe = OneHotEncoding(inte)
    ngram = Ngram([ohe], thread_aware, ngram_length) 
    ae = AE(ngram, mode=AEMode.LOSS, batch_size=batch_size, max_training_time=172800)
    decision_engine = ae    
        
  
    
    # Stopping Randomness
    torch.manual_seed(0)
    random.seed(0)
    numpy.random.seed(0)
    torch.use_deterministic_algorithms(True)
    
    # IDS
    ###################
    generate_and_write_alarms = True
    ids = IDS(data_loader=dataloader,
            resulting_building_block=decision_engine,
            create_alarms=generate_and_write_alarms,
            plot_switch=False)
    
    ###################
    pprint("At evaluation:")
    ids.determine_threshold()   
    
    
    
    performance = ids.detect_parallel()
    
    pprint(performance)
    results = performance.get_results()
    pprint(results)

    # Preparing results
    config_name = f"algorithm_ae_c_{args.config}_lr_{args.learning_rate}_n_{ngram_length}_t_{thread_aware}"
    
    
    # Enrich results with configuration
    results['algorithm'] = 'ae'
    for key in settings_dict.keys():
        results[key] = settings_dict[key]
        
    results['config'] = ids.get_config() # Produces strangely formatted Config-Print
    results['scenario'] =  args.version + "/" + args.scenario
    result_path = f"{args.results}/results_ae_config_{args.config}_lr_{args.learning_rate}_{args.version}_{args.scenario}.json"

    # Saving results
    save_to_json(results, result_path) 
    with open(f"{args.results}/alarms_{config_name}_{args.version}_{args.scenario}.json", 'w') as jsonfile:
        json.dump(performance.alarms.get_alarms_as_dict(), jsonfile, default=str, indent=2)