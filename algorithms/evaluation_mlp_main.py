from multiprocessing import cpu_count
import os
import sys
import json
from pprint import pprint
from argparse import ArgumentParser
import torch
import numpy
import random
from copy import deepcopy
from tqdm.contrib.concurrent import process_map
from functools import reduce
from time import time

from algorithms.decision_engines.mlp import MLP
from algorithms.ids import IDS
from algorithms.features.impl.select import Select
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.one_hot_encoding import OneHotEncoding
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from algorithms.persistance import save_to_json
from algorithms.performance_measurement import Performance

from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction


# CONSTANTS
LEARNING_RATE_CONSTANT = 0.003



class FalseAlertContainer:
    def __init__(self, alarm, alarm_recording_list, window_length, ngram_length, thread_aware) -> None:
        self.alarm = alarm
        self.alarm_recording_list = alarm_recording_list
        self.window_length = window_length
        self.ngram_length = ngram_length
        self.thread_aware = thread_aware
    
    
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


class Container:
    def __init__(self, ids, recording):
        self.ids = ids
        self. recording = recording
        
# Brauche ich um die Systemcalls dem Trainingsdatensatz hinzuzuf??gen
class ArtificialRecording:  
    def __init__(self, name, syscalls):
         self.name = name
         self._syscalls = syscalls
         
    def syscalls(self) -> list:
        return self._syscalls
        
    def __repr__(self) -> str:
        return f"ArtificialRecording, Name: {self.name}, Nr. of Systemcalls: {len(self._syscalls)}"

# Benutze ich um festzustellen, wann ich gen??gend Systemcalls f??r einen False-Alert habe
def enough_calls(dict, max):   
    for key in dict.keys():
        if dict[key] < max:
             return False
    return True    

def calculate(struct: Container) -> Performance:
    # Copy the whole IDS with its building blocks
    working_copy = deepcopy(struct.ids)
    # Calculate the performance on the current recording
    performance = working_copy.detect_on_recording(struct.recording)
    return performance


def construct_Syscalls(container: FalseAlertContainer) -> FalseAlertResult:
    # Nehme dir den momentanen Alarm und die Liste an Aufzeichnungen her
    alarm = container.alarm
    alarm_recording_list = container.alarm_recording_list
    
    # Bestimme die passende Aufzeichnung zum Alarm
    faster_current_basename = os.path.basename(alarm.filepath)
    for recording in alarm_recording_list:
        if os.path.basename(recording.path) == faster_current_basename:
            current_recording = recording
    
    # Extrahiere nun alle Syscalls zwischen Ende und Anfang des False-Alarms und zus??tzlich das Fenster davor
    systemcall_list = [systemcall for systemcall in current_recording.syscalls() if systemcall.line_id >= max([alarm.first_line_id - container.window_length, 0]) and systemcall.line_id <= alarm.last_line_id] 
    
    if container.thread_aware:
        # Check ob es noch Syscalls vor dem Anfang des Alarms - Fenster gibt
        number_of_required_previous_calls = container.ngram_length + container.window_length
        backwards_counter = max([alarm.first_line_id - container.window_length -1, 0]) 
        if backwards_counter != 0:
            # Sammle alle bisher gefundenen ThreadIDs, deren N-Gramme wir nun bef??llen m??ssen
            thread_id_set = set([systemcall.thread_id() for systemcall in systemcall_list])

            # Z??hle die rekonstruierten Syscalls pro ThreadID
            dict = {}
            for thread in thread_id_set:
                dict[thread] = 0
    
            temp_list = []
            for x in current_recording.syscalls():
                temp_list.append(x)
                if x.line_id == alarm.last_line_id:
                    break 
           # Erschaffe eine kleinere Liste an Systemcalls vom Anfang der Datei bis zum Ende des Alarms. Drehe sie dann um.
            temp_list.reverse() 
            # Solange wir nich f??r jede ThreadID genau ngram_length viele Syscalls gefunden haben und noch nicht am Anfang der Datei sind
            while(not enough_calls(dict, number_of_required_previous_calls) and backwards_counter != 0):
                # Finde den Systemcall, der die gleiche LineID wie unser BackwardsCounter hat
                current_call = None
                for call in temp_list:
                    if call.line_id == backwards_counter:
                        current_call = call  
                        break  
                # Ist keine solche LineID vorhanden, dann skippe diesen Wert.
                if current_call is None:
                    backwards_counter -=1 
                    continue
                # Sollten noch Systemcalls f??r diese ThreadID fehlen, dann f??ge diese am Anfang der Liste hinzu
                if current_call.thread_id() in dict.keys() and dict[current_call.thread_id()] < number_of_required_previous_calls:
                    dict[current_call.thread_id()] += 1 
                    systemcall_list.insert(0, current_call)
                        
                backwards_counter -= 1
                
    else:
        # F??lle die Liste nur mit den Calls zwischen Ende und Anfang des Alarms, plus die des Fensters und die von einer ThreadID, also ngram_length viele.
        systemcall_list = [systemcall for systemcall in current_recording.syscalls() if systemcall.line_id >= max([alarm.first_line_id - container.window_length - container.ngram_length, 0]) and systemcall.line_id <= alarm.last_line_id] # mit Fensterbetrachtung
        
    # Als ID nehmen wir das Recording und die Zeit und f??gen als Wert die extrahierten Calls zu.
    result = FalseAlertResult(f"{os.path.basename(current_recording.path)}_{str(round(time()*1000))[-5:]}", systemcall_list)
    # Wir speichern das nochmals in einem Dict um die Parallelisierung zu erm??glichen.
    result.structure[result.name] = result.syscalls
        
    return result  


# Argument Parser f??r bessere die Nutzbarkeit eines einzelnen Scripts, welches dann auf dem Cluster gecallt wird.
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
    parser.add_argument('--config', '-c', choices=['0', '1', '2', '3', '4'], default='0', help='Configuration of the MLP which will be used in this evaluation')
    parser.add_argument('--use-independent-validation', '-u', choices=['True', 'False'], default='False', required=False, help='Indicates if the MLP will use the validation dataset for threshold AND stop of training or only for threshold.')
    parser.add_argument('--learning-rate', '-l', default=LEARNING_RATE_CONSTANT, type=float, choices=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009], help='Learning rate of the mlp algorithm of the new IDS')
    parser.add_argument('--mode', '-m', default = 'retraining', choices=['retraining', 'revalidation', 'conceptdrift'], help='Decides in which mode the playing back will work.')
    parser.add_argument('--freeze-on-retraining', '-f', default='False', choices=['True', 'False'], help='After the retraining of the IDS, will you freeze the original threshold or calculate a new one?')

    return parser.parse_args()
    

# Startpunkt
if __name__ == '__main__':
    
    args = parse_cli_arguments()
    
    # Verwaltet hier die Abbr??che irref??hrender Kombinationen.
     
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
    
     
    pprint("Performing Host-based Intrusion Detection with MLP:")
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
    
    settings_dict = {} # Enth??lt die Konfig-Infos
    
    if args.config == '0':
            
            ##################################### Config 0 ######################################### 
            
            # Settings
            ngram_length = 5
            w2v_vector_size = 5
            w2v_window_size = 10
            thread_aware = True
            hidden_size = 64
            hidden_layers = 3
            batch_size = 256
            w2v_epochs = 1000
            learning_rate = 0.003
            window_length = 10

            settings_dict['ngram_length'] = ngram_length
            settings_dict['w2v_vector_size'] = w2v_vector_size
            settings_dict['w2v_window_size'] = w2v_window_size
            settings_dict['thread_aware'] = thread_aware
            settings_dict['hidden_size'] = hidden_size
            settings_dict['hidden_layers'] = hidden_layers
            settings_dict['batch_size'] = batch_size
            settings_dict['w2v_epochs'] = w2v_epochs
            settings_dict['learning_rate'] = learning_rate
            settings_dict['window_length'] = window_length

            # Building Blocks
            inte = IntEmbedding()

            w2v = W2VEmbedding(word=inte,
                           vector_size=w2v_vector_size,
                           window_size=w2v_window_size,
                           epochs=w2v_epochs,
                           thread_aware=thread_aware,
                           lid_ds_version=args.version,
                           lid_ds_scenario=args.scenario)
            
            ohe = OneHotEncoding(inte)

            ngram = Ngram([w2v], thread_aware, ngram_length + 1) 

            select = Select(ngram, start = 0, end = (w2v_vector_size * ngram_length)) 

            if args.use_independent_validation == 'True':
                mlp = MLP(select,
                    ohe,
                    hidden_size,
                    hidden_layers,
                    batch_size,
                    learning_rate,
                    True
                )
                
            # False-Fall
            else: 
                mlp = MLP(select,
                    ohe,
                    hidden_size,
                    hidden_layers,
                    batch_size,
                    learning_rate,
                    False    
                )
            
            decision_engine = StreamSum(mlp, thread_aware, window_length)
    
    ##################################### Config 1 ######################################### 
    elif args.config == '1':
            
            # Settings
            ngram_length = 3
            w2v_vector_size = 8
            w2v_window_size = 15
            thread_aware = True
            hidden_size = 32
            hidden_layers = 4
            batch_size = 256
            w2v_epochs = 1000
            learning_rate = 0.003
            window_length = 100       
            
            
            settings_dict['ngram_length'] = ngram_length
            settings_dict['w2v_vector_size'] = w2v_vector_size
            settings_dict['w2v_window_size'] = w2v_window_size
            settings_dict['thread_aware'] = thread_aware
            settings_dict['hidden_size'] = hidden_size
            settings_dict['hidden_layers'] = hidden_layers
            settings_dict['batch_size'] = batch_size
            settings_dict['w2v_epochs'] = w2v_epochs
            settings_dict['learning_rate'] = learning_rate
            settings_dict['window_length'] = window_length
            
            
            # Building Blocks

            inte = IntEmbedding()

            w2v = W2VEmbedding(word=inte,
                           vector_size=w2v_vector_size,
                           window_size=w2v_window_size,
                           epochs=w2v_epochs,
                           thread_aware=thread_aware,
                           lid_ds_version=args.version,
                           lid_ds_scenario=args.scenario)
            
            ohe = OneHotEncoding(inte)

            ngram = Ngram([w2v], thread_aware, ngram_length + 1) 

            select = Select(ngram, start = 0, end = (w2v_vector_size * ngram_length)) 

            if args.use_independent_validation == 'True':
                mlp = MLP(select,
                    ohe,
                    hidden_size,
                    hidden_layers,
                    batch_size,
                    learning_rate,
                    True
                )
                
            # False-Fall
            else: 
                mlp = MLP(select,
                    ohe,
                    hidden_size,
                    hidden_layers,
                    batch_size,
                    learning_rate,
                    False    
                )
            
            decision_engine = StreamSum(mlp, thread_aware, window_length)
    
    ##################################### Config 2 ######################################### 
    elif args.config == '2':
            
            # Settings
            ngram_length = 7
            thread_aware = True
            hidden_size = 64
            hidden_layers = 3
            batch_size = 256
            learning_rate = 0.003
            window_length = 5
            
            settings_dict['ngram_length'] = ngram_length
            settings_dict['thread_aware'] = thread_aware
            settings_dict['hidden_size'] = hidden_size
            settings_dict['hidden_layers'] = hidden_layers
            settings_dict['batch_size'] = batch_size
            settings_dict['learning_rate'] = learning_rate
            settings_dict['window_length'] = window_length
            
            # Calculate Embedding_size
            temp_i = IntEmbedding()
            temp_ohe = OneHotEncoding(temp_i)
            mini_ids = IDS(dataloader, temp_ohe, False, False)
            ohe_embedding_size = temp_ohe.get_embedding_size()
            
            # Building Blocks
            inte = IntEmbedding()
            
            ohe = OneHotEncoding(inte)
            
            ngram_ohe = Ngram([ohe], thread_aware, ngram_length + 1)
            
            select_ohe = Select(ngram_ohe, 0, (ngram_length * ohe_embedding_size)) 
            
            if args.use_independent_validation == 'True':
                mlp = MLP(select_ohe,
                    ohe,
                    hidden_size,
                    hidden_layers,
                    batch_size,
                    learning_rate,
                    True
                )
                
            # False-Fall
            else: 
                mlp = MLP(select_ohe,
                    ohe,
                    hidden_size,
                    hidden_layers,
                    batch_size,
                    learning_rate,
                    False    
                )
    
            decision_engine = StreamSum(mlp, thread_aware, window_length)
        
    elif args.config == '3': 
            # Settings
            ngram_length = 5
            thread_aware = True
            hidden_size = 64
            hidden_layers = 4
            batch_size = 512
            learning_rate = 0.003
            window_length = 20
            
            settings_dict['ngram_length'] = ngram_length
            settings_dict['thread_aware'] = thread_aware
            settings_dict['hidden_size'] = hidden_size
            settings_dict['hidden_layers'] = hidden_layers
            settings_dict['batch_size'] = batch_size
            settings_dict['learning_rate'] = learning_rate
            settings_dict['window_length'] = window_length
            
            # Calculate Embedding_size
            temp_i = IntEmbedding()
            temp_ohe = OneHotEncoding(temp_i)
            mini_ids = IDS(dataloader, temp_ohe, False, False)
            ohe_embedding_size = temp_ohe.get_embedding_size()
            
            # Building Blocks
            inte = IntEmbedding()
            
            ohe = OneHotEncoding(inte)
            
            ngram_ohe = Ngram([ohe], thread_aware, ngram_length + 1)
            
            select_ohe = Select(ngram_ohe, 0, (ngram_length * ohe_embedding_size)) 
            
            if args.use_independent_validation == 'True':
                mlp = MLP(select_ohe,
                    ohe,
                    hidden_size,
                    hidden_layers,
                    batch_size,
                    learning_rate,
                    True
                )
                
            # False-Fall
            else: 
                mlp = MLP(select_ohe,
                    ohe,
                    hidden_size,
                    hidden_layers,
                    batch_size,
                    learning_rate,
                    False    
                )
    
            decision_engine = StreamSum(mlp, thread_aware, window_length)
    
    elif args.config == '4':
          # Settings
            ngram_length = 7
            w2v_vector_size = 8
            w2v_window_size = 20
            thread_aware = True
            hidden_size = 64
            hidden_layers = 2
            batch_size = 256
            w2v_epochs = 1000
            learning_rate = 0.003
            window_length = 40       
            
            
            settings_dict['ngram_length'] = ngram_length
            settings_dict['w2v_vector_size'] = w2v_vector_size
            settings_dict['w2v_window_size'] = w2v_window_size
            settings_dict['thread_aware'] = thread_aware
            settings_dict['hidden_size'] = hidden_size
            settings_dict['hidden_layers'] = hidden_layers
            settings_dict['batch_size'] = batch_size
            settings_dict['w2v_epochs'] = w2v_epochs
            settings_dict['learning_rate'] = learning_rate
            settings_dict['window_length'] = window_length
            
            
            # Building Blocks

            inte = IntEmbedding()

            w2v = W2VEmbedding(word=inte,
                           vector_size=w2v_vector_size,
                           window_size=w2v_window_size,
                           epochs=w2v_epochs,
                           thread_aware=thread_aware,
                           lid_ds_version=args.version,
                           lid_ds_scenario=args.scenario)
            
            ohe = OneHotEncoding(inte)

            ngram = Ngram([w2v], thread_aware, ngram_length + 1) 

            select = Select(ngram, start = 0, end = (w2v_vector_size * ngram_length)) 

            if args.use_independent_validation == 'True':
                mlp = MLP(select,
                    ohe,
                    hidden_size,
                    hidden_layers,
                    batch_size,
                    learning_rate,
                    True
                )
                
            # False-Fall
            else: 
                mlp = MLP(select,
                    ohe,
                    hidden_size,
                    hidden_layers,
                    batch_size,
                    learning_rate,
                    False    
                )
            
            decision_engine = StreamSum(mlp, thread_aware, window_length)   
        
    else:
        exit('Unknown configuration of MLP. Exiting.')
        
    
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
    
    # config_name = f"algorithm_mlp_c_{args.config}_i_{args.use_independent_validation}_lr_{args.learning_rate}_n_{ngram_length}_t_{thread_aware}"
    config_name = f"algorithm_mlp_c_{args.config}_lr_{args.learning_rate}_n_{ngram_length}_t_{thread_aware}"
    
    
    # Enrich results with configuration
    results['algorithm'] = 'mlp'
    for key in settings_dict.keys():
        results[key] = settings_dict[key]
        
    results['config'] = ids.get_config() # Produces strangely formatted Config-Print
    results['scenario'] =  args.version + "/" + args.scenario
    # result_path = f"{args.results}/results_mlp_config_{args.config}_i_{args.use_independent_validation}_lr_{args.learning_rate}_{args.version}_{args.scenario}.json"
    result_path = f"{args.results}/results_mlp_config_{args.config}_lr_{args.learning_rate}_{args.version}_{args.scenario}.json"

    # Saving results
    save_to_json(results, result_path) 
    with open(f"{args.results}/alarms_{config_name}_{args.version}_{args.scenario}.json", 'w') as jsonfile:
        json.dump(performance.alarms.get_alarms_as_dict(), jsonfile, default=str, indent=2)
        
    # ---------------------------------------------------------------------------------------------------------#    
        
    # Extracting Systemcalls from False Alarms
    false_alarm_list = [alarm for alarm in performance.alarms.alarm_list if not alarm.correct]
    
    # Stop all of this if we didn't found any false alarms. Empty lists are considered false.
    if not false_alarm_list:
        sys.exit('The decision engine didn\'t found any false alarms which it could play back. Stopping here.')
    
    # Collect all corresponding recordings 
    basename_recording_list = set([os.path.basename(false_alarm.filepath) for false_alarm in false_alarm_list])
    false_alarm_recording_list = [recording for recording in dataloader.test_data() if os.path.basename(recording.path) in basename_recording_list]
   
   
    containerList = []
    counter = 0
    for alarm in false_alarm_list: 
        if args.play_back_count_alarms != 'all' and counter == int(args.play_back_count_alarms):
            break
        containerList.append(FalseAlertContainer(alarm, false_alarm_recording_list, window_length, ngram_length, thread_aware))
        counter += 1    
    
    pprint("Playing back false positive alarms:")
    if sys.platform in ['win32', 'cygwin']:
        max_workers = 4 # Musste das begrenzen da mir sonst alles abschmierte
    else:
        max_workers = min(32, cpu_count() + 4)
    
    false_alarm_results = process_map(construct_Syscalls, containerList, chunksize = 50, max_workers=max_workers)
    final_playback = reduce(FalseAlertResult.add, false_alarm_results)
    
    
    # MODYFIABLE! Hier kann ich auch einstellen, nur einen Teil der False-Alarms ins Training zur??ckgehen zu lassen.
    all_recordings = []  
    counter = 0
    
    # Iteriere durch alle False-Alarms und nutze die jeweiligen SystemCalls. 
    for key in final_playback.structure.keys():
        new_recording = ArtificialRecording(key, final_playback.structure[key])
        all_recordings.append(new_recording)
        counter += 1
    
    if not all_recordings:
        exit(f'Percentage of {args.play_back_percentage} playing back alarms lead to playing back zero false alarms. Program stops.')
    
    
    
    
    if args.mode == 'retraining':
        dataloader.set_retraining_data(all_recordings) # F??gt die neuen Trainingsbeispiele als zus??tzliches Training ein.
    
        settings_dict = {} # Enth??lt die Konfig-Infos
        if args.config == '0':
                
                ##################################### Config 0 ######################################### 
                
                # Settings
                ngram_length = 5
                w2v_vector_size = 5
                w2v_window_size = 10
                thread_aware = True
                hidden_size = 64
                hidden_layers = 3
                batch_size = 256
                w2v_epochs = 1000
                learning_rate = 0.003
                window_length = 10

                settings_dict['ngram_length'] = ngram_length
                settings_dict['w2v_vector_size'] = w2v_vector_size
                settings_dict['w2v_window_size'] = w2v_window_size
                settings_dict['thread_aware'] = thread_aware
                settings_dict['hidden_size'] = hidden_size
                settings_dict['hidden_layers'] = hidden_layers
                settings_dict['batch_size'] = batch_size
                settings_dict['w2v_epochs'] = w2v_epochs
                settings_dict['learning_rate'] = learning_rate
                settings_dict['window_length'] = window_length

                # Building Blocks
                inte = IntEmbedding()

                w2v = W2VEmbedding(word=inte,
                            vector_size=w2v_vector_size,
                            window_size=w2v_window_size,
                            epochs=w2v_epochs,
                            thread_aware=thread_aware,
                            lid_ds_version=args.version,
                            lid_ds_scenario=args.scenario)
                
                ohe = OneHotEncoding(inte)

                ngram = Ngram([w2v], thread_aware, ngram_length + 1) 

                select = Select(ngram, start = 0, end = (w2v_vector_size * ngram_length)) 

                if args.use_independent_validation == 'True':
                    mlp = MLP(select,
                        ohe,
                        hidden_size,
                        hidden_layers,
                        batch_size,
                        learning_rate,
                        True
                    )
                
                # False-Fall
                else: 
                    mlp = MLP(select,
                        ohe,
                        hidden_size,
                        hidden_layers,
                        batch_size,
                        learning_rate,
                        False    
                    )
                
                decision_engine = StreamSum(mlp, thread_aware, window_length)
        
        ##################################### Config 1 ######################################### 
        elif args.config == '1':
                
                # Settings
                ngram_length = 3
                w2v_vector_size = 8
                w2v_window_size = 15
                thread_aware = True
                hidden_size = 32
                hidden_layers = 4
                batch_size = 256
                w2v_epochs = 1000
                learning_rate = 0.003
                window_length = 100       
                
                
                settings_dict['ngram_length'] = ngram_length
                settings_dict['w2v_vector_size'] = w2v_vector_size
                settings_dict['w2v_window_size'] = w2v_window_size
                settings_dict['thread_aware'] = thread_aware
                settings_dict['hidden_size'] = hidden_size
                settings_dict['hidden_layers'] = hidden_layers
                settings_dict['batch_size'] = batch_size
                settings_dict['w2v_epochs'] = w2v_epochs
                settings_dict['learning_rate'] = learning_rate
                settings_dict['window_length'] = window_length
                
                
                # Building Blocks

                inte = IntEmbedding()

                w2v = W2VEmbedding(word=inte,
                            vector_size=w2v_vector_size,
                            window_size=w2v_window_size,
                            epochs=w2v_epochs,
                            thread_aware=thread_aware,
                            lid_ds_version=args.version,
                            lid_ds_scenario=args.scenario)
                
                ohe = OneHotEncoding(inte)

                ngram = Ngram([w2v], thread_aware, ngram_length + 1) 

                select = Select(ngram, start = 0, end = (w2v_vector_size * ngram_length)) 

                if args.use_independent_validation == 'True':
                    mlp = MLP(select,
                        ohe,
                        hidden_size,
                        hidden_layers,
                        batch_size,
                        learning_rate,
                        True
                    )
                
                # False-Fall
                else: 
                    mlp = MLP(select,
                        ohe,
                        hidden_size,
                        hidden_layers,
                        batch_size,
                        learning_rate,
                        False    
                    )
                
                decision_engine = StreamSum(mlp, thread_aware, window_length)
        
        ##################################### Config 2 ######################################### 
        elif args.config == '2':
                
                # Settings
                ngram_length = 7
                thread_aware = True
                hidden_size = 64
                hidden_layers = 3
                batch_size = 256
                learning_rate = 0.003
                window_length = 5

                settings_dict['ngram_length'] = ngram_length
                settings_dict['thread_aware'] = thread_aware
                settings_dict['hidden_size'] = hidden_size
                settings_dict['hidden_layers'] = hidden_layers
                settings_dict['batch_size'] = batch_size
                settings_dict['learning_rate'] = learning_rate
                settings_dict['window_length'] = window_length
                
                # Calculate Embedding_size
                temp_i = IntEmbedding()
                temp_ohe = OneHotEncoding(temp_i)
                mini_ids = IDS(dataloader, temp_ohe, False, False)
                ohe_embedding_size = temp_ohe.get_embedding_size()

                # Building Blocks
                inte = IntEmbedding()
                
                # Benutze hier das alte OHE, da wir sonst Probleme in der L??nge des OHEs bekommen k??nnten. Das orientiert sich ja an den Trainingdaten.
                ohe = OneHotEncoding(inte)
                
                ngram_ohe = Ngram([ohe], thread_aware, ngram_length + 1)
                
                select_ohe = Select(ngram_ohe, 0, (ngram_length * ohe_embedding_size)) 
                
                if args.use_independent_validation == 'True':
                    mlp = MLP(select_ohe,
                        ohe,
                        hidden_size,
                        hidden_layers,
                        batch_size,
                        learning_rate,
                        True
                    )
                
                # False-Fall
                else: 
                    mlp = MLP(select_ohe,
                        ohe,
                        hidden_size,
                        hidden_layers,
                        batch_size,
                        learning_rate,
                        False    
                    )
        
                decision_engine = StreamSum(mlp, thread_aware, window_length)
            
        ##################################### Config 3 ######################################### 
        elif args.config == '3': 
            # Settings
                ngram_length = 5
                thread_aware = True
                hidden_size = 64
                hidden_layers = 4
                batch_size = 512
                learning_rate = 0.003
                window_length = 20

                settings_dict['ngram_length'] = ngram_length
                settings_dict['thread_aware'] = thread_aware
                settings_dict['hidden_size'] = hidden_size
                settings_dict['hidden_layers'] = hidden_layers
                settings_dict['batch_size'] = batch_size
                settings_dict['learning_rate'] = learning_rate
                settings_dict['window_length'] = window_length

                # Calculate Embedding_size
                temp_i = IntEmbedding()
                temp_ohe = OneHotEncoding(temp_i)
                mini_ids = IDS(dataloader, temp_ohe, False, False)
                ohe_embedding_size = temp_ohe.get_embedding_size()

                # Building Blocks
                inte = IntEmbedding()

                ohe = OneHotEncoding(inte)

                ngram_ohe = Ngram([ohe], thread_aware, ngram_length + 1)

                select_ohe = Select(ngram_ohe, 0, (ngram_length * ohe_embedding_size)) 
            
                if args.use_independent_validation == 'True':
                    mlp = MLP(select_ohe,
                        ohe,
                        hidden_size,
                        hidden_layers,
                        batch_size,
                        learning_rate,
                        True
                    )
                
                # False-Fall
                else: 
                    mlp = MLP(select_ohe,
                        ohe,
                        hidden_size,
                        hidden_layers,
                        batch_size,
                        learning_rate,
                        False    
                    )
                decision_engine = StreamSum(mlp, thread_aware, window_length)
    
        ##################################### Config 4 ######################################### 
        elif args.config == '4':
            # Settings
                ngram_length = 7
                w2v_vector_size = 8
                w2v_window_size = 20
                thread_aware = True
                hidden_size = 64
                hidden_layers = 2
                batch_size = 256
                w2v_epochs = 1000
                learning_rate = 0.003
                window_length = 40       


                settings_dict['ngram_length'] = ngram_length
                settings_dict['w2v_vector_size'] = w2v_vector_size
                settings_dict['w2v_window_size'] = w2v_window_size
                settings_dict['thread_aware'] = thread_aware
                settings_dict['hidden_size'] = hidden_size
                settings_dict['hidden_layers'] = hidden_layers
                settings_dict['batch_size'] = batch_size
                settings_dict['w2v_epochs'] = w2v_epochs
                settings_dict['learning_rate'] = learning_rate
                settings_dict['window_length'] = window_length


                # Building Blocks

                inte = IntEmbedding()

                w2v = W2VEmbedding(word=inte,
                           vector_size=w2v_vector_size,
                           window_size=w2v_window_size,
                           epochs=w2v_epochs,
                           thread_aware=thread_aware,
                           lid_ds_version=args.version,
                           lid_ds_scenario=args.scenario)
            
                ohe = OneHotEncoding(inte)

                ngram = Ngram([w2v], thread_aware, ngram_length + 1) 

                select = Select(ngram, start = 0, end = (w2v_vector_size * ngram_length)) 

                if args.use_independent_validation == 'True':
                    mlp = MLP(select,
                        ohe,
                        hidden_size,
                        hidden_layers,
                        batch_size,
                        learning_rate,
                        True
                    )
                
                # False-Fall
                else: 
                    mlp = MLP(select,
                        ohe,
                        hidden_size,
                        hidden_layers,
                        batch_size,
                        learning_rate,
                        False                   
                    )
                decision_engine = StreamSum(mlp, thread_aware, window_length) 
        
        else:
            exit('Unknown configuration of MLP. Exiting.')
    
        # Resetting seeds
        torch.manual_seed(0)
        random.seed(0)
        numpy.random.seed(0)    

        ######## New IDS ########################
        ids_retrained = IDS(data_loader=dataloader,
            resulting_building_block=decision_engine,
            plot_switch=False,
            create_alarms=True)
    
    elif args.mode == 'revalidation':
        dataloader.set_revalidation_data(all_recordings)

        # Keep the old IDS
        ids_retrained = ids
    
    elif args.mode == 'conceptdrift':
        
        # Hier das OHE freezen, damit es seine Gr????e nicht ver??ndern kann, sondern auch alle neuen Calls in den Extra-Slot daf??r legt.
        ohe.set_already_trained(True)
        
        # Set new LR
        mlp.set_learning_rate(args.learning_rate)
        # Overwrite training samples
        mlp._training_set = set()
        dataloader.set_retraining_data(all_recordings) # F??gt die neuen Trainingsbeispiele als zus??tzliches Training ein.
        dataloader.overwrite_training_data_with_retraining()
        
        decision_engine = StreamSum(mlp, thread_aware, window_length)
    
        ######## New IDS ########################
        ids_retrained = IDS(data_loader=dataloader,
            resulting_building_block=decision_engine,
            plot_switch=False,
            create_alarms=True)
    
    if args.mode == 'revalidation':
        pprint(f'Hopefully setting threshold to {performance.max_anomaly_score_fp }')  
        ids_retrained.determine_threshold()
        # ids_retrained.threshold = performance.max_anomaly_score_fp
        # ids_retrained.performance._threshold = performance.max_anomaly_score_fp
        dataloader.unload_revalidation_data()

    elif args.mode == 'retraining' or args.mode == 'conceptdrift':
        dataloader.unload_retraining_data()
        if args.freeze_on_retraining == 'True':
            pprint(f"Freezing Threshold on: {ids.threshold}")
            ids_retrained.threshold = ids.threshold
        else: 
            ids_retrained.determine_threshold()
    

    pprint("At evaluation:")
    performance_new = ids_retrained.detect_parallel()
    pprint(performance_new)        
    results_new = performance_new.get_results()
    pprint(results_new)

    # Preparing second results
    algorithm_name = 'mlp_retrained'
    # config_name = f"algorithm_{algorithm_name}_c_{args.config}_p_{args.play_back_count_alarms}_i_{args.use_independent_validation}_lr_{args.learning_rate}_n_{ngram_length}_w_{window_length}_t_{thread_aware}"
    config_name = f"algorithm_{algorithm_name}_c_{args.config}_p_{args.play_back_count_alarms}_m_{args.mode}_lr_{args.learning_rate}_n_{ngram_length}_w_{window_length}_t_{thread_aware}"

    # Enrich results with configuration 
    results_new['algorithm'] = algorithm_name
    results_new['play_back_count_alarms']= args.play_back_count_alarms
    
    for key in settings_dict.keys():
        results[key] = settings_dict[key]
        
    results_new['config'] = ids.get_config() # Produces strangely formatted Config-Print
    results_new['scenario'] =  args.version + "/" + args.scenario
    # result_new_path = f"{args.results}/results_{algorithm_name}_config_{args.config}_p_{args.play_back_count_alarms}_i_{args.use_independent_validation}_lr_{args.learning_rate}_{args.version}_{args.scenario}.json"
    result_new_path = f"{args.results}/results_{algorithm_name}_config_{args.config}_p_{args.play_back_count_alarms}_m_{args.mode}_lr_{args.learning_rate}_{args.version}_{args.scenario}.json"
    # Save results
    save_to_json(results_new, result_new_path) 
    with open(f"{args.results}/alarms_{config_name}_{args.version}_{args.scenario}.json", 'w') as jsonfile:
        json.dump(performance_new.alarms.get_alarms_as_dict(), jsonfile, default=str, indent=2)
   
   