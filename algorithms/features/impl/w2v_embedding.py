from gensim.models import KeyedVectors, Word2Vec
from algorithms.building_block import BuildingBlock
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.syscall_name import SyscallName
from dataloader.syscall import Syscall
from pprint import pprint
import os

class W2VEmbedding(BuildingBlock):
    """
        Args:
            word: BuildingBlock used as word in the sentences Word2Vec learns from.
            vector_size: the w2v output vector size
            window_size: size of w2v context window (the senctence size)
            epochs: number of epochs for 2v training                                    
            distinct: true if training dataset shall be distinct, gives tremendous increase in training speed
            thread_aware: true if training sentences shall be created thread aware
            unknown_input_value: value that gets set for every dimension if unknown input word is given to w2v model
    """

    def __init__(self,
                 word: BuildingBlock,
                 vector_size: int,
                 window_size: int,
                 epochs: int,
                 distinct: bool = True,
                 thread_aware=True,
                 unknown_input_value: float = 0.0,
                 lid_ds_version = None,
                 lid_ds_scenario = None):
        super().__init__()
        self._vector_size = vector_size
        self._epochs = epochs
        self._distinct = distinct
        self.w2vmodel = None
        self._sentences = []        
        self._window_size = window_size
        
        self._input_bb = word

        self._ngram_bb = Ngram(feature_list=[word],
                               thread_aware=thread_aware,
                               ngram_length=window_size)

        self._unknown_input_value = unknown_input_value
        self._dependency_list = [self._ngram_bb, self._input_bb]
        
        self._lid_ds_version = lid_ds_version
        self._lid_ds_scenario = lid_ds_scenario
        self._base_model_path = 'models/'
        
        # Try to load the model
        self.load_model()

    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall: Syscall):
        """
            gets training systemcalls one after another
            builds sentences(ngrams) from them 
            saves them to training corpus
        """
        if self.w2vmodel is None:
            ngram = self._ngram_bb.get_result(syscall)
            if ngram is not None:
                if self._distinct:
                    if ngram not in self._sentences:
                        self._sentences.append(ngram)
                else:
                    self._sentences.append(ngram)

    def fit(self):
        """
            trains the w2v model on training sentences
        """
        if not self.w2vmodel:
            print(f"w2v.train_set: {len(self._sentences)}".rjust(27))
            model = Word2Vec(sentences=self._sentences,
                             vector_size=self._vector_size,
                             epochs=self._epochs,
                             window=self._window_size,
                             min_count=1,
                             workers=1)
            self.w2vmodel = model
            self.save_model()

    def _calculate(self, syscall: Syscall):
        """
            returns the w2v embedding to a given input            
            if the input is not in the training corpus a pre-defined vector (see: unknown_input_value) is returned

            Returns:
                tuple representing the w2v embedding or None if no embedding can be calculated for the input
        """
        try:
            input = self._input_bb.get_result(syscall)
            if input is not None:
                return tuple(self.w2vmodel.wv[input].tolist())
            else: 
                return None
        except KeyError:
            return tuple([self._unknown_input_value] * self._vector_size)

    def save_model(self): # Eine angepasste Exception wäre gut. TODO
        # Verschiedene Guards die sicherstellen, dass das Modell richtig abgespeichert wird
        if self.w2vmodel is None:
            pprint('Couldn\'t save the model because w2vmodel is None!')
            return
        if self._lid_ds_version is None:
            pprint('Couldn\'t save the model because no lid_ds_version was given!')
            return
        if self._lid_ds_scenario is None:
            pprint('Couldn\'t save the model because no lid_ds_scenario was given!')
            return
        
        ### Create folders if not exists ###
        # Model
        if not os.path.exists(self._base_model_path):
            os.mkdir(self._base_model_path)        
        # Version
        if not os.path.exists(f'{self._base_model_path}/{self._lid_ds_version}'):
            os.mkdir(f'{self._base_model_path}/{self._lid_ds_version}')  
        # Scenario
        if not os.path.exists(f'{self._base_model_path}/{self._lid_ds_version}/{self._lid_ds_scenario}'):
            os.mkdir(f'{self._base_model_path}/{self._lid_ds_version}/{self._lid_ds_scenario}')  
        
        filename = None
        if type(self._input_bb) == type(SyscallName()):
           filename = 'syscall'
        elif type(self._input_bb) == type(IntEmbedding()):
            filename = 'intembedding'
        else: 
            pprint(f'Type of {type(self._input_bb)} not handled. Can\'t save this model.')
            return
        
        # Save modell
        self.w2vmodel.save(f'{self._base_model_path}/{self._lid_ds_version}/{self._lid_ds_scenario}/{filename}_vs_{self._vector_size}_ws_{self._window_size}_e_{self._epochs}_word2vec.modell')
        
    def load_model(self):
        # Check for type and resulting name of model file
        filename = None
        if type(self._input_bb) == type(SyscallName()):
           filename = 'syscall'
        elif type(self._input_bb) == type(IntEmbedding()):
            filename = 'intembedding'
        else: 
            pprint(f'Type of {type(self._input_bb)} not handled. Can\'t load this model.')
            return
        
        # Load model
        try:
            self.w2vmodel = Word2Vec.load(f'{self._base_model_path}/{self._lid_ds_version}/{self._lid_ds_scenario}/{filename}_vs_{self._vector_size}_ws_{self._window_size}_e_{self._epochs}_word2vec.modell')
            pprint('Found w2v model file.')
        except AttributeError:
            pprint('Couldn\'t load the s2v model. We have an attribute error, so maybe there is a problem with the version.')
        except FileNotFoundError:
            pprint('Couldn\'t find a w2v model file.')
        except Exception:
            pprint('Unhandled exception occured whilel loading the w2v model.')