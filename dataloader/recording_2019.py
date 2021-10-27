import datetime
import os
from dataloader.syscall_2019 import Syscall
from distutils.util import strtobool

from enum import IntEnum

from typing import Generator


class RecordingDataParts(IntEnum):
    IMAGE_NAME = 0
    RECORDING_NAME = 1
    IS_EXECUTING_EXPLOIT = 2
    WARMUP_TIME = 3
    RECORDING_TIME = 4
    EXPLOIT_START_TIME = 5


class Recording:
    """

    Single Recording built out of one line from runs.csv of LID-DS 2019

    Args:
        recording_data_list (list): runs.csv line as list
        base_path (str): the base path of the LID-DS 2019 scenario

    """
    def __init__(self, recording_data_list: list, base_path: str):
        self.name = recording_data_list[RecordingDataParts.RECORDING_NAME]
        self.path = os.path.join(base_path, f'{self.name}.txt')
        self.recording_data_list = recording_data_list
        self._metadata = self._collect_metadata()
        self.name = self._metadata['name']


    def syscalls(self) -> Generator[Syscall, None, None]:
        """

            Prepare stream of syscalls,
            yield single lines

            Returns:
            str: syscall text line

        """
        with open(self.path, 'r') as recording_file:
            for syscall in recording_file:
                yield Syscall(syscall)

    def _collect_metadata(self):
        """

            transfers metadata from csv line to same same dict format as from LID-DS 2021

        """
        is_exploit = strtobool(self.recording_data_list[RecordingDataParts.IS_EXECUTING_EXPLOIT].lower())
        return {
            'image': self.recording_data_list[RecordingDataParts.IMAGE_NAME],
            'name': self.name,
            'exploit': is_exploit,
            'recording_time': int(self.recording_data_list[RecordingDataParts.RECORDING_TIME]),
            'time': {
                'exploit': [{
                    'absolute': self._calc_absolute_exploit_time() if is_exploit else None,
                    'relative': int(self.recording_data_list[RecordingDataParts.EXPLOIT_START_TIME]) if is_exploit else None
                }],
                'warmup_end': {
                    'relative': {
                        'relative': int(self.recording_data_list[RecordingDataParts.WARMUP_TIME])
                    }
                }
            }
        }

    def metadata(self) -> dict:
        return self._metadata

    def _calc_absolute_exploit_time(self):
        """

            creates missing absolute timestamp from LID-DS 2019 metadata

        """
        syscall_generator = self.syscalls()
        first_syscall_timestamp = next(syscall_generator).timestamp_datetime()

        # subtracting 2 seconds because of bad precision of relative timestamp in LID-DS 2019
        relative_time = int(self.recording_data_list[RecordingDataParts.WARMUP_TIME]) - 2
        absolute_time = first_syscall_timestamp + datetime.timedelta(seconds=relative_time)

        return absolute_time








