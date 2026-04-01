import logging
import subprocess
from time import sleep

_logger = logging.getLogger(__name__)

def subprocess_with_retry(check_output,
                          arg_list,
                          log_information,
                          number_of_retries=5,
                          timeout_seconds=5,
                          ):
    """Execute subprocess call with retry
    """
    output = None
    attempts = 0
    while attempts < number_of_retries:
        try:
            if check_output:
                output = subprocess.check_output(arg_list)
            else:
                subprocess.check_call(arg_list)
        except subprocess.CalledProcessError as e:
            _logger.warning(e, e.output)
            attempts += 1
            sleep(timeout_seconds)
            if attempts == number_of_retries:
                _logger.error(f"Connection failed for {log_information}.")
        else:
            _logger.info(f"Succesfully got {log_information}.")
            return output

    raise subprocess.SubprocessError(f"Failed to obtain {log_information} after {number_of_retries} retries.")
