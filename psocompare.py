import os
import sys
import re
import getopt
import threading
import time
import src.optimization as optim


N_RUNS = 4


def process_paramfiles(paramfiles: str):
    # Check if the pattern is valid
    valid = re.match(r'^[0-9]+(-[0-9]+)?(,\s*[0-9]+(-[0-9]+)?)*$', paramfiles)
    if not bool(valid):
        print('Invalid arguments.')
        sys.exit(2)

    # Find all N-M patterns
    fromto_patterns = re.findall(r'[0-9]+-[0-9]+', paramfiles)

    for pattern in fromto_patterns:
        # Get first (N) and last (M) numbers from N-M pattern
        first, last = pattern.split('-')
        first = int(first)
        last = int(last)
        # Replace the N-M pattern for comma separated values
        comma_separated = str([i for i in range(first, last+1)])[1:-1]
        paramfiles = paramfiles.replace(pattern, comma_separated)

    params_numbers = set(
        int(n) for n in re.split(r',\s*', paramfiles)
    )
    # Name of parameters files
    return [f'params{p}.yml' for p in sorted(params_numbers)]


def process_algorithms(algorithms: str):
    algorithms = algorithms.lower()
    valid = re.match(r'^[a-z]+(,\s*[a-z]+)*$', algorithms)
    if not bool(valid):
        print('Invalid arguments.')
        sys.exit(2)
    return set(re.split(r',\s*', algorithms))


def main(argv):
    paramsfiles = []
    algorithms = []
    start_from_checkpoint = False

    try:
        opts, _ = getopt.getopt(
            argv, "hp:a:c", ["paramfiles=", "algorithms=", "checkpoint"])
    except getopt.GetoptError:
        # Print debug info
        print('Invalid arguments.')
        sys.exit(2)

    for option, argument in opts:
        # if option in ('-h', '--help'):
        #     print('AJUDA')
        if option in ('-p', '--paramfiles'):
            paramsfiles = process_paramfiles(argument)

        elif option in ('-a', '--algorithms'):
            algorithms = process_algorithms(argument)

        elif option in ('-c', '--checkpoint'):
            start_from_checkpoint = True

    if len(paramsfiles) > 0 and len(algorithms) > 0:
        optim.run_algorithms(algorithms, paramsfiles, start_from_checkpoint, N_RUNS)
    else:
        print('Invalida arguments.')
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])
